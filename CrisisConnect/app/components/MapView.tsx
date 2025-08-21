import mapboxgl from "mapbox-gl";
import { useEffect, useRef, useState } from "react";
import { AlertCircle, Zap, Droplet, Flame, Wind } from 'lucide-react';

// Ensure mapbox-gl doesn't break SSR
if (typeof window !== 'undefined') {
  mapboxgl.workerClass = require('worker-loader!mapbox-gl/dist/mapbox-gl-csp-worker').default;
}

// Risk location type
interface RiskLocation {
  location: string;
  lat: number;
  lon: number;
  composite_risk_score: number;
  risk_category: string;
  hazard_type?: string;
  wave_height?: number;
  precip_mm?: number;
  wind_kph?: number;
}

// Hazard icons
const HazardIcon = ({ type }: { type: string }) => {
  switch (type.toLowerCase()) {
    case 'flood': case 'flash flood': return <Droplet className="text-blue-600" size={16} />;
    case 'storm': case 'coastal_cyclone': return <Wind className="text-gray-600" size={16} />;
    case 'heatwave': case 'wildfire risk': return <Flame className="text-red-600" size={16} />;
    default: return <AlertCircle className="text-orange-600" size={16} />;
  }
};

export default function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const [riskData, setRiskData] = useState<RiskLocation[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch risk data
  useEffect(() => {
    const fetchRisk = async () => {
      try {
        const res = await fetch("http://localhost:8000/risk-assessment");
        if (!res.ok) throw new Error("Failed to fetch risk data");
        const data: RiskLocation[] = await res.json();
        setRiskData(data);
      } catch (err) {
        console.error("Fetch risk failed:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchRisk();
    const interval = setInterval(fetchRisk, 300000); // Refresh every 5 mins
    return () => clearInterval(interval);
  }, []);

  // Initialize Mapbox
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    mapboxgl.accessToken = process.env.VITE_MAPBOX_TOKEN;

    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/streets-v11",
      center: [28.0473, -26.2041], // South Africa
      zoom: 5,
    });

    mapRef.current = map;

    // Add navigation controls
    map.addControl(new mapboxgl.NavigationControl(), "top-right");

    // Add markers when data loads
    const updateMarkers = () => {
      // Remove existing markers
      const existing = document.querySelectorAll('.risk-marker');
      existing.forEach(el => el.remove());

      riskData.forEach(loc => {
        if (!loc.lat || !loc.lon) return;

        const el = document.createElement('div');
        el.className = 'risk-marker';
        el.style.width = '16px';
        el.style.height = '16px';
        el.style.borderRadius = '50%';
        el.style.backgroundColor = loc.composite_risk_score > 70
          ? '#ef4444'  // red-500
          : loc.composite_risk_score > 40
            ? '#f97316'  // orange-500
            : '#22c55e'; // green-500
        el.style.boxShadow = '0 0 0 2px white, 0 0 0 4px rgba(0,0,0,0.2)';
        el.style.cursor = 'pointer';
        el.style.transform = 'translate(-50%, -50%)';

        // Add tooltip on hover
        el.title = `${loc.location} - ${loc.risk_category} Risk (${loc.composite_risk_score.toFixed(1)}%)`;

        el.addEventListener('click', () => {
          new mapboxgl.Popup()
            .setLngLat([loc.lon, loc.lat])
            .setHTML(`
              <div style="font-family: sans-serif; line-height: 1.5">
                <strong>${loc.location}</strong><br/>
                <strong>Hazard:</strong> ${loc.hazard_type || 'Weather Risk'}<br/>
                <strong>Risk:</strong> <span style="color: ${
                  loc.composite_risk_score > 70 ? 'red' : 
                  loc.composite_risk_score > 40 ? 'orange' : 'green'
                }">${loc.risk_category}</span><br/>
                <strong>Score:</strong> ${loc.composite_risk_score.toFixed(1)}%<br/>
                ${loc.precip_mm ? `<strong>Rain (24h):</strong> ${loc.precip_mm} mm<br/>` : ''}
                ${loc.wind_kph ? `<strong>Wind:</strong> ${loc.wind_kph} km/h<br/>` : ''}
                ${loc.wave_height ? `<strong>Wave:</strong> ${loc.wave_height} m<br/>` : ''}
              </div>
            `)
            .addTo(map);
        });

        new mapboxgl.Marker({ element: el })
          .setLngLat([loc.lon, loc.lat])
          .addTo(map);
      });
    };

    // Update markers when riskData changes
    if (!loading) updateMarkers();

    // Cleanup
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [riskData, loading]);

  // Update markers when riskData changes
  useEffect(() => {
    if (mapRef.current && !loading) {
      // Re-run marker update
      const update = () => {
        const existing = document.querySelectorAll('.risk-marker');
        existing.forEach(el => el.remove());
        riskData.forEach(loc => {
          if (!loc.lat || !loc.lon) return;
          const el = document.createElement('div');
          el.className = 'risk-marker';
          el.style.width = '16px';
          el.style.height = '16px';
          el.style.borderRadius = '50%';
          el.style.backgroundColor = loc.composite_risk_score > 70 ? '#ef4444' : loc.composite_risk_score > 40 ? '#f97316' : '#22c55e';
          el.style.boxShadow = '0 0 0 2px white, 0 0 0 4px rgba(0,0,0,0.2)';
          el.style.cursor = 'pointer';
          el.style.transform = 'translate(-50%, -50%)';
          el.title = `${loc.location} - ${loc.risk_category} Risk (${loc.composite_risk_score.toFixed(1)}%)`;

          el.addEventListener('click', () => {
            new mapboxgl.Popup()
              .setLngLat([loc.lon, loc.lat])
              .setHTML(`<strong>${loc.location}</strong><br/>Risk: ${loc.risk_category} (${loc.composite_risk_score.toFixed(1)}%)`)
              .addTo(mapRef.current!);
          });

          new mapboxgl.Marker({ element: el }).setLngLat([loc.lon, loc.lat]).addTo(mapRef.current!);
        });
      };
      update();
    }
  }, [riskData, loading]);

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-black bg-opacity-10">
          <div className="bg-white p-3 rounded shadow text-sm">Loading risk data...</div>
        </div>
      )}
      <div className="w-full h-[500px] rounded-lg overflow-hidden" ref={mapContainer} />
    </div>
  );
}
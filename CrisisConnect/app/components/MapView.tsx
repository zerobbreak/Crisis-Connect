import mapboxgl from "mapbox-gl";
import { useEffect, useRef } from "react";
import { AlertCircle } from 'lucide-react';


export default function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const markerRef = useRef<mapboxgl.Marker | null>(null);

  useEffect(() => {
    if (!mapContainer.current) return;

    //force call 
    console.log("TOKEN:", import.meta.env.VITE_MAPBOX_TOKEN);


    // Set the token from .env file

    mapboxgl.accessToken = "pk.eyJ1IjoiaXRzY2xpZGUiLCJhIjoiY21lZDk5a2RzMDdnMzJsczg4OXZ0aWN4cSJ9.GAdXJR0g-SJAu0zSwbc_TQ";

   // mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;


    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/streets-v11",
      center: [28.0473, -26.2041], // Johannesburg
      zoom: 10,
    });

    // Show user's preferred location as blue dot if available
    try {
      const storedLoc = localStorage.getItem('preferredLocation');
      if (storedLoc) {
        const { lat, lng } = JSON.parse(storedLoc);
        if (typeof lat === 'number' && typeof lng === 'number') {
          const el = document.createElement('div');
          el.style.width = '12px';
          el.style.height = '12px';
          el.style.borderRadius = '50%';
          el.style.backgroundColor = '#2563eb'; // blue-600
          el.style.boxShadow = '0 0 0 2px white';
          const marker = new mapboxgl.Marker({ element: el }).setLngLat([lng, lat]).addTo(map);
          markerRef.current = marker;
          map.setCenter([lng, lat]);
        }
      }
    } catch {}

    return () => {
      if (markerRef.current) {
        markerRef.current.remove();
        markerRef.current = null;
      }
      map.remove();
    };
  }, []);

  return <div className="w-full h-[500px]" ref={mapContainer} />;
}

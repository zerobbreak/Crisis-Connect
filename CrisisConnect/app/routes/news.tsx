import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

type NewsItem = {
  id: string;
  title: string;
  source: string;
  region: string;
  timestamp: string;
  summary: string;
  url?: string;
};

export default function NewsPage() {
  const [items, setItems] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);

  useEffect(() => {
    // Placeholder: Replace with real API call later
    const demo: NewsItem[] = [
      {
        id: "1",
        title: "Severe Flooding in KwaZulu-Natal",
        source: "SA Weather Service",
        region: "South Africa • KZN",
        timestamp: new Date().toLocaleString(),
        summary: "Heavy rains causing localized flooding in low-lying areas. Evacuations underway in several districts.",
        url: "https://www.weathersa.co.za/",
      },
      {
        id: "2",
        title: "Wildfire Risk Elevated in Western Cape",
        source: "Provincial Disaster Management",
        region: "South Africa • Western Cape",
        timestamp: new Date().toLocaleString(),
        summary: "High temperatures and strong winds increase fire danger. Authorities urge caution and report any smoke sightings.",
      },
      {
        id: "3",
        title: "Cyclone Watch for Mozambique Channel",
        source: "Regional Met Center",
        region: "Mozambique Channel",
        timestamp: new Date().toLocaleString(),
        summary: "Tropical disturbance showing signs of intensification. Coastal communities advised to monitor updates closely.",
      },
      {
        id: "4",
        title: "Severe Thunderstorm Outlook for Gauteng",
        source: "SA Weather Service",
        region: "South Africa • Gauteng",
        timestamp: new Date().toLocaleString(),
        summary: "Potential for hail and strong winds this afternoon. Secure outdoor items and avoid flooded roads.",
      },
      {
        id: "5",
        title: "River Level Advisory in Limpopo",
        source: "GDACS",
        region: "South Africa • Limpopo",
        timestamp: new Date().toLocaleString(),
        summary: "River levels rising after upstream releases. Monitor official advisories and stay clear of riverbanks.",
      },
      {
        id: "6",
        title: "Atlantic Storm Bringing Heavy Surf",
        source: "NOAA",
        region: "Namibia • Coastal",
        timestamp: new Date().toLocaleString(),
        summary: "Large swells expected along the coast. Mariners should exercise caution and small craft should remain in port.",
      },
      {
        id: "7",
        title: "Heatwave Conditions Persist",
        source: "Met Office UK",
        region: "Botswana • Central",
        timestamp: new Date().toLocaleString(),
        summary: "Daytime temperatures exceeding 40°C. Stay hydrated and avoid prolonged exposure to the sun.",
      },
      {
        id: "8",
        title: "Monsoon Rains Intensify",
        source: "AccuWeather",
        region: "Mozambique • Northern Provinces",
        timestamp: new Date().toLocaleString(),
        summary: "Widespread rainfall causing transport disruptions. Expect delays and localized flooding.",
      },
    ];
    setItems(demo);
    setLoading(false);
    setSelectedSources(Array.from(new Set(demo.map(d => d.source))));
  }, []);

  const allSources = useMemo(() => Array.from(new Set(items.map(i => i.source))), [items]);
  const filtered = useMemo(
    () => items.filter(i => selectedSources.includes(i.source)),
    [items, selectedSources]
  );

  const toggleSource = (src: string) => {
    setSelectedSources(prev => prev.includes(src) ? prev.filter(s => s !== src) : [...prev, src]);
  };

  return (
    <div className="relative min-h-screen bg-gray-50 p-6 overflow-hidden">
      {/* Rotating globe background */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-20">
        <div
          className="rounded-full animate-spin"
          style={{
            width: 700,
            height: 700,
            animationDuration: '60s',
            backgroundImage: `radial-gradient(circle at 50% 50%, #cfe8ff 0%, #eaf3ff 40%, transparent 41%),
              repeating-conic-gradient(from 0deg, rgba(59,130,246,0.2) 0deg 10deg, rgba(59,130,246,0.05) 10deg 20deg)`,
            border: '1px solid rgba(59,130,246,0.2)'
          }}
        />
      </div>

      <div className="relative max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-gray-900">Crisis & Weather News</h1>
          <Link to="/home" className="text-sm text-blue-600 hover:text-blue-700">Back to Home</Link>
        </div>

        <div className="bg-white/90 backdrop-blur border border-gray-200 rounded-xl shadow-sm">
          <div className="p-4 border-b border-gray-100 space-y-3">
            <div className="text-sm text-gray-600">Live updates across provinces and neighboring countries</div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="text-xs text-gray-500">Sources:</div>
              {allSources.map((s) => (
                <label key={s} className={`flex items-center gap-2 text-sm px-2 py-1 rounded border ${selectedSources.includes(s) ? 'bg-blue-50 border-blue-300 text-blue-700' : 'border-gray-200 text-gray-700'}`}>
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={selectedSources.includes(s)}
                    onChange={() => toggleSource(s)}
                  />
                  <span>{s}</span>
                </label>
              ))}
              <div className="ml-auto flex items-center gap-2">
                <button
                  onClick={() => setSelectedSources(allSources)}
                  className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200"
                >
                  Select All
                </button>
                <button
                  onClick={() => setSelectedSources([])}
                  className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200"
                >
                  Clear All
                </button>
              </div>
            </div>
          </div>

          {loading ? (
            <div className="p-6 text-gray-500">Loading updates…</div>
          ) : filtered.length === 0 ? (
            <div className="p-6 text-gray-500">No news available.</div>
          ) : (
            <ul className="divide-y divide-gray-100">
              {filtered.map((n) => (
                <li key={n.id} className="p-4 hover:bg-gray-50">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-semibold text-gray-900">{n.title}</h3>
                      <div className="text-xs text-gray-500 mt-1">
                        {n.region} • {n.source} • {n.timestamp}
                      </div>
                    </div>
                    {n.url && (
                      <a
                        href={n.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-sm text-blue-600 hover:text-blue-700"
                      >
                        Source
                      </a>
                    )}
                  </div>
                  <p className="text-sm text-gray-700 mt-2">{n.summary}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}



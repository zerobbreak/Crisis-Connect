import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

type SentAlertMeta = {
  id: string;
  createdAt: string | Date;
  location?: { lat: number; lng: number; radius: number; address: string };
  severity: string;
  riskScore: number;
};

export default function StatisticsPage() {
  const [alerts, setAlerts] = useState<SentAlertMeta[]>([]);
  const [weather, setWeather] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    try {
      const existing = localStorage.getItem('sentAlerts');
      const list = existing ? JSON.parse(existing) : [];
      setAlerts(list);
    } catch {}
  }, []);

  const lastLocation = useMemo(() => {
    if (!alerts.length) return null;
    return alerts.find(a => a.location)?.location || null;
  }, [alerts]);

  useEffect(() => {
    const apiKey = import.meta.env.VITE_OPENWEATHER_API_KEY;
    if (!apiKey || !lastLocation) return;
    setLoading(true);
    setError(null);
    fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lastLocation.lat}&lon=${lastLocation.lng}&appid=${apiKey}&units=metric`)
      .then(r => r.ok ? r.json() : Promise.reject('Failed to fetch weather'))
      .then(setWeather)
      .catch((e) => setError(typeof e === 'string' ? e : 'Weather fetch failed'))
      .finally(() => setLoading(false));
  }, [lastLocation]);

  const totals = useMemo(() => {
    const total = alerts.length;
    const bySeverity: Record<string, number> = {};
    let avgRisk = 0;
    alerts.forEach(a => {
      bySeverity[a.severity] = (bySeverity[a.severity] || 0) + 1;
      avgRisk += a.riskScore;
    });
    avgRisk = total ? Math.round(avgRisk / total) : 0;
    return { total, bySeverity, avgRisk };
  }, [alerts]);

  // Build simple time series of alerts per day (last 14 days)
  const trend = useMemo(() => {
    const map: Record<string, number> = {};
    alerts.forEach(a => {
      const d = new Date(a.createdAt);
      const key = `${d.getFullYear()}-${(d.getMonth()+1).toString().padStart(2,'0')}-${d.getDate().toString().padStart(2,'0')}`;
      map[key] = (map[key] || 0) + 1;
    });
    const days: { label: string; value: number }[] = [];
    const now = new Date();
    for (let i = 13; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(now.getDate() - i);
      const key = `${d.getFullYear()}-${(d.getMonth()+1).toString().padStart(2,'0')}-${d.getDate().toString().padStart(2,'0')}`;
      days.push({ label: key.slice(5), value: map[key] || 0 });
    }
    return days;
  }, [alerts]);

  // Risk buckets for donut
  const riskBuckets = useMemo(() => {
    const buckets = { low: 0, medium: 0, high: 0, critical: 0 };
    alerts.forEach(a => {
      if (a.riskScore <= 25) buckets.low++;
      else if (a.riskScore <= 50) buckets.medium++;
      else if (a.riskScore <= 75) buckets.high++;
      else buckets.critical++;
    });
    return buckets;
  }, [alerts]);

  // Chart helpers (pure SVG)
  const SimpleBarChart = ({ data }: { data: { label: string; value: number }[] }) => {
    const width = 560;
    const height = 220;
    const padding = 32;
    const max = Math.max(1, ...data.map(d => d.value));
    const barW = (width - padding * 2) / data.length;
    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        <rect x={0} y={0} width={width} height={height} fill="#ffffff" />
        {data.map((d, i) => {
          const h = ((height - padding * 2) * d.value) / max;
          const x = padding + i * barW + 6;
          const y = height - padding - h;
          return (
            <g key={d.label}>
              <rect x={x} y={y} width={barW - 12} height={h} rx={4} fill="#60a5fa" />
              <text x={x + (barW - 12) / 2} y={height - padding + 14} textAnchor="middle" fontSize="8" fill="#6b7280">
                {d.label}
              </text>
              <text x={x + (barW - 12) / 2} y={y - 4} textAnchor="middle" fontSize="10" fill="#374151">
                {d.value}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  const SimpleLineChart = ({ data }: { data: { label: string; value: number }[] }) => {
    const width = 560;
    const height = 220;
    const padding = 32;
    const max = Math.max(1, ...data.map(d => d.value));
    const stepX = (width - padding * 2) / Math.max(1, data.length - 1);
    const points = data.map((d, i) => {
      const x = padding + i * stepX;
      const y = height - padding - ((height - padding * 2) * d.value) / max;
      return `${x},${y}`;
    }).join(' ');
    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        <polyline fill="none" stroke="#93c5fd" strokeWidth="2" points={points} />
        {data.map((d, i) => {
          const x = padding + i * stepX;
          const y = height - padding - ((height - padding * 2) * d.value) / max;
          return <circle key={i} cx={x} cy={y} r={3} fill="#2563eb" />;
        })}
      </svg>
    );
  };

  const DonutChart = ({ buckets }: { buckets: { low: number; medium: number; high: number; critical: number } }) => {
    const width = 240;
    const height = 240;
    const r = 90;
    const cx = width / 2;
    const cy = height / 2;
    const total = Object.values(buckets).reduce((a, b) => a + b, 0) || 1;
    const segments: { color: string; value: number; label: string }[] = [
      { color: '#10b981', value: buckets.low, label: '≤25' },
      { color: '#f59e0b', value: buckets.medium, label: '26–50' },
      { color: '#f97316', value: buckets.high, label: '51–75' },
      { color: '#ef4444', value: buckets.critical, label: '76–100' },
    ];
    let angle = -Math.PI / 2;

    const arcs = segments.map((s, i) => {
      const slice = (s.value / total) * Math.PI * 2;
      const x1 = cx + r * Math.cos(angle);
      const y1 = cy + r * Math.sin(angle);
      angle += slice;
      const x2 = cx + r * Math.cos(angle);
      const y2 = cy + r * Math.sin(angle);
      const largeArc = slice > Math.PI ? 1 : 0;
      const d = `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`;
      return <path key={i} d={d} fill={s.color} opacity={0.85} />;
    });

    return (
      <div className="flex items-center gap-4">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-60 h-60">
          <g>{arcs}</g>
          <circle cx={cx} cy={cy} r={50} fill="#fff" />
          <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle" className="text-gray-800" fontSize="14">
            Risk
          </text>
        </svg>
        <div className="text-sm text-gray-700 space-y-1">
          <div className="flex items-center gap-2"><span className="w-3 h-3 inline-block rounded" style={{background:'#10b981'}}></span> ≤25: {buckets.low}</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 inline-block rounded" style={{background:'#f59e0b'}}></span> 26–50: {buckets.medium}</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 inline-block rounded" style={{background:'#f97316'}}></span> 51–75: {buckets.high}</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 inline-block rounded" style={{background:'#ef4444'}}></span> 76–100: {buckets.critical}</div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">Statistics</h1>
          <Link to="/home" className="text-sm text-blue-600 hover:text-blue-700">Back to Home</Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white border rounded-xl p-4">
            <div className="text-sm text-gray-500">Total Alerts Sent</div>
            <div className="text-3xl font-semibold text-gray-800 mt-1">{totals.total}</div>
          </div>
          <div className="bg-white border rounded-xl p-4">
            <div className="text-sm text-gray-500">Average Risk Score</div>
            <div className="text-3xl font-semibold text-gray-800 mt-1">{totals.avgRisk}</div>
          </div>
          <div className="bg-white border rounded-xl p-4">
            <div className="text-sm text-gray-500">Latest Area</div>
            <div className="text-sm text-gray-800 mt-1">{lastLocation?.address || '—'}</div>
          </div>
        </div>

        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">By Severity</h2>
          {Object.keys(totals.bySeverity).length === 0 ? (
            <div className="text-sm text-gray-500">No data yet</div>
          ) : (
            <SimpleBarChart
              data={Object.entries(totals.bySeverity).map(([label, value]) => ({ label, value }))}
            />
          )}
        </div>

        <div className="bg-white border rounded-xl p-4">
          <h2 className="font-semibold mb-2">Current Weather</h2>
          {!lastLocation && <div className="text-sm text-gray-500">No recent alert with location.</div>}
          {lastLocation && loading && <div className="text-sm text-gray-500">Loading…</div>}
          {error && <div className="text-sm text-red-600">{error}</div>}
          {lastLocation && weather && !loading && !error && (
            <div className="text-sm text-gray-700 space-y-1">
              <div className="text-lg font-medium">{weather.name || lastLocation.address}</div>
              <div>Temp: {Math.round(weather.main?.temp)}°C, Feels: {Math.round(weather.main?.feels_like)}°C</div>
              <div>Condition: {weather.weather?.[0]?.main} ({weather.weather?.[0]?.description})</div>
              <div>Humidity: {weather.main?.humidity}% • Wind: {Math.round(weather.wind?.speed)} m/s</div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-white border rounded-xl p-4">
            <h2 className="font-semibold mb-2">Alerts in Last 14 Days</h2>
            <SimpleLineChart data={trend} />
          </div>
          <div className="bg-white border rounded-xl p-4">
            <h2 className="font-semibold mb-2">Risk Distribution</h2>
            <DonutChart buckets={riskBuckets} />
          </div>
        </div>
      </div>
    </div>
  );
}



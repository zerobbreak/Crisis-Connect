import { useState } from 'react';
import { useNavigate } from 'react-router';
import MapView from '../components/MapView';
import AlertForm from '../components/AlertForm';
import TopNav from '../components/TopNav';





export default function Dashboard()
 {
  const [alerts, setAlerts] = useState<string[]>([]);
  const navigate = useNavigate();
  const [lastAlert, setLastAlert] = useState<{
    message: string;
    severity: string;
    riskScore: number;
    languages: string[];
    channels: string[];
    recipients: string;
    location: { lat: number; lng: number; radius: number; address: string };
    createdAt: Date;
  } | null>(null);

  const [weather, setWeather] = useState<any>(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [weatherError, setWeatherError] = useState<string | null>(null);

  async function fetchWeather(lat: number, lon: number) {
    const apiKey = import.meta.env.VITE_OPENWEATHER_API_KEY;
    if (!apiKey) {
      setWeatherError('Weather API key not configured. Set VITE_OPENWEATHER_API_KEY.');
      return;
    }
    setWeatherLoading(true);
    setWeatherError(null);
    try {
      const res = await fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`);
      if (!res.ok) throw new Error('Failed to fetch weather');
      const data = await res.json();
      setWeather(data);
    } catch (e: any) {
      setWeatherError(e.message || 'Weather fetch failed');
    } finally {
      setWeatherLoading(false);
    }
  }


  return (
    <div className="min-h-screen grid grid-rows-[auto_1fr]">
      <header className="flex items-center justify-between p-4 bg-white shadow">
        <h1 className="text-lg font-semibold">CrisisConnect Admin</h1>
        <button
          onClick={() => {
            localStorage.removeItem('token');
            navigate('/login');
          }}
          className="px-3 py-1 bg-gray-100 rounded"
        >
          Logout
        </button>
      </header>
      <TopNav />
      <main className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-4">
        <section className="col-span-2 bg-white rounded shadow p-2">
          <MapView />
        </section>
        <section className="bg-white rounded shadow p-4">
          <AlertForm
            onSent={(payload) => {
              setAlerts(prev => [...prev, new Date().toLocaleString()]);
              setLastAlert(payload);
              fetchWeather(payload.location.lat, payload.location.lng);
            }}
          />
          <div className="mt-6">
            <h2 className="font-semibold mb-2">Recent alerts</h2>
            <ul className="space-y-2 max-h-64 overflow-auto">
              {alerts.map((a, idx) => (
                <li key={idx} className="border rounded p-2 text-sm">{a}</li>
              ))}
              {alerts.length === 0 && <div className="text-sm text-gray-500">No alerts yet.</div>}
            </ul>
          </div>
          {lastAlert && (
            <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="border rounded p-4 bg-white">
                <h3 className="font-semibold mb-2">Current Weather</h3>
                {weatherLoading && <div className="text-sm text-gray-500">Loading weather…</div>}
                {weatherError && <div className="text-sm text-red-600">{weatherError}</div>}
                {weather && !weatherLoading && !weatherError && (
                  <div className="text-sm text-gray-700 space-y-1">
                    <div className="text-lg font-medium">{weather.name || 'Selected Location'}</div>
                    <div>Temp: {Math.round(weather.main?.temp)}°C, Feels: {Math.round(weather.main?.feels_like)}°C</div>
                    <div>Condition: {weather.weather?.[0]?.main} ({weather.weather?.[0]?.description})</div>
                    <div>Humidity: {weather.main?.humidity}% • Wind: {Math.round(weather.wind?.speed)} m/s</div>
                  </div>
                )}
              </div>
              <div className="border rounded p-4 bg-white">
                <h3 className="font-semibold mb-2">Risk & Analysis</h3>
                <div className="text-sm text-gray-700 space-y-1">
                  <div><span className="font-medium">Severity:</span> {lastAlert.severity}</div>
                  <div><span className="font-medium">Risk Score:</span> {lastAlert.riskScore}/100</div>
                  <div><span className="font-medium">Languages:</span> {lastAlert.languages.join(', ')}</div>
                  <div><span className="font-medium">Channels:</span> {lastAlert.channels.join(', ')}</div>
                  <div><span className="font-medium">Radius:</span> {lastAlert.location.radius} km</div>
                  <div className="text-xs text-gray-500">{lastAlert.location.address}</div>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}



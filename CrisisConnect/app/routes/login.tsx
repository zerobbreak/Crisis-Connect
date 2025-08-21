import { useNavigate } from "react-router";
import { useState } from "react";

export default function Login() {
  const navigate = useNavigate();
  const [preferredLangs, setPreferredLangs] = useState("");
  const [address, setAddress] = useState<string>("");

  const handleLogin = async () => {
    // token for my API, will need to come back to create or connect API 
    localStorage.setItem("token", "fake-action");
    try {
      if (preferredLangs.trim()) {
        const langs = preferredLangs.split(',').map(l => l.trim()).filter(Boolean);
        if (langs.length) localStorage.setItem('preferredLanguages', JSON.stringify(langs));
      }
      if (address.trim()) {
        // Try simple geocode with OpenWeather if key present
        const apiKey = import.meta.env.VITE_OPENWEATHER_API_KEY;
        if (apiKey) {
          const q = encodeURIComponent(address.trim());
          await fetch(`https://api.openweathermap.org/geo/1.0/direct?q=${q}&limit=1&appid=${apiKey}`)
            .then(r => r.ok ? r.json() : Promise.reject('geo fail'))
            .then((arr) => {
              if (Array.isArray(arr) && arr[0]?.lat && arr[0]?.lon) {
                localStorage.setItem('preferredLocation', JSON.stringify({ lat: arr[0].lat, lng: arr[0].lon, address }));
              }
            })
            .catch(() => {});
        }
      }
    } catch {}
    navigate("/home");
  };

  return (
    <div className="p-5">
      <h2 className="text-2xl font-semibold mb-4">Login</h2>
      <input
        className="border rounded p-2 w-full mb-2"
        placeholder="Username"
      />
      <input
        type="password"
        className="border rounded p-2 w-full mb-4"
        placeholder="Password"
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-4">
        <input
          className="border rounded p-2 w-full"
          placeholder="Preferred languages (e.g. en,af)"
          value={preferredLangs}
          onChange={(e) => setPreferredLangs(e.target.value)}
        />
        <input
          className="border rounded p-2 w-full"
          placeholder="Preferred address (city, suburb, etc.)"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        />
      </div>
      <button
        onClick={handleLogin}
        className="bg-sky-600 text-white py-2 px-4 rounded"
      >
        Login
      </button>
    </div>
  );
}

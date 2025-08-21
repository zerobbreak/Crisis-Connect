import { useNavigate } from "react-router";
import { useState } from "react";

export default function Signup() {
  const navigate = useNavigate();
  const [preferredLangs, setPreferredLangs] = useState("");
  const [address, setAddress] = useState<string>("");

  const handleSignup = async () => {
    alert("Account created successfully!");
    try {
      if (preferredLangs.trim()) {
        const langs = preferredLangs.split(',').map(l => l.trim()).filter(Boolean);
        if (langs.length) localStorage.setItem('preferredLanguages', JSON.stringify(langs));
      }
      if (address.trim()) {
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
    navigate("/login");
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Sign Up</h2>
      <input placeholder="Username" className="border rounded p-2 mb-2 w-full" />
      <input
        type="password"
        placeholder="Password"
        className="border rounded p-2 mb-2 w-full"
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
        onClick={handleSignup}
        className="bg-sky-600 text-white py-2 px-4 rounded"
      >
        Sign Up
      </button>
    </div>
  );
}

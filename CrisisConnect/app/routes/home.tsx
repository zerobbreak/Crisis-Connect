/*import type { Route } from "./+types/home";
import { Welcome } from "../welcome/welcome";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function Home() {
  return <Welcome />;
}*/
import { useState } from "react";
import { useNavigate } from "react-router";
import MapView from "../components/MapView";
import AlertForm from "../components/AlertForm";
import TopNav from "../components/TopNav";
import AlertManagementSystem from "~/components/AlertManagement";

export default function Dashboard() {
  const [alerts, setAlerts] = useState<string[]>([]);
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 bg-white shadow-sm">
        <h1 className="text-xl font-bold text-gray-800">CrisisConnect Admin</h1>
        <button
          onClick={() => {
            localStorage.removeItem("token");
            navigate("/login");
          }}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-md shadow transition"
        >
          Logout
        </button>
      </header>

      <TopNav />
      {/* Main Content */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
        
        {/* Map Section */}
        <section className="col-span-2 bg-white rounded-lg shadow p-3">
          <h2 className="text-lg font-semibold text-gray-700 mb-3">Incident Map</h2>
          <div className="h-[500px] rounded-lg overflow-hidden border">
            <MapView />
          </div>
        </section>

        {/* Alerts Section */}
        <section className="bg-white rounded-lg shadow p-5">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">Send an Alert</h2>
          <AlertForm
            onSent={() =>
              setAlerts((prev) => [...prev, new Date().toLocaleString()])
            }
          />
          <AlertManagementSystem />

          {/* Recent Alerts */}
          <div className="mt-8">
            <h3 className="font-semibold text-gray-700 mb-3">Recent Alerts</h3>
            <ul className="space-y-2 max-h-64 overflow-y-auto">
              {alerts.map((a, idx) => (
                <li
                  key={idx}
                  className="border rounded-md p-2 text-sm text-gray-600 bg-gray-50"
                >
                  {a}
                </li>
              ))}
              {alerts.length === 0 && (
                <div className="text-sm text-gray-400 italic">
                  No alerts yet.
                </div>
              )}
            </ul>
          </div>
        </section>
      </main>
    </div>
  );
}

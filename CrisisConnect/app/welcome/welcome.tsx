// src/components/Welcome.tsx
import React from "react";

export function Welcome() 
{
  const dashboards = [
    { name: "Incident Reports", path: "/dashboard/incidents", icon: "📄" },
    { name: "User Management", path: "/dashboard/users", icon: "👥" },
    { name: "Analytics", path: "/dashboard/analytics", icon: "📊" },
    { name: "Settings", path: "/dashboard/settings", icon: "⚙️" },
  ];

  return (
    <main className="flex items-center justify-center pt-16 pb-4">
      <div className="flex-1 flex flex-col items-center gap-10 min-h-0 px-4">
        {/* Header */}
        <header className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100">
            Welcome to Crisis Connect
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            Manage incidents, monitor reports, and coordinate responses — all in one place.
          </p>
        </header>

        {/* Dashboard Links */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-3xl">
          {dashboards.map((dash) => (
            <a
              key={dash.name}
              href={dash.path}
              className="flex items-center gap-4 p-6 rounded-2xl border border-gray-200 dark:border-gray-700 hover:shadow-md transition bg-white dark:bg-gray-800"
            >
              <span className="text-2xl">{dash.icon}</span>
              <span className="text-lg font-medium text-gray-800 dark:text-gray-200">
                {dash.name}
              </span>
            </a>
          ))}
        </div>
      </div>
    </main>
  );
}

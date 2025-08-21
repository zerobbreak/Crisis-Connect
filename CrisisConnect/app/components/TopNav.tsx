import { NavLink } from "react-router";

export default function TopNav() {
  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-2">
      <div className="flex items-center gap-4 text-sm">
        <NavLink
          to="/home"
          className={({ isActive }) =>
            `px-2 py-1 rounded ${isActive ? "text-blue-600 font-semibold" : "text-gray-600 hover:text-gray-800"}`
          }
        >
          Home
        </NavLink>
        <NavLink
          to="/news"
          className={({ isActive }) =>
            `px-2 py-1 rounded ${isActive ? "text-blue-600 font-semibold" : "text-gray-600 hover:text-gray-800"}`
          }
        >
          News
        </NavLink>
        <NavLink
          to="/dashboard"
          className={({ isActive }) =>
            `px-2 py-1 rounded ${isActive ? "text-blue-600 font-semibold" : "text-gray-600 hover:text-gray-800"}`
          }
        >
          Dashboard
        </NavLink>
        <NavLink
          to="/statistics"
          className={({ isActive }) =>
            `px-2 py-1 rounded ${isActive ? "text-blue-600 font-semibold" : "text-gray-600 hover:text-gray-800"}`
          }
        >
          Statistics
        </NavLink>
        <NavLink
          to="/login"
          className={({ isActive }) =>
            `px-2 py-1 rounded ${isActive ? "text-blue-600 font-semibold" : "text-gray-600 hover:text-gray-800"}`
          }
        >
          Login
        </NavLink>
        <NavLink
          to="/signup"
          className={({ isActive }) =>
            `px-2 py-1 rounded ${isActive ? "text-blue-600 font-semibold" : "text-gray-600 hover:text-gray-800"}`
          }
        >
          Sign Up
        </NavLink>
      </div>
    </nav>
  );
}



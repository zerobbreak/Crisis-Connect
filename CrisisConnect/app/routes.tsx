/*import type { RouteConfig } from "@react-router/dev/routes";

export default
 [
  { path: "/", file: "routes/home.tsx" },      // Default landing page
  { path: "/home", file: "routes/home.tsx" },  // Add this route for /home
  { path: "/login", file: "routes/login.tsx" },
  { path: "/signup", file: "routes/signup.tsx" },
  { path: "/dashboard", file: "pages/dashboard.tsx" },
] satisfies RouteConfig; */

export default [
  { path: "/", file: "routes/redirectHome.tsx" }, // redirect / to /home
  { path: "/home", file: "routes/home.tsx" },
  { path: "/login", file: "routes/login.tsx" },
  { path: "/signup", file: "routes/signup.tsx" },
  { path: "/dashboard", file: "pages/dashboard.tsx" },
  { path: "/news", file: "routes/news.tsx" },
  { path: "/statistics", file: "routes/statistics.tsx" },
] satisfies RouteConfig;import type { RouteConfig } from "@react-router/dev/routes";

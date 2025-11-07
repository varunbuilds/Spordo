import { StrictMode, Suspense } from "react";
import React from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App.jsx";

const Home = React.lazy(() => import("./pages/Home"));
const SportsCoach = React.lazy(() => import("./pages/SportsCoach"));
const Dashboard = React.lazy(() => import("./pages/Dashboard"));
const LivePredictions = React.lazy(() => import("./pages/LivePredictions"));

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Home /> },
      { path: "sports-coach", element: <SportsCoach /> },
      { path: "live-predictions", element: <LivePredictions /> },
      { path: "dashboard", element: <Dashboard /> },
    ],
  },
]);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Suspense fallback={<div>Loading...</div>}>
      <RouterProvider router={router} />
    </Suspense>
  </StrictMode>
);

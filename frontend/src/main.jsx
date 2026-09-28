import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/theme.css";
import App from "./App.jsx";

// Production only, so dev hot-reload is never served from a stale cache.
if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/serviceWorker.js").catch(() => {});
  });
}

createRoot(document.getElementById("root")).render(<App />);

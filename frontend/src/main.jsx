import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/theme.css";
import App from "./App.jsx";

// Offline queue / service worker registration: not built yet (Session 7b).
createRoot(document.getElementById("root")).render(<App />);

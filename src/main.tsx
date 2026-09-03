import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { StartupErrorBoundary } from "./StartupErrorBoundary";
import "./index.css";

// Revealing the window is handled by reveal-window.ts, loaded via its own
// <script> tag ahead of this one in index.html -- see that file for why it
// isn't done here.
// Suppress the webview's native right-click menu (Back/Refresh/Save as/Print/
// Inspect) -- it's Chromium browser chrome, not an app affordance.
window.addEventListener("contextmenu", (event) => event.preventDefault());

const root = document.getElementById("root");

if (!root) {
  throw new Error("ASTRA startup failed: the #root element is missing.");
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <StartupErrorBoundary>
      <App />
    </StartupErrorBoundary>
  </React.StrictMode>,
);

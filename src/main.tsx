import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { StartupErrorBoundary } from "./StartupErrorBoundary";
import "./index.css";

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

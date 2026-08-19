import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

/** Separate from vite.config.ts on purpose.
 *
 * That config is an async factory carrying Tauri's dev-server settings — a
 * fixed port with strictPort, an HMR websocket and a src-tauri watch ignore —
 * none of which apply to a test run, and strictPort in particular makes a test
 * run fail outright if the app is already running on 1420.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    // @ts-expect-error process is a nodejs global
    alias: { "@": path.resolve(process.cwd(), "./src") },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
  },
});

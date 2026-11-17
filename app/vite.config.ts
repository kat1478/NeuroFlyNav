import { defineConfig } from "vitest/config";

// Relative base: the same build works on GitHub Pages (/NeuroFlyNav/) and on CloudFront (/).
export default defineConfig({
  base: "./",
  // The map chunk (MapLibre) is large but loaded only by the city view.
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1200 },
  worker: { format: "es" },
  test: { environment: "jsdom" },
});

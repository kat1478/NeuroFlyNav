import { defineConfig } from "vitest/config";

// Relative base: the same build works on GitHub Pages (/NeuroFlyNav/) and on CloudFront (/).
export default defineConfig({
  base: "./",
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "jsdom" },
});

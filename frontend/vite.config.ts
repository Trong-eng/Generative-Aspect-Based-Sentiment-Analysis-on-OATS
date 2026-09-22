/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  // Optional dev-only proxy so a relative `/predict` can reach a local backend without CORS.
  const proxyTarget = env.DEV_PROXY_TARGET;
  return {
    plugins: [react()],
    server: proxyTarget ? { proxy: { "/predict": { target: proxyTarget, changeOrigin: true } } } : undefined,
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["./src/test/setup.ts"],
      css: false,
    },
  };
});

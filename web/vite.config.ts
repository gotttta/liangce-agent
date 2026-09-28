/// <reference types="vitest/config" />
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    // 开发模式不需要 CORS：/api 由 Vite 代理到本地 FastAPI（计划 §6.4）。
    // LIANGCE_API_URL 用于指向别的本地实例（如验收用 fake 后端），默认 8765。
    proxy: {
      "/api": process.env.LIANGCE_API_URL ?? "http://127.0.0.1:8765",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/tests/setup.ts"],
    include: ["src/tests/**/*.test.{ts,tsx}", "tests/**/*.test.{ts,tsx}"],
  },
});

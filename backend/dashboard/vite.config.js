import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard is a static single-page app, served by the FastAPI service
// itself (docdebt.service.app mounts dist/ under /dashboard) rather than run
// as its own server — one process to deploy, matching the project's
// "one docker compose up" goal. In local development, `npm run dev` proxies
// /api and /webhooks to the FastAPI dev server instead, so no CORS setup is
// needed on either side.
export default defineConfig({
  base: "/dashboard/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

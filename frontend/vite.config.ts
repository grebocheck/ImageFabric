import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

function isExpectedSocketClose(err: unknown): boolean {
  const code = (err as { code?: string } | null)?.code;
  return code === "ECONNABORTED" || code === "ECONNRESET";
}

// Dev server proxies API + WebSocket to the FastAPI backend (port 8260),
// so the browser only ever talks to the Vite origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8260", changeOrigin: true },
      "/ws": {
        target: "ws://127.0.0.1:8260",
        ws: true,
        configure(proxy) {
          proxy.on("error", (err) => {
            if (isExpectedSocketClose(err)) return;
            console.error("[vite] ws proxy error:", err);
          });
        },
      },
    },
  },
});

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import type { ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// Errors that just mean "the backend isn't up yet" or "this socket closed" —
// all transient and self-healing (the client reconnects once uvicorn binds).
// We don't want them spamming the dev console on every startup; we surface a
// single throttled hint instead so a genuinely-down backend is still visible.
const TRANSIENT = new Set(["ECONNREFUSED", "ECONNRESET", "ECONNABORTED", "ETIMEDOUT", "EPIPE"]);

let lastHint = 0;
function quietProxyErrors(proxy: { on(ev: "error", cb: (err: unknown) => void): void }, backend: string) {
  proxy.on("error", (err) => {
    const code = (err as { code?: string } | null)?.code;
    if (code && TRANSIENT.has(code)) {
      const now = Date.now();
      if (now - lastHint > 5000) {
        lastHint = now;
        console.log(`[vite] backend ${backend} not reachable yet — proxy will retry…`);
      }
      return;
    }
    console.error("[vite] proxy error:", err);
  });
}

// Dev server proxies API + WebSocket to the FastAPI backend, so the browser
// only ever talks to the Vite origin.
export default defineConfig(({ mode }) => {
  const rootEnv = loadEnv(mode, ROOT, "HFAB_");
  const env = (name: string, fallback: string) => process.env[name] ?? rootEnv[name] ?? fallback;
  const backendHost = env("HFAB_HOST", "127.0.0.1");
  const proxyHost = backendHost === "0.0.0.0" || backendHost === "::" ? "127.0.0.1" : backendHost;
  const backend = `${proxyHost}:${env("HFAB_PORT", "8260")}`;

  return {
    plugins: [react(), tailwindcss()],
    server: {
      host: env("HFAB_FRONTEND_HOST", "") || undefined,
      port: Number(env("HFAB_FRONTEND_PORT", "5173")),
      proxy: {
        "/api": {
          target: `http://${backend}`,
          changeOrigin: true,
          configure: (proxy) => quietProxyErrors(proxy, backend),
        } satisfies ProxyOptions,
        "/ws": {
          target: `ws://${backend}`,
          ws: true,
          configure: (proxy) => quietProxyErrors(proxy, backend),
        } satisfies ProxyOptions,
      },
    },
  };
});

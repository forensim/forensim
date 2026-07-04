import { useState, useEffect } from "react";
import { apiClient } from "../api/client";

type StatusState = "checking" | "online" | "offline";

function StatusDot({ status }: { status: StatusState }) {
  const color =
    status === "online"
      ? "bg-emerald-500"
      : status === "offline"
      ? "bg-red-500"
      : "bg-amber-500";

  return (
    <span className="relative flex h-2 w-2 shrink-0">
      {status === "online" && (
        <span
          className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60"
          aria-hidden="true"
        />
      )}
      <span className={`relative inline-flex rounded-full h-2 w-2 ${color}`} />
    </span>
  );
}

export default function ApiStatusBar() {
  const [status, setStatus] = useState<StatusState>("checking");
  const [version, setVersion] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    let mounted = true;

    const check = async () => {
      const t0 = performance.now();
      try {
        const h = await apiClient.health();
        const ms = Math.round(performance.now() - t0);
        if (mounted) {
          setStatus("online");
          setVersion(h.version ?? null);
          setLatency(ms);
        }
      } catch {
        if (mounted) {
          setStatus("offline");
          setLatency(null);
        }
      }
    };

    check();
    const id = setInterval(check, 10_000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div
      className="flex items-center gap-2.5 px-4 py-1.5 border-b border-zinc-800/70
                 bg-zinc-950/80 backdrop-blur text-xs select-none"
      role="status"
      aria-live="polite"
    >
      <StatusDot status={status} />

      <span className="text-zinc-600">API</span>

      {status === "checking" && (
        <span className="text-amber-500/80 animate-pulse">connecting…</span>
      )}

      {status === "online" && (
        <>
          <span className="text-emerald-400 font-medium">online</span>
          {version && (
            <span className="text-zinc-700 font-mono">v{version}</span>
          )}
          {latency !== null && (
            <span
              className={`font-mono ${
                latency < 100 ? "text-zinc-700" : "text-amber-600"
              }`}
            >
              {latency}ms
            </span>
          )}
        </>
      )}

      {status === "offline" && (
        <>
          <span className="text-red-400 font-medium">offline</span>
          <span className="text-zinc-600">—</span>
          <code className="font-mono text-zinc-500">
            uvicorn forensim.api.main:app --port 8008
          </code>
        </>
      )}

      {/* Push version label to the far right */}
      <span className="ml-auto font-mono text-zinc-800 text-[10px]">
        forensim · phase 5
      </span>
    </div>
  );
}

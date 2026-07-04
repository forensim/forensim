import { useState, useEffect } from "react";
import { apiClient } from "../api/client";

export default function ApiStatusBar() {
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const check = async () => {
      try {
        const h = await apiClient.health();
        if (mounted) {
          setStatus("online");
          setVersion(h.version ?? null);
        }
      } catch {
        if (mounted) setStatus("offline");
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
    <div className="flex items-center gap-2 px-3 py-1.5 bg-zinc-950 border-b border-zinc-800 text-xs">
      <span
        className={`w-2 h-2 rounded-full ${
          status === "online"
            ? "bg-green-500"
            : status === "offline"
            ? "bg-red-500"
            : "bg-amber-500 animate-pulse"
        }`}
      />
      <span className="text-zinc-500">
        API&nbsp;
        {status === "checking" && "connecting…"}
        {status === "online" && (
          <span className="text-green-400">
            online{version ? ` v${version}` : ""}
          </span>
        )}
        {status === "offline" && (
          <span className="text-red-400">
            offline — start:{" "}
            <code className="font-mono text-zinc-400">
              uvicorn forensim.api.main:app --port 8008
            </code>
          </span>
        )}
      </span>
    </div>
  );
}

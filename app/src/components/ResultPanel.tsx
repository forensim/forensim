import type { ReconstructResponse } from "../api/types";
import { Badge } from "./ui/Badge";

interface ResultPanelProps {
  result: ReconstructResponse;
}

function PathRow({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) return null;
  return (
    <div className="space-y-1">
      <span className="text-xs font-semibold uppercase tracking-widest text-zinc-500">{label}</span>
      <div className="flex items-center gap-2 bg-zinc-950 border border-zinc-800 rounded px-3 py-2">
        <span className="flex-1 text-xs font-mono text-zinc-300 break-all">{value}</span>
        <button
          type="button"
          title="Copy path"
          onClick={() => navigator.clipboard.writeText(value)}
          className="shrink-0 text-zinc-600 hover:text-amber-500 transition-colors text-xs px-1"
        >
          ⎘
        </button>
      </div>
    </div>
  );
}

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center gap-1 bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3">
      <span className="text-lg font-bold text-amber-500 font-mono">{value}</span>
      <span className="text-xs text-zinc-500 uppercase tracking-wider">{label}</span>
    </div>
  );
}

export default function ResultPanel({ result }: ResultPanelProps) {
  const isSuccess = result.status === "success";

  return (
    <div className="space-y-5 animate-fade-up">
      {/* Status banner */}
      <div
        className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${
          isSuccess
            ? "bg-emerald-950/60 border-emerald-800/60 text-emerald-400"
            : "bg-red-950/60 border-red-800/60 text-red-400"
        }`}
      >
        <span className="text-base font-bold">{isSuccess ? "✓" : "✗"}</span>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <p className="font-semibold text-sm">
              {isSuccess ? "Reconstruction complete" : "Reconstruction failed"}
            </p>
            <Badge variant={isSuccess ? "success" : "danger"}>
              {result.status.toUpperCase()}
            </Badge>
          </div>
          {result.message && (
            <p className="text-xs opacity-70 mt-0.5">{result.message}</p>
          )}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3">
        <StatPill
          label="Duration"
          value={`${result.duration_seconds.toFixed(1)}s`}
        />
        <StatPill
          label="Status"
          value={result.status.toUpperCase()}
        />
      </div>

      {/* Output paths */}
      <div className="space-y-3">
        <PathRow label="USD Scene" value={result.usd_path} />
        <PathRow label="Point Cloud (PLY)" value={result.ply_path} />
      </div>

      {/* Open in Explorer hint */}
      {isSuccess && result.usd_path && (
        <p className="text-xs text-zinc-600 text-center">
          Scene saved as USD — open in NVIDIA Omniverse or USD Composer to inspect.
        </p>
      )}
    </div>
  );
}

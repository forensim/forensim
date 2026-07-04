import { useState, useCallback } from "react";
import ApiStatusBar from "./components/ApiStatusBar";
import EvidencePicker from "./components/EvidencePicker";
import ReconstructionPanel from "./components/ReconstructionPanel";
import ResultPanel from "./components/ResultPanel";
import type { ReconstructResponse } from "./api/types";

type Tab = "evidence" | "reconstruct" | "results";

function TabButton({
  label,
  active,
  disabled,
  onClick,
  badge,
}: {
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  badge?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`relative px-5 py-3 text-sm font-medium transition-colors
        ${active
          ? "text-amber-500 border-b-2 border-amber-500"
          : "text-zinc-500 hover:text-zinc-300 border-b-2 border-transparent"
        }
        disabled:opacity-40 disabled:cursor-not-allowed`}
    >
      {label}
      {badge && (
        <span className="ml-2 px-1.5 py-0.5 text-[10px] rounded-full bg-amber-500/20 text-amber-500 font-mono">
          {badge}
        </span>
      )}
    </button>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("evidence");
  const [selection, setSelection] = useState<{
    imageDir: string;
    workspaceDir: string;
  } | null>(null);
  const [result, setResult] = useState<ReconstructResponse | null>(null);

  const handleSelectionChange = useCallback(
    (sel: { imageDir: string; workspaceDir: string } | null) => {
      setSelection(sel);
    },
    []
  );

  const handleResult = useCallback((res: ReconstructResponse) => {
    setResult(res);
    setTab("results");
  }, []);

  return (
    <div className="flex flex-col h-screen bg-[#09090b] text-zinc-200 overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-zinc-800 bg-zinc-950 shrink-0">
        <div className="flex items-center gap-3">
          {/* Logo mark */}
          <div className="w-7 h-7 rounded bg-amber-500 flex items-center justify-center text-zinc-950 font-black text-sm select-none">
            FS
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-zinc-100">ForenSim</h1>
            <p className="text-[10px] text-zinc-600 leading-none">
              Forensic Scene Reconstruction
            </p>
          </div>
        </div>
        <div className="text-[10px] font-mono text-zinc-700 select-none">v0.1.0</div>
      </header>

      {/* ── API status ──────────────────────────────────────────── */}
      <ApiStatusBar />

      {/* ── Tab bar ─────────────────────────────────────────────── */}
      <nav className="flex px-4 border-b border-zinc-800 bg-zinc-950 shrink-0">
        <TabButton
          label="Evidence"
          active={tab === "evidence"}
          onClick={() => setTab("evidence")}
          badge={selection ? "✓" : undefined}
        />
        <TabButton
          label="Reconstruct"
          active={tab === "reconstruct"}
          disabled={!selection}
          onClick={() => setTab("reconstruct")}
        />
        <TabButton
          label="Results"
          active={tab === "results"}
          disabled={!result}
          onClick={() => setTab("results")}
          badge={result ? "1" : undefined}
        />
      </nav>

      {/* ── Main content ────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto p-5">
        {tab === "evidence" && (
          <div className="max-w-4xl mx-auto space-y-4">
            <div>
              <h2 className="text-base font-semibold text-zinc-100">Load Evidence</h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                Select a folder of scene photographs and an output workspace for
                reconstruction results.
              </p>
            </div>
            <EvidencePicker onSelectionChange={handleSelectionChange} />
            {selection && (
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => setTab("reconstruct")}
                  className="px-5 py-2 rounded-lg bg-amber-500 hover:bg-amber-400
                             text-zinc-950 font-semibold text-sm transition-colors"
                >
                  Continue →
                </button>
              </div>
            )}
          </div>
        )}

        {tab === "reconstruct" && selection && (
          <div className="max-w-2xl mx-auto space-y-4">
            <div>
              <h2 className="text-base font-semibold text-zinc-100">Reconstruction</h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                Configure and run the COLMAP → Gaussian Splatting → USD pipeline.
              </p>
            </div>
            {/* Evidence summary */}
            <div className="flex gap-3 text-xs text-zinc-500 font-mono bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
              <span className="text-zinc-600">imgs:</span>
              <span className="text-zinc-400 truncate flex-1">{selection.imageDir}</span>
            </div>
            <div className="flex gap-3 text-xs text-zinc-500 font-mono bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
              <span className="text-zinc-600">out:</span>
              <span className="text-zinc-400 truncate flex-1">{selection.workspaceDir}</span>
            </div>
            <ReconstructionPanel
              imageDir={selection.imageDir}
              workspaceDir={selection.workspaceDir}
              onResult={handleResult}
            />
          </div>
        )}

        {tab === "results" && result && (
          <div className="max-w-2xl mx-auto space-y-4">
            <div>
              <h2 className="text-base font-semibold text-zinc-100">Results</h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                Reconstruction output — load the USD scene in Omniverse to continue.
              </p>
            </div>
            <ResultPanel result={result} />
            <div className="flex justify-center">
              <button
                type="button"
                onClick={() => {
                  setResult(null);
                  setTab("evidence");
                }}
                className="px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700
                           text-zinc-300 text-sm font-medium transition-colors border border-zinc-700"
              >
                Start New Case
              </button>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer className="shrink-0 px-5 py-2 border-t border-zinc-800 bg-zinc-950
                         flex items-center justify-between text-[10px] text-zinc-700">
        <span>ForenSim · Forensic Scene Reconstruction Platform</span>
        <span className="font-mono">Phase 1.5</span>
      </footer>
    </div>
  );
}

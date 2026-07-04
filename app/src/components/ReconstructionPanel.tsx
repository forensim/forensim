import { useState, useCallback, useRef } from "react";
import { apiClient } from "../api/client";
import type { ReconstructResponse, ProgressEvent } from "../api/types";
import { ProgressBar } from "./ui/ProgressBar";

interface ReconstructionPanelProps {
  imageDir: string;
  workspaceDir: string;
  onResult: (result: ReconstructResponse) => void;
}

type Method = "gaussian_splatting" | "nerf" | "nurec";
type Matcher = "exhaustive" | "sequential";

interface Settings {
  method: Method;
  matcher: Matcher;
  maxSteps: number;
  gsplatFallback: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  method: "gaussian_splatting",
  matcher: "exhaustive",
  maxSteps: 30000,
  gsplatFallback: true,
};

function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
      <span className="text-sm text-zinc-400">{label}</span>
      <div className="ml-4">{children}</div>
    </div>
  );
}

function Select<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.currentTarget.value as T)}
      className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-sm text-zinc-200
                 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export default function ReconstructionPanel({
  imageDir,
  workspaceDir,
  onResult,
}: ReconstructionPanelProps) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const appendLog = useCallback((msg: string) => {
    setLog((prev) => [...prev.slice(-199), msg]);
    setTimeout(() => logEndRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }, []);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    setLog([]);
    setProgress(null);

    // Subscribe to SSE progress stream
    unsubRef.current = apiClient.streamProgress(
      (evt) => {
        setProgress(evt);
        appendLog(`[${evt.step}] ${evt.message}`);
      },
      () => {
        // stream closed normally
      },
      (err) => appendLog(`[error] ${err.message}`)
    );

    try {
      const result = await apiClient.runReconstruction({
        image_dir: imageDir,
        workspace_dir: workspaceDir,
        method: settings.method,
        matcher: settings.matcher,
        max_steps: settings.maxSteps,
        gsplat_fallback: settings.gsplatFallback,
      });
      appendLog(`Done in ${result.duration_seconds.toFixed(1)}s — status: ${result.status}`);
      onResult(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      appendLog(`[error] ${msg}`);
    } finally {
      unsubRef.current?.();
      unsubRef.current = null;
      setRunning(false);
    }
  }, [imageDir, workspaceDir, settings, onResult, appendLog]);

  const handleCancel = useCallback(() => {
    unsubRef.current?.();
    unsubRef.current = null;
    setRunning(false);
    appendLog("[cancelled] Reconstruction stopped by user.");
  }, [appendLog]);

  const disabled = !imageDir || !workspaceDir;

  return (
    <div className="space-y-4">
      {/* Settings */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-500 mb-3">
          Reconstruction Settings
        </h3>
        <SettingRow label="Method">
          <Select<Method>
            value={settings.method}
            options={[
              { value: "gaussian_splatting", label: "Gaussian Splatting" },
              { value: "nerf", label: "NeRF" },
              { value: "nurec", label: "NuRec (NVIDIA)" },
            ]}
            onChange={(v) => setSettings((s) => ({ ...s, method: v }))}
          />
        </SettingRow>
        <SettingRow label="Feature Matcher">
          <Select<Matcher>
            value={settings.matcher}
            options={[
              { value: "exhaustive", label: "Exhaustive" },
              { value: "sequential", label: "Sequential" },
            ]}
            onChange={(v) => setSettings((s) => ({ ...s, matcher: v }))}
          />
        </SettingRow>
        <SettingRow label="Training Steps">
          <input
            type="number"
            min={1000}
            max={100000}
            step={1000}
            value={settings.maxSteps}
            onChange={(e) => setSettings((s) => ({ ...s, maxSteps: Number(e.currentTarget.value) }))}
            className="w-28 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-sm text-zinc-200
                       focus:outline-none focus:border-amber-500"
          />
        </SettingRow>
        <SettingRow label="Use Fallback Exporter">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.gsplatFallback}
              onChange={(e) => setSettings((s) => ({ ...s, gsplatFallback: e.currentTarget.checked }))}
              className="accent-amber-500 w-4 h-4"
            />
            <span className="text-sm text-zinc-400">
              {settings.gsplatFallback ? "Fallback (fast)" : "Real trainer (CUDA)"}
            </span>
          </label>
        </SettingRow>
      </div>

      {/* Progress bar */}
      {progress && (
        <div className="space-y-2 animate-fade-up">
          <div className="flex justify-between text-xs">
            <span className="text-zinc-400 font-mono">{progress.step}</span>
            <span className="text-zinc-500 font-mono tabular-nums">{progress.percent}%</span>
          </div>
          <ProgressBar
            value={progress.percent}
            variant={progress.percent === 100 ? "success" : "amber"}
            height="h-2"
          />
          <p className="text-xs text-zinc-500 italic">{progress.message}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-3 rounded bg-red-950 border border-red-800 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={handleRun}
          disabled={disabled || running}
          className="flex-1 py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400
                     disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed
                     text-zinc-950 font-semibold text-sm transition-colors"
        >
          {running ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-zinc-950/30 border-t-zinc-950 rounded-full animate-spin" />
              Reconstructing…
            </span>
          ) : (
            "Run Reconstruction"
          )}
        </button>
        {running && (
          <button
            type="button"
            onClick={handleCancel}
            className="px-4 py-2.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300
                       font-semibold text-sm transition-colors border border-zinc-700"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Log console */}
      {log.length > 0 && (
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-zinc-800 bg-zinc-900/50">
            <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest">
              Pipeline Log
            </span>
            <span className="text-[10px] font-mono text-zinc-700">{log.length} lines</span>
          </div>
          <div className="p-3 max-h-44 overflow-y-auto font-mono text-xs space-y-0.5">
            {log.map((line, i) => (
              <div
                key={i}
                className={
                  line.startsWith("[error]")
                    ? "text-red-400"
                    : line.startsWith("[cancelled]")
                    ? "text-amber-400"
                    : line.startsWith("Done")
                    ? "text-emerald-400"
                    : "text-zinc-500"
                }
              >
                <span className="text-zinc-700 select-none mr-2">{String(i + 1).padStart(3, " ")}</span>
                {line}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}

      {disabled && !running && (
        <p className="text-xs text-zinc-600 text-center">
          Select an evidence folder and workspace to enable reconstruction.
        </p>
      )}
    </div>
  );
}

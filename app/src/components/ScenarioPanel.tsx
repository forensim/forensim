import { useState, useCallback, useRef } from "react";
import type { TrajectoryData, SimRunResult } from "../api/types";

// ── Types ──────────────────────────────────────────────────────────────────────

/** A simulated object with initial conditions */
interface SimObject {
  id: string;
  name: string;
  velocity: [number, number, number];
  angularVelocity: [number, number, number];
}

interface SimResponse {
  status: string;
  results: SimRunResult[];
}

export interface ScenarioPanelProps {
  usdPath: string | null;
  onResults: (results: SimRunResult[], trajectories: TrajectoryData[]) => void;
}

// ── Constants ──────────────────────────────────────────────────────────────────

const TRAJECTORY_COLORS = ["#f59e0b", "#22d3ee", "#a78bfa", "#34d399"];

const DEFAULT_N_STEPS = 300;
const DT = 1 / 60;

function makeObject(): SimObject {
  return {
    id: crypto.randomUUID(),
    name: "",
    velocity: [0, 0, 0],
    angularVelocity: [0, 0, 0],
  };
}

// ── Small helper components ────────────────────────────────────────────────────

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-500 mb-3">
      {children}
    </h3>
  );
}

function VectorInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: [number, number, number];
  onChange: (v: [number, number, number]) => void;
}) {
  const axes = ["X", "Y", "Z"] as const;
  return (
    <div className="space-y-1">
      <span className="text-xs text-amber-500/80">{label}</span>
      <div className="flex gap-2">
        {axes.map((axis, i) => (
          <label key={axis} className="flex-1 flex flex-col gap-0.5">
            <span className="text-[10px] text-zinc-600 font-mono text-center">{axis}</span>
            <input
              type="number"
              step={0.1}
              value={value[i]}
              onChange={(e) => {
                const next = [...value] as [number, number, number];
                next[i] = parseFloat(e.currentTarget.value) || 0;
                onChange(next);
              }}
              className="w-full bg-zinc-950 border border-zinc-700 rounded px-2 py-1
                         text-xs font-mono text-zinc-200 text-right
                         focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500/40"
            />
          </label>
        ))}
      </div>
    </div>
  );
}

function ObjectCard({
  obj,
  onUpdate,
  onRemove,
}: {
  obj: SimObject;
  onUpdate: (updated: SimObject) => void;
  onRemove: () => void;
}) {
  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-3">
      {/* Header row */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          placeholder="Object name in USD"
          value={obj.name}
          onChange={(e) => onUpdate({ ...obj, name: e.currentTarget.value })}
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2.5 py-1.5
                     text-sm font-mono text-zinc-200 placeholder-zinc-600
                     focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500/40"
        />
        <button
          type="button"
          onClick={onRemove}
          className="shrink-0 px-3 py-1.5 rounded bg-red-950 border border-red-800
                     text-red-400 text-xs font-semibold hover:bg-red-900 transition-colors"
        >
          Remove
        </button>
      </div>

      {/* Velocity */}
      <VectorInput
        label="Velocity (m/s)"
        value={obj.velocity}
        onChange={(v) => onUpdate({ ...obj, velocity: v })}
      />

      {/* Angular velocity */}
      <VectorInput
        label="Angular velocity (rad/s)"
        value={obj.angularVelocity}
        onChange={(v) => onUpdate({ ...obj, angularVelocity: v })}
      />
    </div>
  );
}

function ResultCard({
  result,
  color,
}: {
  result: SimRunResult;
  color: string;
}) {
  const objectName = result.scenario.object_name;
  const entries = Object.entries(result.final_positions);

  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-3">
      {/* Title row */}
      <div className="flex items-center gap-2">
        {/* Color swatch */}
        <span
          className="w-3 h-3 rounded-sm shrink-0"
          style={{ backgroundColor: color }}
        />
        <h4 className="text-sm font-semibold font-mono text-zinc-100 truncate">{objectName}</h4>
        <span className="ml-auto text-xs font-mono text-amber-500 shrink-0">
          {result.trajectory_length} frames
        </span>
      </div>

      {/* Final positions */}
      {entries.length > 0 ? (
        <div className="space-y-1">
          <span className="text-[10px] uppercase tracking-widest text-zinc-600">
            Final positions
          </span>
          {entries.map(([objKey, pos]) => (
            <div
              key={objKey}
              className="flex items-baseline gap-2 bg-zinc-900 rounded px-2.5 py-1.5"
            >
              <span className="text-xs font-mono text-zinc-400 shrink-0 truncate max-w-[8rem]">
                {objKey}
              </span>
              <span className="ml-auto text-xs font-mono text-zinc-300 shrink-0">
                [{pos.map((v) => v.toFixed(3)).join(", ")}]
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-zinc-600">No final position data.</p>
      )}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ScenarioPanel({ usdPath, onResults }: ScenarioPanelProps) {
  const [objects, setObjects] = useState<SimObject[]>([makeObject()]);
  const [nSteps, setNSteps] = useState(DEFAULT_N_STEPS);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<SimRunResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // ── Object list mutations ──────────────────────────────────────────────────

  const handleAddObject = useCallback(() => {
    setObjects((prev) => [...prev, makeObject()]);
  }, []);

  const handleUpdateObject = useCallback((updated: SimObject) => {
    setObjects((prev) => prev.map((o) => (o.id === updated.id ? updated : o)));
  }, []);

  const handleRemoveObject = useCallback((id: string) => {
    setObjects((prev) => prev.filter((o) => o.id !== id));
  }, []);

  // ── Derive TrajectoryData from API results ─────────────────────────────────

  const deriveTrajectories = useCallback(
    (simResults: SimRunResult[]): TrajectoryData[] => {
      return simResults
        .filter((r) => r.trajectory_length > 0)
        .map((r, i) => {
          const objectName = r.scenario.object_name;
          const color = TRAJECTORY_COLORS[i % TRAJECTORY_COLORS.length];

          // Placeholder: straight line from origin to the first final_position entry
          const finalPos = Object.values(r.final_positions)[0] ?? [0, 0, 0];
          const end: [number, number, number] = [
            finalPos[0] ?? 0,
            finalPos[1] ?? 0,
            finalPos[2] ?? 0,
          ];

          return {
            id: objectName,
            color,
            points: [[0, 0, 0], end] as [number, number, number][],
          };
        });
    },
    []
  );

  // ── Run simulation ─────────────────────────────────────────────────────────

  const handleRun = useCallback(async () => {
    if (!usdPath || objects.length === 0 || running) return;

    setRunning(true);
    setError(null);
    setResults(null);

    const controller = new AbortController();
    abortRef.current = controller;

    const scenarios = objects.map((obj) => ({
      object_name: obj.name || `object_${obj.id.slice(0, 6)}`,
      velocity: [...obj.velocity],
      angular_velocity: [...obj.angularVelocity],
    }));

    const body = {
      usd_path: usdPath,
      scenarios,
      n_steps: nSteps,
      dt: DT,
    };

    try {
      const response = await fetch("http://127.0.0.1:8008/simulate/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => `HTTP ${response.status}`);
        throw new Error(`Server error ${response.status}: ${text}`);
      }

      const data: SimResponse = await response.json();
      setResults(data.results);

      const trajectories = deriveTrajectories(data.results);
      onResults(data.results, trajectories);
    } catch (err) {
      if ((err as { name?: string }).name === "AbortError") {
        // cancelled by user — no error banner
      } else {
        const msg = err instanceof Error ? err.message : String(err);
        setError(msg);
      }
    } finally {
      abortRef.current = null;
      setRunning(false);
    }
  }, [usdPath, objects, nSteps, running, deriveTrajectories, onResults]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // ── Run button disabled logic ──────────────────────────────────────────────

  const canRun = !!usdPath && objects.length > 0 && !running;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">
      {/* No USD path prompt */}
      {!usdPath && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-6 text-center space-y-1">
          <p className="text-sm text-zinc-400">No scene loaded.</p>
          <p className="text-xs text-zinc-600">
            Select an Evidence folder and run reconstruction first.
          </p>
        </div>
      )}

      {/* USD path display */}
      {usdPath && (
        <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
          <span className="text-xs text-zinc-600 font-mono shrink-0">scene:</span>
          <span className="flex-1 text-xs font-mono text-zinc-300 truncate">{usdPath}</span>
        </div>
      )}

      {/* ── Part 1: Object list ── */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
        <SectionHeading>Simulation Objects</SectionHeading>

        {objects.length === 0 ? (
          <p className="text-xs text-zinc-600 py-2">
            No objects defined. Add at least one object to run a simulation.
          </p>
        ) : (
          <div className="space-y-3">
            {objects.map((obj) => (
              <ObjectCard
                key={obj.id}
                obj={obj}
                onUpdate={handleUpdateObject}
                onRemove={() => handleRemoveObject(obj.id)}
              />
            ))}
          </div>
        )}

        <button
          type="button"
          onClick={handleAddObject}
          className="w-full py-2 rounded-lg border border-dashed border-zinc-700
                     text-zinc-500 hover:text-zinc-300 hover:border-zinc-500
                     text-sm transition-colors"
        >
          + Add Object
        </button>
      </div>

      {/* ── Part 2: Simulation settings ── */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <SectionHeading>Simulation Settings</SectionHeading>
        <div className="space-y-0 divide-y divide-zinc-800">
          {/* nSteps */}
          <div className="flex items-center justify-between py-2.5">
            <label
              htmlFor="n-steps-input"
              className="text-sm text-amber-500/90"
            >
              Steps (@&nbsp;60&nbsp;Hz)
            </label>
            <input
              id="n-steps-input"
              type="number"
              min={60}
              max={3000}
              step={60}
              value={nSteps}
              onChange={(e) => setNSteps(Number(e.currentTarget.value))}
              className="w-28 bg-zinc-950 border border-zinc-700 rounded px-2 py-1
                         text-sm font-mono text-zinc-200 text-right
                         focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500/40"
            />
          </div>

          {/* dt (display only) */}
          <div className="flex items-center justify-between py-2.5">
            <span className="text-sm text-amber-500/90">Time step</span>
            <span className="text-sm font-mono text-zinc-400">0.0167 s / step</span>
          </div>

          {/* Computed duration */}
          <div className="flex items-center justify-between py-2.5">
            <span className="text-sm text-zinc-600">Duration</span>
            <span className="text-sm font-mono text-zinc-500">
              {(nSteps / 60).toFixed(1)} s
            </span>
          </div>
        </div>
      </div>

      {/* ── Part 3: Error banner ── */}
      {error && (
        <div className="flex items-start gap-3 p-3 rounded-lg bg-red-950 border border-red-800">
          <span className="text-red-400 text-base shrink-0">✗</span>
          <p className="text-sm text-red-400 break-all">{error}</p>
        </div>
      )}

      {/* ── Part 3: Run / Cancel buttons ── */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun}
          className="flex-1 py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400
                     disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed
                     text-zinc-950 font-semibold text-sm transition-colors"
        >
          {running ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-zinc-950/30 border-t-zinc-950 rounded-full animate-spin" />
              Simulating…
            </span>
          ) : (
            "Run Simulation"
          )}
        </button>

        {running && (
          <button
            type="button"
            onClick={handleCancel}
            className="px-4 py-2.5 rounded-lg bg-zinc-800 hover:bg-zinc-700
                       text-zinc-300 font-semibold text-sm transition-colors border border-zinc-700"
          >
            Cancel
          </button>
        )}
      </div>

      {/* ── Part 4: Result cards ── */}
      {results && results.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <SectionHeading>Simulation Results</SectionHeading>
            <span className="ml-auto text-xs font-mono text-zinc-600 pb-3">
              {results.length} scenario{results.length !== 1 ? "s" : ""}
            </span>
          </div>
          {results.map((r, i) => (
            <ResultCard
              key={r.scenario.object_name + i}
              result={r}
              color={TRAJECTORY_COLORS[i % TRAJECTORY_COLORS.length]}
            />
          ))}
        </div>
      )}

      {/* Empty results notice */}
      {results && results.length === 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-4 text-center">
          <p className="text-xs text-zinc-600">
            Simulation completed but returned no results.
          </p>
        </div>
      )}
    </div>
  );
}

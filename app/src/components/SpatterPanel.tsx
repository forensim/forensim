/**
 * SpatterPanel — Blood Spatter / Fluid Simulation
 *
 * Allows the analyst to configure a blood-spatter SPH simulation:
 *   • Set the source point (x, y, z) and ejection velocity
 *   • Control droplet count, spread angle, and random seed
 *   • View the resulting droplet impact map rendered on a 2D Canvas
 *   • Read the forensic pattern analysis (type, directionality, stringing, etc.)
 *   • Export the analysis as an evidence source for the Sensitivity panel
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { apiClient } from "../api/client";
import type { SpatterRequest, SpatterResponse, SpatterImpact } from "../api/types";
import { Badge } from "./ui/Badge";
import { ProgressBar } from "./ui/ProgressBar";
import { Skeleton } from "./ui/Skeleton";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Vec3 { x: number; y: number; z: number }

const PATTERN_VARIANT: Record<string, "amber" | "danger" | "info" | "success" | "neutral"> = {
  impact:    "danger",
  cast_off:  "amber",
  projected: "info",
  transfer:  "neutral",
  unknown:   "neutral",
};

const PATTERN_LABEL: Record<string, string> = {
  impact:    "Impact Spatter",
  cast_off:  "Cast-off",
  projected: "Projected / Arterial",
  transfer:  "Transfer / Contact",
  unknown:   "Unknown",
};

// ── Canvas renderer ───────────────────────────────────────────────────────────

function useSpatterCanvas(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  impacts: SpatterImpact[],
  centroid: [number, number] | null,
  estimatedSource: [number, number, number] | null,
  directionVector: [number, number] | null,
) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || impacts.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const pad = 40;

    // Compute world bounds
    const xs = impacts.map((d) => d.position_2d[0]);
    const zs = impacts.map((d) => d.position_2d[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);
    const rangeX = maxX - minX || 1;
    const rangeZ = maxZ - minZ || 1;
    const scale = Math.min((W - pad * 2) / rangeX, (H - pad * 2) / rangeZ);

    const toCanvas = (wx: number, wz: number): [number, number] => [
      pad + (wx - minX) * scale,
      H - pad - (wz - minZ) * scale,
    ];

    // Background
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 0.5;
    const gridStep = 0.1; // 10 cm
    for (let gx = Math.floor(minX / gridStep) * gridStep; gx <= maxX + gridStep; gx += gridStep) {
      const [cx] = toCanvas(gx, minZ);
      ctx.beginPath(); ctx.moveTo(cx, pad); ctx.lineTo(cx, H - pad); ctx.stroke();
    }
    for (let gz = Math.floor(minZ / gridStep) * gridStep; gz <= maxZ + gridStep; gz += gridStep) {
      const [, cy] = toCanvas(minX, gz);
      ctx.beginPath(); ctx.moveTo(pad, cy); ctx.lineTo(W - pad, cy); ctx.stroke();
    }

    // Draw stain ellipses
    for (const impact of impacts) {
      const [cx, cy] = toCanvas(impact.position_2d[0], impact.position_2d[1]);
      const a = Math.max(1, (impact.stain_major_mm / 1000) * scale / 2);
      const b = Math.max(0.5, (impact.stain_minor_mm / 1000) * scale / 2);
      const angleDeg = impact.stain_angle;
      const angleRad = (angleDeg * Math.PI) / 180;

      // Colour by impact angle: perpendicular = red, shallow = amber
      const t = impact.impact_angle / 90; // 0 = grazing, 1 = perpendicular
      const r = Math.round(220 + (1 - t) * 35);
      const g = Math.round(t * 40);
      const bVal = Math.round(t * 30);
      ctx.fillStyle = `rgba(${r},${g},${bVal},0.75)`;
      ctx.strokeStyle = `rgba(${r},${g},${bVal},0.4)`;
      ctx.lineWidth = 0.5;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angleRad);
      ctx.beginPath();
      ctx.ellipse(0, 0, a, b, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Centroid
    if (centroid) {
      const [cx, cy] = toCanvas(centroid[0], centroid[1]);
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#f59e0b";
      ctx.fill();
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Direction vector arrow from centroid
    if (centroid && directionVector) {
      const [cx, cy] = toCanvas(centroid[0], centroid[1]);
      const arrowLen = 40;
      const [dvx, dvz] = directionVector;
      const norm = Math.sqrt(dvx * dvx + dvz * dvz) || 1;
      const ex = cx + (dvx / norm) * arrowLen;
      // z increases away from viewer in world → decreases in canvas y
      const ey = cy - (dvz / norm) * arrowLen;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(ey - cy, ex - cx);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 10 * Math.cos(angle - 0.4), ey - 10 * Math.sin(angle - 0.4));
      ctx.lineTo(ex - 10 * Math.cos(angle + 0.4), ey - 10 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = "#f59e0b";
      ctx.fill();
    }

    // Estimated source projection (cross)
    if (estimatedSource) {
      const [sx, sy] = toCanvas(estimatedSource[0], estimatedSource[2]);
      const cs = 8;
      ctx.strokeStyle = "#34d399";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx - cs, sy); ctx.lineTo(sx + cs, sy);
      ctx.moveTo(sx, sy - cs); ctx.lineTo(sx, sy + cs);
      ctx.stroke();
      ctx.font = "9px monospace";
      ctx.fillStyle = "#34d399";
      ctx.fillText("AOO", sx + cs + 3, sy + 4);
    }

    // Scale bar
    const barWorld = 0.5; // 50 cm
    const barPx = barWorld * scale;
    ctx.strokeStyle = "#3f3f46";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, H - 12);
    ctx.lineTo(pad + barPx, H - 12);
    ctx.stroke();
    ctx.font = "9px monospace";
    ctx.fillStyle = "#52525b";
    ctx.fillText("0.5 m", pad + barPx + 4, H - 9);
  }, [canvasRef, impacts, centroid, estimatedSource, directionVector]);
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Vec3Input({
  label,
  value,
  onChange,
  unit = "m",
  step = "0.1",
}: {
  label: string;
  value: Vec3;
  onChange: (v: Vec3) => void;
  unit?: string;
  step?: string;
}) {
  const field = (axis: keyof Vec3) => (
    <label className="flex flex-col gap-0.5 flex-1">
      <span className="text-[9px] text-zinc-600 uppercase text-center">{axis.toUpperCase()}</span>
      <input
        type="number"
        step={step}
        value={value[axis]}
        onChange={(e) =>
          onChange({ ...value, [axis]: parseFloat(e.currentTarget.value) || 0 })
        }
        className="w-full bg-zinc-950 border border-zinc-800 rounded px-1.5 py-1
                   text-xs font-mono text-zinc-200 text-center
                   focus:outline-none focus:border-amber-500"
      />
    </label>
  );

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5">
        <span className="text-xs text-zinc-500 font-medium w-28 shrink-0">{label}</span>
        <span className="text-[10px] text-zinc-700">({unit})</span>
      </div>
      <div className="flex gap-1.5">{field("x")}{field("y")}{field("z")}</div>
    </div>
  );
}

function KpiCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2.5 text-center">
      <div className="text-base font-bold font-mono text-amber-400 leading-tight">{value}</div>
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mt-0.5">{label}</div>
      {sub && <div className="text-[9px] text-zinc-700 mt-0.5">{sub}</div>}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function SpatterPanel() {
  // ── Config state ────────────────────────────────────────────────────────────
  const [sourcePos, setSourcePos] = useState<Vec3>({ x: 0, y: 1.2, z: 0 });
  const [sourceVel, setSourceVel] = useState<Vec3>({ x: 4.0, y: -1.5, z: 0 });
  const [nDroplets, setNDroplets] = useState(200);
  const [spreadAngle, setSpreadAngle] = useState(45);
  const [maxTime, setMaxTime] = useState(3.0);
  const [seed, setSeed] = useState(42);

  // ── Result state ────────────────────────────────────────────────────────────
  const [result, setResult] = useState<SpatterResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ── Canvas rendering ────────────────────────────────────────────────────────
  useSpatterCanvas(
    canvasRef,
    result?.impacts ?? [],
    result?.pattern_centroid ?? null,
    result?.estimated_source ?? null,
    result?.direction_vector ?? null,
  );

  // ── Simulation runner ───────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    const t0 = performance.now();

    const req: SpatterRequest = {
      source_position: [sourcePos.x, sourcePos.y, sourcePos.z],
      source_velocity: [sourceVel.x, sourceVel.y, sourceVel.z],
      n_droplets: nDroplets,
      velocity_spread_angle: spreadAngle,
      max_time: maxTime,
      seed,
    };

    try {
      const res = await apiClient.runSpatter(req);
      setResult(res);
      setElapsed(Math.round(performance.now() - t0));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }, [sourcePos, sourceVel, nDroplets, spreadAngle, maxTime, seed]);

  // ── Clear canvas when no result ─────────────────────────────────────────────
  useEffect(() => {
    if (!result && canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.fillStyle = "#09090b";
        ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  }, [result]);

  const analysis = result?.analysis;

  return (
    <div className="space-y-5 animate-fade-up">

      {/* ── Config card ── */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-500">
          Simulation Parameters
        </h3>

        <Vec3Input
          label="Source Position"
          value={sourcePos}
          onChange={setSourcePos}
          unit="m"
          step="0.05"
        />
        <Vec3Input
          label="Ejection Velocity"
          value={sourceVel}
          onChange={setSourceVel}
          unit="m/s"
          step="0.5"
        />

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <label className="space-y-1">
            <span className="text-[10px] text-zinc-600 uppercase tracking-widest">Droplets</span>
            <input
              type="number"
              min={10}
              max={2000}
              step={10}
              value={nDroplets}
              onChange={(e) => setNDroplets(Number(e.currentTarget.value))}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1
                         text-xs font-mono text-zinc-200
                         focus:outline-none focus:border-amber-500"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[10px] text-zinc-600 uppercase tracking-widest">Spread °</span>
            <input
              type="number"
              min={1}
              max={90}
              step={1}
              value={spreadAngle}
              onChange={(e) => setSpreadAngle(Number(e.currentTarget.value))}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1
                         text-xs font-mono text-zinc-200
                         focus:outline-none focus:border-amber-500"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[10px] text-zinc-600 uppercase tracking-widest">Max Time s</span>
            <input
              type="number"
              min={0.5}
              max={10}
              step={0.5}
              value={maxTime}
              onChange={(e) => setMaxTime(Number(e.currentTarget.value))}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1
                         text-xs font-mono text-zinc-200
                         focus:outline-none focus:border-amber-500"
            />
          </label>
          <label className="space-y-1">
            <span className="text-[10px] text-zinc-600 uppercase tracking-widest">Seed</span>
            <input
              type="number"
              min={0}
              step={1}
              value={seed}
              onChange={(e) => setSeed(Number(e.currentTarget.value))}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1
                         text-xs font-mono text-zinc-200
                         focus:outline-none focus:border-amber-500"
            />
          </label>
        </div>

        <button
          type="button"
          onClick={handleRun}
          disabled={running}
          className="w-full py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400
                     disabled:opacity-50 disabled:cursor-not-allowed
                     text-zinc-950 font-bold text-sm transition-colors"
        >
          {running ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-zinc-950/30 border-t-zinc-950 rounded-full animate-spin" />
              Simulating…
            </span>
          ) : (
            "Run Spatter Simulation"
          )}
        </button>

        {running && <ProgressBar variant="amber" height="h-1" />}
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="p-3 rounded-lg bg-red-950/60 border border-red-800/60 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* ── Skeleton while running ── */}
      {running && !result && (
        <div className="space-y-3">
          <Skeleton className="h-48 w-full rounded-xl" />
          <div className="grid grid-cols-4 gap-3">
            {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-14 rounded-lg" />)}
          </div>
        </div>
      )}

      {/* ── Results ── */}
      {result && !running && (
        <div className="space-y-4 animate-fade-up">

          {/* Pattern type + header */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <Badge variant={PATTERN_VARIANT[analysis?.pattern_type ?? "unknown"] ?? "neutral"}>
                {PATTERN_LABEL[analysis?.pattern_type ?? "unknown"] ?? "Unknown"}
              </Badge>
              {analysis?.confidence !== undefined && (
                <span className="text-xs text-zinc-500">
                  {(analysis.confidence * 100).toFixed(0)}% confidence
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-xs text-zinc-700">
              <span>{result.n_impacts} impacts · {result.n_airborne} airborne</span>
              {elapsed != null && <span className="font-mono">{elapsed}ms</span>}
            </div>
          </div>

          {/* KPI row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KpiCard
              label="Directionality"
              value={`${((analysis?.directionality_score ?? 0) * 100).toFixed(0)}%`}
              sub="flow coherence"
            />
            <KpiCard
              label="Stringing"
              value={`${(analysis?.stringing_factor ?? 0).toFixed(2)}×`}
              sub="major / minor ratio"
            />
            <KpiCard
              label="Impact Angle"
              value={`${(analysis?.impact_angle_mean ?? 0).toFixed(1)}°`}
              sub="mean incidence"
            />
            <KpiCard
              label="Mean Stain"
              value={`${(analysis?.mean_stain_area_mm2 ?? 0).toFixed(2)} mm²`}
              sub="average area"
            />
          </div>

          {/* Droplet map canvas */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 bg-zinc-900/50">
              <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-600">
                Impact Map — Floor Projection
              </span>
              <div className="flex items-center gap-3 text-[10px] text-zinc-700">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
                  Centroid
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 inline-block border border-emerald-500" />
                  AOO estimate
                </span>
              </div>
            </div>
            <canvas
              ref={canvasRef}
              width={600}
              height={400}
              className="w-full"
              style={{ imageRendering: "pixelated" }}
            />
          </div>

          {/* Geometry readout */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 space-y-1">
              <p className="text-[10px] uppercase tracking-widest text-zinc-600">Pattern Centroid</p>
              <p className="font-mono text-xs text-zinc-300">
                x={result.pattern_centroid[0].toFixed(3)} m,{" "}
                z={result.pattern_centroid[1].toFixed(3)} m
              </p>
              <p className="font-mono text-xs text-zinc-600">
                spread ±{result.pattern_spread_radius.toFixed(3)} m
              </p>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 space-y-1">
              <p className="text-[10px] uppercase tracking-widest text-zinc-600">Est. Area of Origin</p>
              <p className="font-mono text-xs text-emerald-400">
                x={result.estimated_source[0].toFixed(3)}{" "}
                y={result.estimated_source[1].toFixed(3)}{" "}
                z={result.estimated_source[2].toFixed(3)} m
              </p>
              <p className="text-[10px] text-zinc-700">
                {analysis?.source_height.toFixed(2)} m above floor ·{" "}
                {analysis?.source_distance.toFixed(2)} m from centroid
              </p>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 space-y-1">
              <p className="text-[10px] uppercase tracking-widest text-zinc-600">Log-likelihood</p>
              <p className="font-mono text-xs text-amber-400">
                {analysis?.log_likelihood.toFixed(4)}
              </p>
              <p className="text-[10px] text-zinc-600">
                for Bayesian integration
              </p>
            </div>
          </div>

          {/* Analysis notes */}
          {analysis && analysis.notes.length > 0 && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 space-y-2">
              <h4 className="text-[10px] uppercase tracking-widest text-zinc-600 font-semibold">
                Forensic Interpretation
              </h4>
              <ul className="space-y-1.5">
                {analysis.notes.map((note, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-zinc-400">
                    <span className="text-amber-600 mt-0.5 shrink-0">▸</span>
                    {note}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* ── Empty state ── */}
      {!result && !running && !error && (
        <div className="flex flex-col items-center justify-center gap-3 py-16
                        bg-zinc-950/50 border border-zinc-800 border-dashed rounded-xl">
          <span className="text-4xl opacity-20 select-none">🩸</span>
          <p className="text-sm text-zinc-600">
            Configure source position and ejection velocity, then run the simulation.
          </p>
          <p className="text-xs text-zinc-800">
            Uses a ballistic SPH droplet model with bloodstain pattern analysis formulas.
          </p>
        </div>
      )}
    </div>
  );
}

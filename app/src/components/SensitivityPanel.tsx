import { useState, useCallback } from "react";
import { apiClient } from "../api/client";
import type {
  EvidenceSourceModel,
  HypothesisResult,
  SensitivityResultItem,
} from "../api/types";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface SensitivityPanelProps {
  /** Ranked hypotheses from a previous /api/infer/rank call. */
  hypotheses?: HypothesisResult[];
}

// ── Small shared UI pieces ─────────────────────────────────────────────────────

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-semibold text-amber-500 uppercase tracking-widest mb-2">
      {children}
    </h3>
  );
}

// ── Evidence source editor ─────────────────────────────────────────────────────

interface EvidenceSourceEditorProps {
  sources: EvidenceSourceModel[];
  hypothesisCount: number;
  onAdd: () => void;
  onRemove: (idx: number) => void;
  onChange: (idx: number, field: keyof EvidenceSourceModel, value: unknown) => void;
}

function EvidenceSourceEditor({
  sources,
  hypothesisCount,
  onAdd,
  onRemove,
  onChange,
}: EvidenceSourceEditorProps) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <SectionHeading>Evidence Sources</SectionHeading>
        <button
          type="button"
          onClick={onAdd}
          className="text-xs px-2 py-0.5 rounded bg-zinc-700 hover:bg-zinc-600
                     text-zinc-300 font-medium transition-colors"
        >
          + Add Source
        </button>
      </div>

      {sources.length === 0 && (
        <p className="text-xs text-zinc-600 italic">
          No evidence sources defined. Add one and fill in the per-hypothesis
          log-likelihood deltas.
        </p>
      )}

      {sources.map((src, idx) => (
        <div
          key={idx}
          className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-2"
        >
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={src.name}
              onChange={(e) => onChange(idx, "name", e.currentTarget.value)}
              placeholder='e.g. "blood_spatter_roi"'
              className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1
                         text-xs font-mono text-zinc-200 placeholder-zinc-600
                         focus:outline-none focus:border-amber-500"
            />
            <label className="flex items-center gap-1 text-xs text-zinc-500">
              weight
              <input
                type="number"
                step={0.1}
                min={0}
                max={10}
                value={src.weight ?? 1.0}
                onChange={(e) =>
                  onChange(
                    idx,
                    "weight",
                    parseFloat(e.currentTarget.value) || 1.0
                  )
                }
                className="w-14 bg-zinc-900 border border-zinc-700 rounded px-1.5 py-0.5
                           text-xs font-mono text-zinc-200
                           focus:outline-none focus:border-amber-500"
              />
            </label>
            <button
              type="button"
              onClick={() => onRemove(idx)}
              className="text-zinc-600 hover:text-red-400 transition-colors text-sm font-bold"
            >
              ×
            </button>
          </div>

          <div className="space-y-1">
            <span className="text-[10px] text-zinc-600">
              Log-likelihood deltas (one per hypothesis, space or comma separated)
            </span>
            <textarea
              rows={1}
              value={src.log_likelihood_delta.join(", ")}
              onChange={(e) => {
                const raw = e.currentTarget.value;
                const parsed = raw
                  .split(/[\s,]+/)
                  .map((v) => parseFloat(v))
                  .filter((v) => !isNaN(v));
                onChange(idx, "log_likelihood_delta", parsed);
              }}
              placeholder={
                hypothesisCount > 0
                  ? Array(hypothesisCount).fill("0.0").join(", ")
                  : "0.0, 0.0, ..."
              }
              className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1
                         text-xs font-mono text-zinc-200 placeholder-zinc-600 resize-none
                         focus:outline-none focus:border-amber-500"
            />
            <span className="text-[10px] text-zinc-600">
              {src.log_likelihood_delta.length} value
              {src.log_likelihood_delta.length !== 1 ? "s" : ""}
              {hypothesisCount > 0 && (
                <span
                  className={
                    src.log_likelihood_delta.length !== hypothesisCount
                      ? " text-red-400"
                      : " text-emerald-600"
                  }
                >
                  {" "}
                  (need {hypothesisCount})
                </span>
              )}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Impact bar ─────────────────────────────────────────────────────────────────

function ImpactBar({
  impact,
  maxAbs,
}: {
  impact: number;
  maxAbs: number;
}) {
  const pct = maxAbs > 0 ? (Math.abs(impact) / maxAbs) * 100 : 0;
  const positive = impact >= 0;
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          style={{ width: `${pct}%` }}
          className={`h-full rounded-full transition-all duration-300 ${
            positive ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
      </div>
      <span
        className={`text-xs font-mono tabular-nums w-16 text-right shrink-0 ${
          positive ? "text-emerald-400" : "text-red-400"
        }`}
      >
        {positive ? "+" : ""}
        {(impact * 100).toFixed(1)}%
      </span>
    </div>
  );
}

// ── Results card ───────────────────────────────────────────────────────────────

interface ResultsCardProps {
  results: SensitivityResultItem[];
  baselineTopPosterior: number;
  topHypothesis: string | null;
}

function ResultsCard({
  results,
  baselineTopPosterior,
  topHypothesis,
}: ResultsCardProps) {
  const maxAbs = Math.max(...results.map((r) => Math.abs(r.impact)), 1e-9);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <SectionHeading>Sensitivity Results</SectionHeading>
        <span className="text-xs font-mono text-zinc-500">
          baseline P(top) ={" "}
          <span className="text-zinc-300">
            {(baselineTopPosterior * 100).toFixed(1)}%
          </span>
        </span>
      </div>

      {topHypothesis && (
        <div className="flex items-center gap-2 px-3 py-2 rounded
                        bg-amber-500/10 border border-amber-500/30 text-xs text-amber-400">
          <span className="font-semibold shrink-0">Top hypothesis:</span>
          <span className="truncate">{topHypothesis}</span>
        </div>
      )}

      <div className="text-[10px] text-zinc-600 px-1">
        Bars show how much each evidence source changes the top-hypothesis
        posterior when removed (leave-one-out). Positive = evidence supports
        the top hypothesis.
      </div>

      <div className="space-y-3">
        {results.map((r) => (
          <div key={r.evidence_name} className="space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-mono text-zinc-300 truncate">
                {r.evidence_name}
              </span>
              <div className="flex items-center gap-3 shrink-0 text-[10px] text-zinc-600 font-mono">
                {r.rank_change !== 0 && (
                  <span className="text-amber-400">
                    rank {r.rank_change > 0 ? "↓" : "↑"}{Math.abs(r.rank_change)}
                  </span>
                )}
                <span>
                  LOO:{" "}
                  <span className="text-zinc-400">
                    {(r.loo_top_posterior * 100).toFixed(1)}%
                  </span>
                </span>
              </div>
            </div>
            <ImpactBar impact={r.impact} maxAbs={maxAbs} />
          </div>
        ))}
      </div>

      <div className="pt-2 border-t border-zinc-800 grid grid-cols-2 gap-3">
        {results.slice(0, 4).map((r) => (
          <div
            key={r.evidence_name}
            className="bg-zinc-950 rounded border border-zinc-800 px-3 py-2 space-y-0.5"
          >
            <div className="text-[10px] text-zinc-600 truncate font-mono">
              {r.evidence_name}
            </div>
            <div
              className={`text-sm font-bold tabular-nums ${
                r.impact >= 0 ? "text-emerald-400" : "text-red-400"
              }`}
            >
              {r.impact >= 0 ? "+" : ""}
              {(r.impact_pct).toFixed(1)}%
            </div>
            <div className="text-[10px] text-zinc-600">impact on P(top)</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Hypotheses preview ─────────────────────────────────────────────────────────

function HypothesesPreview({ hypotheses }: { hypotheses: HypothesisResult[] }) {
  if (hypotheses.length === 0) return null;
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-2">
      <SectionHeading>Current Hypothesis Rankings</SectionHeading>
      <div className="space-y-1">
        {hypotheses.map((h) => (
          <div
            key={h.rank}
            className="flex items-center gap-3 text-xs px-2 py-1.5 rounded bg-zinc-950"
          >
            <span className="font-mono text-zinc-600 w-4 text-right shrink-0">
              #{h.rank}
            </span>
            <span className="flex-1 truncate text-zinc-300">{h.description}</span>
            <span className="font-mono text-amber-400 shrink-0">
              {(h.posterior * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── SensitivityPanel ──────────────────────────────────────────────────────────

export default function SensitivityPanel({ hypotheses = [] }: SensitivityPanelProps) {
  const [sources, setSources] = useState<EvidenceSourceModel[]>([
    {
      name: "blood_spatter_roi",
      log_likelihood_delta: hypotheses.map(() => 0.5),
      weight: 1.0,
    },
  ]);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<SensitivityResultItem[] | null>(null);
  const [baselineTop, setBaselineTop] = useState(0);
  const [topHyp, setTopHyp] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ── Source management ────────────────────────────────────────────────────

  const handleAddSource = useCallback(() => {
    setSources((prev) => [
      ...prev,
      {
        name: "",
        log_likelihood_delta: Array(hypotheses.length).fill(0) as number[],
        weight: 1.0,
      },
    ]);
  }, [hypotheses.length]);

  const handleRemoveSource = useCallback((idx: number) => {
    setSources((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const handleChangeSource = useCallback(
    (idx: number, field: keyof EvidenceSourceModel, value: unknown) => {
      setSources((prev) =>
        prev.map((s, i) => (i === idx ? { ...s, [field]: value } : s))
      );
    },
    []
  );

  // ── Run sensitivity analysis ─────────────────────────────────────────────

  const canRun =
    hypotheses.length > 0 &&
    sources.length > 0 &&
    sources.every(
      (s) =>
        s.name.trim() !== "" &&
        s.log_likelihood_delta.length === hypotheses.length
    );

  const handleRun = useCallback(async () => {
    if (!canRun) return;
    setRunning(true);
    setError(null);
    try {
      const res = await apiClient.runSensitivity({
        hypotheses,
        evidence_sources: sources.map((s) => ({
          name: s.name,
          log_likelihood_delta: s.log_likelihood_delta,
          weight: s.weight ?? 1.0,
        })),
      });
      setResults(res.results);
      setBaselineTop(res.baseline_top_posterior);
      setTopHyp(res.top_hypothesis);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [canRun, hypotheses, sources]);

  // ── Validation feedback ──────────────────────────────────────────────────

  const validationMsg =
    hypotheses.length === 0
      ? "Run inference first to load ranked hypotheses."
      : !canRun
      ? "Each evidence source needs a name and one delta per hypothesis."
      : null;

  return (
    <div className="space-y-4">
      {/* Hypotheses preview */}
      <HypothesesPreview hypotheses={hypotheses} />

      {hypotheses.length === 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 text-center">
          <p className="text-sm text-zinc-600">
            No inference results available yet.
          </p>
          <p className="text-xs text-zinc-700 mt-1">
            Go to the Infer tab, run hypothesis ranking, then return here.
          </p>
        </div>
      )}

      {/* Evidence source editor */}
      <EvidenceSourceEditor
        sources={sources}
        hypothesisCount={hypotheses.length}
        onAdd={handleAddSource}
        onRemove={handleRemoveSource}
        onChange={handleChangeSource}
      />

      {/* Run button */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun || running}
          className="px-5 py-2 rounded-lg bg-amber-500 hover:bg-amber-400
                     text-zinc-950 font-semibold text-sm transition-colors
                     disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {running ? "Analysing…" : "Run Sensitivity Analysis"}
        </button>

        {validationMsg && (
          <span className="text-xs text-zinc-600 italic">{validationMsg}</span>
        )}
      </div>

      {error && (
        <p className="text-xs text-red-400 font-mono bg-red-500/10 border border-red-500/20 rounded px-3 py-2 break-all">
          {error}
        </p>
      )}

      {/* Results */}
      {results && results.length > 0 && (
        <ResultsCard
          results={results}
          baselineTopPosterior={baselineTop}
          topHypothesis={topHyp}
        />
      )}

      {results && results.length === 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <p className="text-xs text-zinc-600 italic">
            No results returned. Ensure evidence sources have valid deltas.
          </p>
        </div>
      )}
    </div>
  );
}

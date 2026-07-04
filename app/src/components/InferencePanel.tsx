import { useState, useCallback, useRef } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

/** One candidate event hypothesis */
interface Hypothesis {
  id: string;
  description: string;
  events: string[];            // ordered event names e.g. ["shot_fired", "casing_ejected", "impact"]
  physxLogLikelihood: number;  // from simulation, default -1
}

/** Result from the /api/infer/rank endpoint */
interface HypothesisResult {
  rank: number;
  description: string;
  log_probability: number;
  posterior: number;
  events: string[];
}

interface InferResponse {
  status: string;
  hypotheses: HypothesisResult[];
}

/** POST body sent to /api/infer/rank */
interface InferRequest {
  sequences: string[][];
  descriptions: string[];
  transition_matrix: number[][];
  initial_probs: number[];
  event_vocab: string[];
  physx_log_likelihoods: number[];
}

export interface InferencePanelProps {
  /** Optional physx log-likelihoods from simulation (parallel to hypotheses list).
   *  If provided, pre-fills the physxLogLikelihood field for each hypothesis. */
  simulationLogLikelihoods?: number[];
}

// ── Default constants ──────────────────────────────────────────────────────────

const DEFAULT_VOCAB: string[] = [
  "shot_fired",
  "casing_ejected",
  "target_impact",
  "ricochet",
  "secondary_impact",
];

const MAX_EVENTS_PER_HYPOTHESIS = 8;
const API_BASE = "http://127.0.0.1:8008";

// ── Helper functions ───────────────────────────────────────────────────────────

function buildUniformMatrix(n: number): number[][] {
  const p = n > 0 ? 1 / n : 0;
  return Array.from({ length: n }, () => Array(n).fill(p) as number[]);
}

function buildUniformInitial(n: number): number[] {
  const p = n > 0 ? 1 / n : 0;
  return Array(n).fill(p) as number[];
}

function makeHypothesis(): Hypothesis {
  return {
    id: crypto.randomUUID(),
    description: "",
    events: [""],
    physxLogLikelihood: -1.0,
  };
}

const DEFAULT_HYPOTHESES: Hypothesis[] = [
  {
    id: crypto.randomUUID(),
    description: "Single shot, east entry",
    events: ["shot_fired", "casing_ejected", "target_impact"],
    physxLogLikelihood: -1.0,
  },
  {
    id: crypto.randomUUID(),
    description: "Ricochet scenario",
    events: ["shot_fired", "ricochet", "secondary_impact"],
    physxLogLikelihood: -2.5,
  },
];

// ── Small shared UI components ─────────────────────────────────────────────────

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-semibold text-amber-500 uppercase tracking-widest mb-2">
      {children}
    </h3>
  );
}

// ── Section 1: Event Vocabulary ────────────────────────────────────────────────

interface VocabSectionProps {
  vocab: string[];
  onAdd: (word: string) => void;
  onRemove: (word: string) => void;
}

function VocabSection({ vocab, onAdd, onRemove }: VocabSectionProps) {
  const [inputValue, setInputValue] = useState("");

  const handleAdd = useCallback(() => {
    const trimmed = inputValue.trim();
    if (trimmed && !vocab.includes(trimmed)) {
      onAdd(trimmed);
      setInputValue("");
    }
  }, [inputValue, vocab, onAdd]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") handleAdd();
    },
    [handleAdd]
  );

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <SectionHeading>Event Vocabulary</SectionHeading>
      <div className="flex gap-2 mb-3">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          placeholder='e.g. "shot_fired"'
          className="flex-1 bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-sm
                     text-zinc-200 placeholder-zinc-600
                     focus:outline-none focus:border-amber-500"
        />
        <button
          type="button"
          onClick={handleAdd}
          className="px-3 py-1 rounded bg-amber-500 hover:bg-amber-400
                     text-zinc-950 font-semibold text-sm transition-colors"
        >
          Add
        </button>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {vocab.map((word) => (
          <span
            key={word}
            className="flex items-center gap-1 px-2 py-0.5 rounded-full
                       bg-amber-500/15 border border-amber-500/40 text-amber-400 text-xs font-mono"
          >
            {word}
            <button
              type="button"
              onClick={() => onRemove(word)}
              aria-label={`Remove ${word}`}
              className="ml-0.5 text-amber-400/70 hover:text-amber-200 transition-colors leading-none"
            >
              ×
            </button>
          </span>
        ))}
        {vocab.length === 0 && (
          <span className="text-xs text-zinc-600 italic">No events defined.</span>
        )}
      </div>
    </div>
  );
}

// ── Section 2: Transition Matrix ───────────────────────────────────────────────

interface TransitionMatrixSectionProps {
  vocab: string[];
  matrixJson: string;
  matrixError: string | null;
  onMatrixChange: (json: string, error: string | null) => void;
}

function TransitionMatrixSection({
  vocab,
  matrixJson,
  matrixError,
  onMatrixChange,
}: TransitionMatrixSectionProps) {
  const [open, setOpen] = useState(false);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const raw = e.currentTarget.value;
      if (raw.trim() === "") {
        onMatrixChange("", null);
        return;
      }
      try {
        const parsed: unknown = JSON.parse(raw);
        const n = vocab.length;
        if (
          !Array.isArray(parsed) ||
          parsed.length !== n ||
          !(parsed as unknown[]).every(
            (row) => Array.isArray(row) && (row as unknown[]).length === n
          )
        ) {
          onMatrixChange(raw, `Expected a ${n}×${n} 2D array (vocab size = ${n})`);
        } else {
          onMatrixChange(raw, null);
        }
      } catch {
        onMatrixChange(raw, "Invalid JSON");
      }
    },
    [vocab.length, onMatrixChange]
  );

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 w-full text-left group"
      >
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-widest group-hover:text-zinc-200 transition-colors">
          Advanced: Transition Matrix
        </span>
        <svg
          className={`w-3.5 h-3.5 text-zinc-500 transition-transform duration-200 ${open ? "rotate-180" : "rotate-0"}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={2.5}
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="m19 9-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="mt-3 space-y-2">
          <p className="text-xs text-zinc-500">
            Leave blank to use uniform transitions (equal probability for all event pairs).
          </p>
          <textarea
            rows={4}
            value={matrixJson}
            onChange={handleChange}
            placeholder={`Paste a ${vocab.length}×${vocab.length} JSON 2D array here…`}
            className={`w-full bg-zinc-950 border rounded px-2 py-1.5 text-xs font-mono
                        text-zinc-200 resize-y focus:outline-none
                        ${matrixError
                          ? "border-red-600 focus:border-red-500"
                          : "border-zinc-700 focus:border-amber-500"
                        }`}
          />
          {matrixError && (
            <p className="text-xs text-red-400 font-mono">{matrixError}</p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Section 3: Hypothesis card ─────────────────────────────────────────────────

interface HypothesisCardProps {
  hypothesis: Hypothesis;
  vocab: string[];
  onUpdate: (h: Hypothesis) => void;
  onRemove: () => void;
}

function HypothesisCard({ hypothesis, vocab, onUpdate, onRemove }: HypothesisCardProps) {
  const setDescription = useCallback(
    (description: string) => onUpdate({ ...hypothesis, description }),
    [hypothesis, onUpdate]
  );

  const setPhysxLogLikelihood = useCallback(
    (v: number) => onUpdate({ ...hypothesis, physxLogLikelihood: v }),
    [hypothesis, onUpdate]
  );

  const setEventAt = useCallback(
    (index: number, value: string) => {
      const next = [...hypothesis.events];
      next[index] = value;
      onUpdate({ ...hypothesis, events: next });
    },
    [hypothesis, onUpdate]
  );

  const addEventSlot = useCallback(() => {
    if (hypothesis.events.length < MAX_EVENTS_PER_HYPOTHESIS) {
      onUpdate({ ...hypothesis, events: [...hypothesis.events, vocab[0] ?? ""] });
    }
  }, [hypothesis, onUpdate, vocab]);

  const removeEventSlot = useCallback(
    (index: number) => {
      if (hypothesis.events.length <= 1) return;
      const next = hypothesis.events.filter((_, i) => i !== index);
      onUpdate({ ...hypothesis, events: next });
    },
    [hypothesis, onUpdate]
  );

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
      {/* Header: description + delete */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={hypothesis.description}
          onChange={(e) => setDescription(e.currentTarget.value)}
          placeholder="e.g. Shooter at east window"
          className="flex-1 bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-sm
                     text-zinc-200 placeholder-zinc-600
                     focus:outline-none focus:border-amber-500"
        />
        <button
          type="button"
          onClick={onRemove}
          aria-label="Delete hypothesis"
          className="shrink-0 px-2.5 py-1 rounded bg-red-950 border border-red-800
                     text-red-400 text-xs font-semibold hover:bg-red-900 transition-colors"
        >
          🗑
        </button>
      </div>

      {/* Event sequence */}
      <div>
        <span className="text-[10px] uppercase tracking-widest text-zinc-600">
          Event sequence
        </span>
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          {hypothesis.events.map((ev, idx) => (
            <div key={idx} className="flex items-center gap-1">
              {idx > 0 && (
                <span className="text-zinc-600 text-xs select-none">→</span>
              )}
              <div className="flex items-center gap-0.5">
                <select
                  value={ev}
                  onChange={(e) => setEventAt(idx, e.currentTarget.value)}
                  className="bg-zinc-950 border border-zinc-700 rounded px-2 py-1
                             text-sm text-zinc-200
                             focus:outline-none focus:border-amber-500"
                >
                  {vocab.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                  {/* Keep current value selectable even if removed from vocab */}
                  {!vocab.includes(ev) && ev && (
                    <option value={ev}>{ev}</option>
                  )}
                </select>
                <button
                  type="button"
                  onClick={() => removeEventSlot(idx)}
                  disabled={hypothesis.events.length <= 1}
                  aria-label="Remove event"
                  className="px-1.5 py-1 rounded text-zinc-600 hover:text-red-400
                             disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-xs"
                >
                  −
                </button>
              </div>
            </div>
          ))}
          {hypothesis.events.length < MAX_EVENTS_PER_HYPOTHESIS && (
            <button
              type="button"
              onClick={addEventSlot}
              disabled={vocab.length === 0}
              aria-label="Add event slot"
              className="px-2 py-1 rounded border border-zinc-700 text-zinc-400
                         hover:border-amber-500 hover:text-amber-400
                         disabled:opacity-30 disabled:cursor-not-allowed
                         text-xs transition-colors"
            >
              +
            </button>
          )}
        </div>
      </div>

      {/* PhysX log-likelihood */}
      <div className="flex items-center gap-3">
        <label className="text-xs text-zinc-500 shrink-0">PhysX ln P(E|H)</label>
        <input
          type="number"
          step={0.01}
          value={hypothesis.physxLogLikelihood}
          onChange={(e) =>
            setPhysxLogLikelihood(parseFloat(e.currentTarget.value) || 0)
          }
          className="w-28 bg-zinc-950 border border-zinc-700 rounded px-2 py-1
                     text-sm font-mono text-zinc-200
                     focus:outline-none focus:border-amber-500"
        />
      </div>
    </div>
  );
}

// ── Section 5: Result card ─────────────────────────────────────────────────────

interface ResultCardProps {
  result: HypothesisResult;
  rank1LogProb: number;
}

function ResultCard({ result, rank1LogProb }: ResultCardProps) {
  const isRank1 = result.rank === 1;
  const isRank2 = result.rank === 2;

  const barColor = isRank1
    ? "bg-amber-500"
    : isRank2
      ? "bg-cyan-500"
      : "bg-violet-500";

  const badgeClass = isRank1
    ? "bg-amber-500 text-zinc-950"
    : "bg-zinc-700 text-zinc-300";

  const posteriorPct = (result.posterior * 100).toFixed(1);
  const barWidth = Math.min(result.posterior * 100, 100);

  const bayesFactor =
    !isRank1
      ? Math.exp(result.log_probability - rank1LogProb)
      : null;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
      {/* Header row */}
      <div className="flex items-start gap-3">
        <span
          className={`shrink-0 px-2 py-0.5 rounded text-xs font-bold font-mono ${badgeClass}`}
        >
          #{result.rank}
        </span>
        <span className="font-semibold text-white text-sm leading-snug">{result.description}</span>
      </div>

      {/* Events chips */}
      <div className="flex flex-wrap items-center gap-1">
        {result.events.map((ev, idx) => (
          <span key={idx} className="flex items-center gap-1">
            {idx > 0 && <span className="text-zinc-600 text-xs select-none">→</span>}
            <span className="px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-300 text-xs font-mono">
              {ev}
            </span>
          </span>
        ))}
      </div>

      {/* Posterior bar */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-widest text-zinc-600">Posterior</span>
          <span className="text-xs font-mono text-zinc-300">{posteriorPct}%</span>
        </div>
        <div className="bg-zinc-800 rounded-full h-2 w-full overflow-hidden">
          <div
            className={`${barColor} h-2 rounded-full transition-all duration-500`}
            style={{ width: `${barWidth}%` }}
          />
        </div>
      </div>

      {/* Log-probability + Bayes Factor */}
      <div className="flex items-center gap-4">
        <span className="text-xs font-mono text-zinc-500">
          ln P = {result.log_probability.toFixed(4)}
        </span>
        {bayesFactor !== null && (
          <span className="text-xs text-zinc-500">
            BF vs #1:{" "}
            <span className="font-mono text-zinc-400">
              {bayesFactor.toFixed(4)}×
            </span>
          </span>
        )}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function InferencePanel({
  simulationLogLikelihoods,
}: InferencePanelProps) {
  // Merge simulationLogLikelihoods into the default hypotheses at mount-time.
  const initialHypotheses: Hypothesis[] = DEFAULT_HYPOTHESES.map((h, i) => ({
    ...h,
    physxLogLikelihood:
      simulationLogLikelihoods?.[i] ?? h.physxLogLikelihood,
  }));

  // ── State ──────────────────────────────────────────────────────────────────

  const [vocab, setVocab] = useState<string[]>(DEFAULT_VOCAB);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>(initialHypotheses);

  const [matrixJson, setMatrixJson] = useState<string>("");
  const [matrixError, setMatrixError] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [inferError, setInferError] = useState<string | null>(null);
  const [results, setResults] = useState<HypothesisResult[] | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // ── Vocab mutations ────────────────────────────────────────────────────────

  const handleAddVocab = useCallback((word: string) => {
    setVocab((prev) => (prev.includes(word) ? prev : [...prev, word]));
  }, []);

  const handleRemoveVocab = useCallback((word: string) => {
    setVocab((prev) => prev.filter((w) => w !== word));
  }, []);

  // ── Transition matrix ──────────────────────────────────────────────────────

  const handleMatrixChange = useCallback((json: string, error: string | null) => {
    setMatrixJson(json);
    setMatrixError(error);
  }, []);

  /** Returns the parsed matrix or a freshly built uniform one. */
  const resolveMatrix = useCallback((): number[][] => {
    if (matrixJson.trim() && !matrixError) {
      try {
        return JSON.parse(matrixJson) as number[][];
      } catch {
        // fall through to uniform
      }
    }
    return buildUniformMatrix(vocab.length);
  }, [matrixJson, matrixError, vocab.length]);

  // ── Hypothesis mutations ───────────────────────────────────────────────────

  const handleAddHypothesis = useCallback(() => {
    setHypotheses((prev) => [...prev, makeHypothesis()]);
  }, []);

  const handleUpdateHypothesis = useCallback((updated: Hypothesis) => {
    setHypotheses((prev) => prev.map((h) => (h.id === updated.id ? updated : h)));
  }, []);

  const handleRemoveHypothesis = useCallback((id: string) => {
    setHypotheses((prev) => prev.filter((h) => h.id !== id));
  }, []);

  // ── Run inference ──────────────────────────────────────────────────────────

  const handleRunInference = useCallback(async () => {
    if (hypotheses.length === 0 || running) return;

    // Cancel any previous in-flight request
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setRunning(true);
    setInferError(null);
    setResults(null);

    const transitionMatrix = resolveMatrix();
    const initialProbs = buildUniformInitial(vocab.length);

    const body: InferRequest = {
      sequences: hypotheses.map((h) => h.events),
      descriptions: hypotheses.map((h) => h.description),
      transition_matrix: transitionMatrix,
      initial_probs: initialProbs,
      event_vocab: vocab,
      physx_log_likelihoods: hypotheses.map((h) => h.physxLogLikelihood),
    };

    try {
      const response = await fetch(`${API_BASE}/api/infer/rank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => `HTTP ${response.status}`);
        throw new Error(`Server error ${response.status}: ${text}`);
      }

      const data = (await response.json()) as InferResponse;
      const sorted = [...data.hypotheses].sort((a, b) => a.rank - b.rank);
      setResults(sorted);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const msg = err instanceof Error ? err.message : String(err);
      setInferError(msg);
    } finally {
      setRunning(false);
    }
  }, [hypotheses, running, vocab, resolveMatrix]);

  // ── Derived values for result display ─────────────────────────────────────

  const rank1LogProb = results?.[0]?.log_probability ?? 0;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">
      {/* ── Section 1: Vocabulary ──────────────────────────────────── */}
      <VocabSection
        vocab={vocab}
        onAdd={handleAddVocab}
        onRemove={handleRemoveVocab}
      />

      {/* ── Section 2: Transition Matrix ──────────────────────────── */}
      <TransitionMatrixSection
        vocab={vocab}
        matrixJson={matrixJson}
        matrixError={matrixError}
        onMatrixChange={handleMatrixChange}
      />

      {/* ── Section 3: Hypotheses ─────────────────────────────────── */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
        <SectionHeading>Hypotheses</SectionHeading>

        {hypotheses.length === 0 && (
          <p className="text-xs text-zinc-600 italic">
            No hypotheses defined. Add one below.
          </p>
        )}

        {hypotheses.map((h) => (
          <HypothesisCard
            key={h.id}
            hypothesis={h}
            vocab={vocab}
            onUpdate={handleUpdateHypothesis}
            onRemove={() => handleRemoveHypothesis(h.id)}
          />
        ))}

        {/* Add hypothesis button */}
        <button
          type="button"
          onClick={handleAddHypothesis}
          className="w-full py-2 rounded-lg border border-dashed border-zinc-700
                     text-amber-500 text-sm font-medium hover:border-amber-500/60
                     hover:bg-amber-500/5 transition-colors"
        >
          + Add Hypothesis
        </button>
      </div>

      {/* ── Section 4: Run Inference ───────────────────────────────── */}
      <button
        type="button"
        onClick={handleRunInference}
        disabled={running || hypotheses.length === 0}
        className="w-full py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400
                   disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed
                   text-zinc-950 font-semibold text-sm transition-colors"
      >
        {running ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-zinc-950/30 border-t-zinc-950 rounded-full animate-spin" />
            Running Inference…
          </span>
        ) : (
          "Run Inference"
        )}
      </button>

      {/* ── Error banner ──────────────────────────────────────────── */}
      {inferError && (
        <div className="p-3 rounded bg-red-950 border border-red-800 text-red-400 text-sm">
          {inferError}
        </div>
      )}

      {/* ── Section 5: Results ────────────────────────────────────── */}
      {results && results.length > 0 && (
        <div className="space-y-3">
          <SectionHeading>Ranked Hypotheses</SectionHeading>
          {results.map((r) => (
            <ResultCard key={r.rank} result={r} rank1LogProb={rank1LogProb} />
          ))}
        </div>
      )}

      {results && results.length === 0 && (
        <div className="p-3 rounded bg-zinc-900 border border-zinc-800 text-zinc-500 text-sm text-center">
          No results returned by the inference engine.
        </div>
      )}
    </div>
  );
}

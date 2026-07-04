import { useState, useCallback } from "react";
import ApiStatusBar from "./components/ApiStatusBar";
import EvidencePicker from "./components/EvidencePicker";
import ReconstructionPanel from "./components/ReconstructionPanel";
import ResultPanel from "./components/ResultPanel";
import SplatViewer from "./components/SplatViewer";
import ScenarioPanel from "./components/ScenarioPanel";
import InferencePanel from "./components/InferencePanel";
import ExportPanel from "./components/ExportPanel";
import AnnotationPanel from "./components/AnnotationPanel";
import NuRecPanel from "./components/NuRecPanel";
import SensitivityPanel from "./components/SensitivityPanel";
import SpatterPanel from "./components/SpatterPanel";
import type { Annotation, InferResponse, ReconstructResponse, SimulateResponse, TrajectoryData, SimRunResult } from "./api/types";

type Tab = "evidence" | "annotate" | "reconstruct" | "view" | "simulate" | "spatter" | "infer" | "sensitivity" | "nurec" | "export";

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

  // Evidence selection
  const [selection, setSelection] = useState<{
    imageDir: string;
    workspaceDir: string;
  } | null>(null);

  // Evidence images and annotations
  const [imagePaths, setImagePaths] = useState<string[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);

  // Reconstruction result
  const [result, setResult] = useState<ReconstructResponse | null>(null);

  // Simulation trajectories to overlay on the splat viewer
  const [trajectories, setTrajectories] = useState<TrajectoryData[]>([]);

  // PhysX log-likelihoods from simulation — passed to InferencePanel
  const [simLogLikelihoods, setSimLogLikelihoods] = useState<number[]>([]);

  // Full inference result and simulation result — passed to ExportPanel
  const [inferenceResult, setInferenceResult] = useState<InferResponse | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulateResponse | null>(null);

  const handleSelectionChange = useCallback(
    (sel: { imageDir: string; workspaceDir: string } | null) => {
      setSelection(sel);
    },
    []
  );

  const handleImagesChange = useCallback((images: string[]) => {
    setImagePaths(images);
  }, []);

  const handleAnnotationsChange = useCallback((anns: Annotation[]) => {
    setAnnotations(anns);
  }, []);

  const handleResult = useCallback((res: ReconstructResponse) => {
    setResult(res);
    setTab("view");
  }, []);

  const handleInferenceResult = useCallback((res: InferResponse) => {
    setInferenceResult(res);
  }, []);

  const handleSimResults = useCallback(
    (results: SimRunResult[], newTrajectories: TrajectoryData[]) => {
      setTrajectories(newTrajectories);
      // Derive per-scenario log-likelihoods from trajectory lengths as a proxy
      // (trajectory_length = 0 → effectively -inf; longer → better evidence of motion)
      const logLikelihoods = results.map((r) =>
        r.trajectory_length > 0 ? -1.0 / Math.max(r.trajectory_length, 1) : -10.0
      );
      setSimLogLikelihoods(logLikelihoods);
      setSimulationResult({ status: "success", results });
      // Switch to 3D view to show overlaid trajectories
      setTab("view");
    },
    []
  );

  const hasResult = result !== null;
  const plyPath = result?.ply_path ?? null;
  const usdPath = result?.usd_path ?? null;

  return (
    <div className="flex flex-col h-screen bg-[#09090b] text-zinc-200 overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-zinc-800 bg-zinc-950 shrink-0">
        <div className="flex items-center gap-3">
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
        <div className="text-[10px] font-mono text-zinc-700 select-none">v0.1.0 · Phase 4C</div>
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
          label="Annotate"
          active={tab === "annotate"}
          disabled={!selection}
          onClick={() => setTab("annotate")}
          badge={imagePaths.length > 0 ? `${annotations.length}A` : undefined}
        />
        <TabButton
          label="Reconstruct"
          active={tab === "reconstruct"}
          disabled={!selection}
          onClick={() => setTab("reconstruct")}
        />
        <TabButton
          label="3D View"
          active={tab === "view"}
          disabled={!hasResult}
          onClick={() => setTab("view")}
          badge={hasResult ? (trajectories.length > 0 ? `${trajectories.length}T` : "✓") : undefined}
        />
        <TabButton
          label="Simulate"
          active={tab === "simulate"}
          disabled={!hasResult}
          onClick={() => setTab("simulate")}
        />
        <TabButton
          label="Spatter"
          active={tab === "spatter"}
          onClick={() => setTab("spatter")}
        />
        <TabButton
          label="Infer"
          active={tab === "infer"}
          onClick={() => setTab("infer")}
          badge={simLogLikelihoods.length > 0 ? `${simLogLikelihoods.length}H` : undefined}
        />
        <TabButton
          label="Sensitivity"
          active={tab === "sensitivity"}
          onClick={() => setTab("sensitivity")}
          badge={inferenceResult ? "✓" : undefined}
        />
        <TabButton
          label="NuRec"
          active={tab === "nurec"}
          onClick={() => setTab("nurec")}
        />
        <TabButton
          label="Export"
          active={tab === "export"}
          onClick={() => setTab("export")}
        />
      </nav>

      {/* ── Main content ────────────────────────────────────────── */}
      <main className="flex-1 overflow-hidden flex flex-col">
        {/* ── Evidence tab ──────────────────────────────────────── */}
        {tab === "evidence" && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-4xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">Load Evidence</h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Select a folder of scene photographs and an output workspace for
                  reconstruction results.
                </p>
              </div>
              <EvidencePicker
                onSelectionChange={handleSelectionChange}
                onImagesChange={handleImagesChange}
              />
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
          </div>
        )}

        {/* ── Annotate tab ──────────────────────────────────────── */}
        {tab === "annotate" && selection && (
          <div className="flex-1 flex flex-col p-5 min-h-0">
            <div className="max-w-6xl mx-auto w-full flex flex-col flex-1 min-h-0 gap-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">Evidence Annotation</h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Draw regions of interest on evidence images and tag them. Annotations are
                  automatically fed into the inference pipeline as weighted evidence.
                </p>
              </div>
              <AnnotationPanel
                imagePaths={imagePaths}
                workspaceDir={selection.workspaceDir}
                onAnnotationsChange={handleAnnotationsChange}
              />
            </div>
          </div>
        )}

        {/* ── Reconstruct tab ───────────────────────────────────── */}
        {tab === "reconstruct" && selection && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-2xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">Reconstruction</h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Configure and run the COLMAP → Gaussian Splatting → USD pipeline.
                </p>
              </div>
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
              {result && (
                <div>
                  <ResultPanel result={result} />
                  <div className="flex justify-end mt-3">
                    <button
                      type="button"
                      onClick={() => setTab("view")}
                      className="px-5 py-2 rounded-lg bg-amber-500 hover:bg-amber-400
                                 text-zinc-950 font-semibold text-sm transition-colors"
                    >
                      Open in 3D View →
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── 3D View tab ───────────────────────────────────────── */}
        {tab === "view" && (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* path strip */}
            {plyPath && (
              <div className="shrink-0 px-4 py-2 border-b border-zinc-800 bg-zinc-950
                              flex items-center justify-between gap-4">
                <span className="text-[11px] font-mono text-zinc-500 truncate flex-1">
                  {plyPath}
                </span>
                {trajectories.length > 0 && (
                  <span className="text-[11px] text-amber-500 shrink-0">
                    {trajectories.length} trajectory overlay{trajectories.length !== 1 ? "s" : ""}
                  </span>
                )}
              </div>
            )}
            <div className="flex-1">
              <SplatViewer
                plyPath={plyPath}
                trajectories={trajectories}
                className="w-full h-full"
              />
            </div>
            {!plyPath && (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-zinc-600 text-sm">
                  No PLY file from reconstruction yet.
                </p>
              </div>
            )}
          </div>
        )}

        {/* ── Simulate tab ──────────────────────────────────────── */}
        {tab === "simulate" && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-2xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">Physics Simulation</h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Define objects and initial conditions, then run a Monte Carlo
                  PhysX simulation. Results are overlaid on the 3D scene.
                </p>
              </div>
              {usdPath && (
                <div className="flex gap-3 text-xs font-mono bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
                  <span className="text-zinc-600">usd:</span>
                  <span className="text-zinc-400 truncate flex-1">{usdPath}</span>
                </div>
              )}
              <ScenarioPanel usdPath={usdPath} onResults={handleSimResults} />
            </div>
          </div>
        )}

        {/* ── Spatter tab ───────────────────────────────────────── */}
        {tab === "spatter" && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-3xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">Blood Spatter Analysis</h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Simulate blood droplet ballistics using a physically-derived SPH model.
                  Visualise the floor impact map, classify the pattern type, and extract
                  the area-of-origin for Bayesian evidence integration.
                </p>
              </div>
              <SpatterPanel />
            </div>
          </div>
        )}

        {/* ── Infer tab ─────────────────────────────────────────── */}
        {tab === "infer" && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-3xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">
                  Probabilistic Inference
                </h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Define forensic hypotheses and rank them by Bayesian posterior
                  probability. PhysX simulation scores are pre-loaded when available.
                </p>
              </div>
              {simLogLikelihoods.length > 0 && (
                <div className="flex items-center gap-2 px-3 py-2 rounded
                                bg-amber-500/10 border border-amber-500/30 text-xs text-amber-400">
                  <span className="font-semibold">PhysX scores loaded:</span>
                  <span className="font-mono">
                    {simLogLikelihoods.map((l) => l.toFixed(3)).join(", ")}
                  </span>
                </div>
              )}
              <InferencePanel
                simulationLogLikelihoods={simLogLikelihoods}
                annotations={annotations}
                onResult={handleInferenceResult}
              />
            </div>
          </div>
        )}

        {/* ── Sensitivity tab ─────────────────────────────────────── */}
        {tab === "sensitivity" && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-3xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">
                  Sensitivity Analysis
                </h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Determine which evidence sources most influence the top-ranked
                  hypothesis. Uses leave-one-out re-ranking to quantify each
                  source's impact on posterior probability.
                </p>
              </div>
              <SensitivityPanel
                hypotheses={inferenceResult?.hypotheses ?? []}
              />
            </div>
          </div>
        )}

        {/* ── NuRec tab ───────────────────────────────────────────── */}
        {tab === "nurec" && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-3xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">
                  NVIDIA NuRec
                </h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Connect to a NuRec gRPC server, browse available neural
                  radiance scenes, and render photorealistic frames at arbitrary
                  camera poses.
                </p>
              </div>
              <NuRecPanel />
            </div>
          </div>
        )}

        {/* ── Export tab ─────────────────────────────────────────── */}
        {tab === "export" && (
          <div className="flex-1 overflow-y-auto p-5 animate-fade-up">
            <div className="max-w-3xl mx-auto space-y-4">
              <div>
                <h2 className="text-base font-semibold text-zinc-100">Export & Report</h2>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Generate a PDF case report, export the USD scene, or render a
                  flythrough video of the reconstruction and simulation.
                </p>
              </div>
              <ExportPanel
                imageDir={selection?.imageDir}
                workspaceDir={selection?.workspaceDir}
                reconstruction={result}
                simulation={simulationResult}
                inference={inferenceResult}
                trajectories={trajectories}
              />
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer className="shrink-0 px-5 py-2 border-t border-zinc-800 bg-zinc-950
                         flex items-center justify-between text-[10px] text-zinc-700">
        <span>ForenSim · Forensic Scene Reconstruction Platform</span>
        <span className="font-mono">Phase 4</span>
      </footer>
    </div>
  );
}

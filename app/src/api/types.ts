/**
 * Request payload to start a 3D reconstruction job.
 *
 * Mirrors the FastAPI `ReconstructRequest` Pydantic model.
 */
export interface ReconstructRequest {
  /** Directory containing the input images. */
  image_dir: string;
  /** Directory used for intermediate workspace files. */
  workspace_dir: string;
  /** Reconstruction algorithm to use. */
  method?: "gaussian_splatting" | "nerf" | "nurec";
  /** Feature matching strategy. */
  matcher?: "exhaustive" | "sequential";
  /** Maximum training steps (method-dependent). */
  max_steps?: number;
  /** Whether to fall back to Gaussian splatting when NeRF fails. */
  gsplat_fallback?: boolean;
}

/**
 * Response returned when a reconstruction job completes.
 *
 * Mirrors the FastAPI `ReconstructResponse` Pydantic model.
 */
export interface ReconstructResponse {
  /** Job outcome status. */
  status: "success" | "failed" | string;
  /** Absolute path to the generated USD scene. */
  usd_path: string;
  /** Absolute path to the generated PLY point cloud, if any. */
  ply_path: string | null;
  /** Total duration of the reconstruction job in seconds. */
  duration_seconds: number;
  /** Optional human-readable message, especially useful on failure. */
  message?: string;
}

/**
 * Health check response.
 *
 * Mirrors the FastAPI `HealthResponse` Pydantic model.
 */
export interface HealthResponse {
  /** Service status, e.g. "ok". */
  status: string;
  /** API version string. */
  version: string;
}

/**
 * Server-Sent Event describing reconstruction progress.
 *
 * Mirrors the FastAPI `ProgressEvent` Pydantic model.
 */
export interface ProgressEvent {
  /** Current processing step identifier. */
  step: string;
  /** Completion percentage in the range [0, 100]. */
  percent: number;
  /** Human-readable status message for the current step. */
  message: string;
}

// ── Simulation types ──────────────────────────────────────────────────────────

/** Initial conditions for one object in a simulation scenario. */
export interface ScenarioRequest {
  object_name: string;
  velocity: [number, number, number];
  angular_velocity: [number, number, number];
}

/** Request payload to run a Monte Carlo PhysX simulation. */
export interface SimulateRequest {
  usd_path: string;
  scenarios: ScenarioRequest[];
  n_steps?: number;
  dt?: number;
}

/** Per-scenario result returned by the simulation API. */
export interface SimRunResult {
  scenario: { object_name: string; velocity: number[] };
  final_positions: Record<string, number[]>;
  trajectory_length: number;
}

/** Top-level simulation response. */
export interface SimulateResponse {
  status: string;
  results: SimRunResult[];
}

/** A 3D trajectory overlay for the splat viewer. */
export interface TrajectoryData {
  id: string;
  color: string;
  points: [number, number, number][];
}

// ── Inference types ───────────────────────────────────────────────────────────

/** Request payload to rank forensic event hypotheses. */
export interface InferRequest {
  sequences: string[][];
  descriptions: string[];
  transition_matrix: number[][];
  initial_probs: number[];
  event_vocab: string[];
  physx_log_likelihoods?: number[];
  use_hmm?: boolean;
  emission_matrix?: number[][];
  annotations?: Annotation[];
  annotation_strength?: number;
}

/** Per-hypothesis result from the inference API. */
export interface HypothesisResult {
  rank: number;
  description: string;
  log_probability: number;
  posterior: number;
  events: string[];
  bayes_factor: number | null;
}

/** Top-level inference response. */
export interface InferResponse {
  status: string;
  hypotheses: HypothesisResult[];
  posterior_entropy: number | null;
  map_description: string | null;
}

// ── Sensitivity analysis types ────────────────────────────────────────────────

/** One evidence source descriptor for sensitivity analysis. */
export interface EvidenceSourceModel {
  /** Human-readable label, e.g. "blood_spatter_roi". */
  name: string;
  /** Per-hypothesis additive log-likelihood contribution. */
  log_likelihood_delta: number[];
  /** Current weighting factor (default 1.0). */
  weight?: number;
}

/** Request payload for the sensitivity analysis endpoint. */
export interface SensitivityRequest {
  /** Already-ranked hypotheses from a previous /api/infer/rank call. */
  hypotheses: HypothesisResult[];
  /** Evidence sources to analyse. */
  evidence_sources: EvidenceSourceModel[];
}

/** Result for one evidence source. */
export interface SensitivityResultItem {
  evidence_name: string;
  /** P(top hypothesis | all evidence). */
  baseline_top_posterior: number;
  /** P(top hypothesis | all evidence except this source). */
  loo_top_posterior: number;
  /** baseline_top_posterior - loo_top_posterior (positive = boosts top hyp). */
  impact: number;
  /** impact / baseline_top_posterior × 100. */
  impact_pct: number;
  /** How many positions the top hypothesis drops when this evidence is removed. */
  rank_change: number;
}

/** Top-level sensitivity response. */
export interface SensitivityResponse {
  status: string;
  results: SensitivityResultItem[];
  baseline_top_posterior: number;
  top_hypothesis: string | null;
}

// ── Annotation types ──────────────────────────────────────────────────────────

/** Supported annotation ROI shapes. */
export type AnnotationShape = "rect" | "polygon";

/** A single evidence ROI annotation. */
export interface Annotation {
  id: string;
  image_path: string;
  shape: AnnotationShape;
  /** Rectangle: [[x1, y1], [x2, y2]]. Polygon: [[x1, y1], [x2, y2], ...]. */
  coordinates: [number, number][];
  tag: string;
  description?: string;
  confidence?: number;
  metadata?: Record<string, unknown>;
}

/** Request payload to save annotations for a workspace. */
export interface SaveAnnotationsRequest {
  workspace_dir: string;
  annotations: Annotation[];
}

/** Response returned when annotations are saved. */
export interface SaveAnnotationsResponse {
  status: string;
  saved_path: string;
  count: number;
}

/** Response returned when annotations are loaded. */
export interface LoadAnnotationsResponse {
  status: string;
  annotations: Annotation[];
}

// ── NuRec types ───────────────────────────────────────────────────────────────

/** Health-check response for the NuRec gRPC server. */
export interface NuRecHealthResponse {
  status: string;
  address: string;
  reachable: boolean;
}

/** Metadata for one scene managed by the NuRec server. */
export interface NuRecSceneInfo {
  name: string;
  id: string;
  asset_path: string;
  description: string;
}

/** Response listing available NuRec scenes. */
export interface NuRecListScenesResponse {
  status: string;
  scenes: NuRecSceneInfo[];
}

/** Request to load a specific scene into NuRec. */
export interface NuRecLoadSceneRequest {
  scene_id: string;
  address?: string;
}

/** Response returned when a scene is loaded. */
export interface NuRecLoadSceneResponse {
  status: string;
  loaded: boolean;
  scene_id: string;
}

/** 6-DOF camera pose for NuRec rendering. */
export interface NuRecCameraPose {
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

/** Request payload to render a single frame via NuRec. */
export interface NuRecRenderRequest {
  scene_id: string;
  pose: NuRecCameraPose;
  width?: number;
  height?: number;
  address?: string;
}

/** Response returned when a frame is rendered. */
export interface NuRecRenderResponse {
  status: string;
  width: number;
  height: number;
  /** Base64-encoded PNG image data. */
  image_base64: string;
}

// ── Export / Report types ─────────────────────────────────────────────────────

/** Request payload to generate a PDF forensic report. */
export interface ReportRequest {
  case_title: string;
  examiner: string;
  notes?: string;
  output_path: string;
  reconstruction?: Record<string, unknown> | null;
  simulation?: Record<string, unknown> | null;
  inference?: Record<string, unknown> | null;
  screenshot_bytes?: string[];
}

/** Response returned when a PDF report is generated. */
export interface ReportResponse {
  status: string;
  output_path: string;
}

/** Request payload to package a USD scene into a zip archive. */
export interface UsdExportRequest {
  usd_path: string;
  output_path: string;
}

/** Response returned when a USD scene is packaged. */
export interface UsdExportResponse {
  status: string;
  output_path: string;
}

/** Request payload to generate a flythrough video. */
export interface VideoRequest {
  ply_path?: string | null;
  trajectories?: TrajectoryData[];
  output_path: string;
  duration_seconds?: number;
  fps?: number;
}

/** Response returned when a flythrough video is generated. */
export interface VideoResponse {
  status: string;
  output_path: string;
}

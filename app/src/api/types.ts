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

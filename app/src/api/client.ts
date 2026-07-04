import type {
  HealthResponse,
  InferRequest,
  InferResponse,
  LoadAnnotationsResponse,
  NuRecHealthResponse,
  NuRecListScenesResponse,
  NuRecLoadSceneRequest,
  NuRecLoadSceneResponse,
  NuRecRenderRequest,
  NuRecRenderResponse,
  ProgressEvent,
  ReconstructRequest,
  ReconstructResponse,
  ReportRequest,
  ReportResponse,
  SaveAnnotationsRequest,
  SaveAnnotationsResponse,
  SensitivityRequest,
  SensitivityResponse,
  SimulateRequest,
  SimulateResponse,
  UsdExportRequest,
  UsdExportResponse,
  VideoRequest,
  VideoResponse,
} from "./types";

/**
 * HTTP client for the ForenSim reconstruction API.
 *
 * All methods use the native `fetch` API and throw descriptive `Error`
 * objects when the server returns a non-2xx response or the network fails.
 */
export class ApiClient {
  /** Base URL of the reconstruction API. */
  readonly baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl =
      baseUrl ??
      import.meta.env?.VITE_API_URL ??
      "http://127.0.0.1:8008";
  }

  /**
   * Build a full URL from a relative API path.
   *
   * @param path - Relative path, e.g. `/health`.
   * @returns The full URL string.
   */
  private url(path: string): string {
    const base = this.baseUrl.replace(/\/$/, "");
    const cleanPath = path.startsWith("/") ? path : `/${path}`;
    return `${base}${cleanPath}`;
  }

  /**
   * Check whether an HTTP response is OK and parse JSON on success.
   *
   * @param response - The `fetch` Response object.
   * @returns The parsed JSON body.
   * @throws Error when the response status is not 2xx.
   */
  private async parseJson<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `API request failed: ${response.status} ${response.statusText}${body ? ` - ${body}` : ""}`
      );
    }
    return response.json() as Promise<T>;
  }

  /**
   * Perform a health check against the API.
   *
   * @returns The health status and version.
   * @throws Error when the request fails or the response is invalid.
   */
  async health(): Promise<HealthResponse> {
    const response = await fetch(this.url("/health"), {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    return this.parseJson<HealthResponse>(response);
  }

  /**
   * Start a 3D reconstruction job.
   *
   * @param req - Reconstruction request parameters.
   * @returns The reconstruction result including output paths and duration.
   * @throws Error when the request fails or the response is invalid.
   */
  async runReconstruction(
    req: ReconstructRequest
  ): Promise<ReconstructResponse> {
    const response = await fetch(this.url("/api/reconstruct/run"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<ReconstructResponse>(response);
  }

  /**
   * Subscribe to reconstruction progress events via Server-Sent Events.
   *
   * @param onEvent - Called for each progress message received.
   * @param onDone - Called when the stream closes normally.
   * @param onError - Called when the connection fails or a malformed event is received.
   * @returns A function that closes the EventSource connection.
   */
  streamProgress(
    onEvent: (e: ProgressEvent) => void,
    onDone: () => void,
    onError: (e: Error) => void
  ): () => void {
    const source = new EventSource(this.url("/api/reconstruct/progress"));

    source.addEventListener("message", (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data) as ProgressEvent;
        onEvent(parsed);
      } catch (err) {
        onError(
          new Error(
            `Failed to parse progress event: ${event.data} (${
              err instanceof Error ? err.message : String(err)
            })`
          )
        );
      }
    });

    source.addEventListener("error", () => {
      if (source.readyState === EventSource.CLOSED) {
        onDone();
        onError(new Error("Progress SSE connection closed unexpectedly."));
      } else if (source.readyState === EventSource.OPEN) {
        onError(new Error("Progress SSE connection encountered an error."));
      }
    });

    return () => {
      source.close();
    };
  }

  /**
   * Run a Monte Carlo PhysX simulation across multiple scenarios.
   *
   * @param req - Simulation request with USD path, scenarios, and step settings.
   * @returns Simulation response with per-scenario results.
   * @throws Error when the request fails or the response is invalid.
   */
  async runSimulation(req: SimulateRequest): Promise<SimulateResponse> {
    const response = await fetch(this.url("/api/simulate/run"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<SimulateResponse>(response);
  }

  /**
   * Run probabilistic hypothesis ranking via the Bayesian inference engine.
   *
   * @param req - Inference request with sequences, vocab, transition matrix, and optional PhysX scores.
   * @returns Ranked hypotheses with posteriors, Bayes factors, and entropy.
   */
  async runInference(req: InferRequest): Promise<InferResponse> {
    const response = await fetch(this.url("/api/infer/rank"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<InferResponse>(response);
  }

  /**
   * Run sensitivity analysis to determine which evidence sources most affect
   * the top-hypothesis posterior.
   *
   * @param req - Sensitivity request with ranked hypotheses and evidence sources.
   * @returns Per-evidence impact scores sorted by absolute influence.
   */
  async runSensitivity(req: SensitivityRequest): Promise<SensitivityResponse> {
    const response = await fetch(this.url("/api/infer/sensitivity"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<SensitivityResponse>(response);
  }

  /**
   * Save evidence ROI annotations to a workspace sidecar.
   *
   * @param req - Save request with workspace directory and annotations.
   * @returns Path to the saved annotation file and count.
   */
  async saveAnnotations(
    req: SaveAnnotationsRequest
  ): Promise<SaveAnnotationsResponse> {
    const response = await fetch(this.url("/api/annotate/save"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<SaveAnnotationsResponse>(response);
  }

  /**
   * Load evidence ROI annotations for a workspace.
   *
   * @param workspaceDir - Absolute workspace directory.
   * @returns Saved annotations for the workspace.
   */
  async loadAnnotations(
    workspaceDir: string
  ): Promise<LoadAnnotationsResponse> {
    const response = await fetch(
      this.url(
        `/api/annotate/load?workspace_dir=${encodeURIComponent(workspaceDir)}`
      ),
      {
        method: "GET",
        headers: { Accept: "application/json" },
      }
    );
    return this.parseJson<LoadAnnotationsResponse>(response);
  }

  // ── NuRec gRPC proxy ──────────────────────────────────────────────────────

  /**
   * Check connectivity to the NuRec gRPC server.
   *
   * @param address - NuRec server address (default: "localhost:8080").
   * @returns Reachability status and address.
   */
  async nurecHealth(address = "localhost:8080"): Promise<NuRecHealthResponse> {
    const response = await fetch(
      this.url(`/api/nurec/health?address=${encodeURIComponent(address)}`),
      {
        method: "GET",
        headers: { Accept: "application/json" },
      }
    );
    return this.parseJson<NuRecHealthResponse>(response);
  }

  /**
   * List scenes available on the NuRec server.
   *
   * @param address - NuRec server address.
   * @returns Available scenes metadata.
   */
  async nurecListScenes(
    address = "localhost:8080"
  ): Promise<NuRecListScenesResponse> {
    const response = await fetch(
      this.url(`/api/nurec/scenes?address=${encodeURIComponent(address)}`),
      {
        method: "GET",
        headers: { Accept: "application/json" },
      }
    );
    return this.parseJson<NuRecListScenesResponse>(response);
  }

  /**
   * Load a scene into the NuRec server's renderer.
   *
   * @param req - Load scene request with scene_id and optional server address.
   * @returns Whether the load succeeded.
   */
  async nurecLoadScene(
    req: NuRecLoadSceneRequest
  ): Promise<NuRecLoadSceneResponse> {
    const response = await fetch(this.url("/api/nurec/load"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<NuRecLoadSceneResponse>(response);
  }

  /**
   * Render a single frame at the given camera pose via NuRec.
   *
   * @param req - Render request with scene_id, pose, and resolution.
   * @returns Base64-encoded PNG image data.
   */
  async nurecRender(req: NuRecRenderRequest): Promise<NuRecRenderResponse> {
    const response = await fetch(this.url("/api/nurec/render"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<NuRecRenderResponse>(response);
  }

  // ── Export / Report ───────────────────────────────────────────────────────

  /**
   * Generate a PDF forensic report from case metadata and pipeline results.
   *
   * @param req - Report request with case title, examiner notes, and optional results.
   * @returns Path to the generated PDF.
   */
  async createReport(req: ReportRequest): Promise<ReportResponse> {
    const response = await fetch(this.url("/api/export/report"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<ReportResponse>(response);
  }

  /**
   * Package a USD scene and its sibling assets into a zip archive.
   *
   * @param req - USD export request with input scene and output zip paths.
   * @returns Path to the generated zip archive.
   */
  async exportUsd(req: UsdExportRequest): Promise<UsdExportResponse> {
    const response = await fetch(this.url("/api/export/usd"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<UsdExportResponse>(response);
  }

  /**
   * Download a file from the API server by absolute path.
   *
   * @param filePath - Absolute path on the server to the file to download.
   * @returns The file as a Blob.
   */
  async downloadFile(filePath: string): Promise<Blob> {
    const response = await fetch(
      this.url(`/api/export/download?path=${encodeURIComponent(filePath)}`),
      {
        method: "GET",
        headers: { Accept: "application/octet-stream" },
      }
    );
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `Download failed: ${response.status} ${response.statusText}${body ? ` - ${body}` : ""}`
      );
    }
    return response.blob();
  }

  /**
   * Generate an MP4 flythrough video from a 3D point cloud and trajectories.
   *
   * @param req - Video request with PLY path, trajectories, and render settings.
   * @returns Path to the generated MP4.
   */
  async createVideo(req: VideoRequest): Promise<VideoResponse> {
    const response = await fetch(this.url("/api/export/video"), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
    return this.parseJson<VideoResponse>(response);
  }
}

/** Default singleton API client instance. */
export const apiClient = new ApiClient();

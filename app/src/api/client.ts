import type {
  HealthResponse,
  ProgressEvent,
  ReconstructRequest,
  ReconstructResponse,
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
    const response = await fetch(this.url("/reconstruct/run"), {
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
    const source = new EventSource(this.url("/reconstruct/progress"));

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
   * Check whether the API is reachable.
   *
   * @returns `true` if the health endpoint responds, otherwise `false`.
   * Does not throw.
   */
  async checkApiAvailable(): Promise<boolean> {
    try {
      await this.health();
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Singleton API client used throughout the ForenSim frontend.
 *
 * The base URL defaults to `http://127.0.0.1:8008` and can be overridden by
 * setting the `VITE_API_URL` environment variable before the app is built.
 */
export const apiClient = new ApiClient();

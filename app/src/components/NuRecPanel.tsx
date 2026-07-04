import { useState, useCallback } from "react";
import { apiClient } from "../api/client";
import type {
  NuRecCameraPose,
  NuRecSceneInfo,
} from "../api/types";

// ── Types ──────────────────────────────────────────────────────────────────────

type ConnectionState = "idle" | "checking" | "connected" | "unreachable";

const DEFAULT_ADDRESS = "localhost:8080";

// ── Small shared UI pieces ─────────────────────────────────────────────────────

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-semibold text-amber-500 uppercase tracking-widest mb-2">
      {children}
    </h3>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className="text-zinc-600 w-28 shrink-0">{label}</span>
      <span className="text-zinc-300 truncate">{value}</span>
    </div>
  );
}

// ── Connection card ────────────────────────────────────────────────────────────

interface ConnectionCardProps {
  address: string;
  onAddressChange: (a: string) => void;
  state: ConnectionState;
  onConnect: () => void;
}

function ConnectionCard({
  address,
  onAddressChange,
  state,
  onConnect,
}: ConnectionCardProps) {
  const statusColor =
    state === "connected"
      ? "text-emerald-400"
      : state === "unreachable"
      ? "text-red-400"
      : state === "checking"
      ? "text-amber-400 animate-pulse"
      : "text-zinc-500";

  const statusLabel =
    state === "connected"
      ? "Connected"
      : state === "unreachable"
      ? "Unreachable"
      : state === "checking"
      ? "Checking…"
      : "Not connected";

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
      <SectionHeading>NuRec Server</SectionHeading>

      <div className="flex gap-2">
        <input
          type="text"
          value={address}
          onChange={(e) => onAddressChange(e.currentTarget.value)}
          placeholder="localhost:8080"
          className="flex-1 bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5
                     text-sm text-zinc-200 font-mono placeholder-zinc-600
                     focus:outline-none focus:border-amber-500"
        />
        <button
          type="button"
          onClick={onConnect}
          disabled={state === "checking"}
          className="px-3 py-1.5 rounded bg-amber-500 hover:bg-amber-400
                     text-zinc-950 font-semibold text-sm transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Connect
        </button>
      </div>

      <div className={`flex items-center gap-2 text-xs ${statusColor}`}>
        <span
          className={`w-2 h-2 rounded-full inline-block ${
            state === "connected"
              ? "bg-emerald-400"
              : state === "unreachable"
              ? "bg-red-400"
              : state === "checking"
              ? "bg-amber-400"
              : "bg-zinc-600"
          }`}
        />
        {statusLabel}
        {state === "unreachable" && (
          <span className="text-zinc-600 ml-1">
            — ensure NuRec Docker container is running
          </span>
        )}
      </div>

      {state !== "connected" && (
        <p className="text-[11px] text-zinc-600">
          Launch NuRec with:{" "}
          <code className="font-mono text-zinc-400">
            docker run --gpus all -p 8080:8080 nvcr.io/nvidia/nre-ga:latest
          </code>
        </p>
      )}
    </div>
  );
}

// ── Scene list card ────────────────────────────────────────────────────────────

interface SceneListCardProps {
  scenes: NuRecSceneInfo[];
  selectedId: string | null;
  loading: boolean;
  onRefresh: () => void;
  onSelect: (id: string) => void;
  onLoad: (id: string) => void;
  loadingSceneId: string | null;
}

function SceneListCard({
  scenes,
  selectedId,
  loading,
  onRefresh,
  onSelect,
  onLoad,
  loadingSceneId,
}: SceneListCardProps) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <SectionHeading>Available Scenes</SectionHeading>
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors
                     disabled:opacity-40 font-mono"
        >
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      {scenes.length === 0 && !loading && (
        <p className="text-xs text-zinc-600 italic">
          No scenes found. Connect to a NuRec server first.
        </p>
      )}

      <div className="space-y-1 max-h-48 overflow-y-auto">
        {scenes.map((scene) => (
          <div
            key={scene.id}
            onClick={() => onSelect(scene.id)}
            className={`flex items-center gap-3 px-3 py-2 rounded cursor-pointer
                        transition-colors group
                        ${
                          selectedId === scene.id
                            ? "bg-amber-500/15 border border-amber-500/40"
                            : "bg-zinc-950 border border-transparent hover:border-zinc-700"
                        }`}
          >
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold text-zinc-200 truncate">
                {scene.name}
              </div>
              {scene.description && (
                <div className="text-[10px] text-zinc-600 truncate">
                  {scene.description}
                </div>
              )}
              <div className="text-[10px] text-zinc-700 font-mono truncate">
                {scene.asset_path}
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onLoad(scene.id);
              }}
              disabled={loadingSceneId === scene.id}
              className="shrink-0 px-2 py-0.5 rounded text-[10px] font-semibold
                         bg-zinc-700 hover:bg-zinc-600 text-zinc-200 transition-colors
                         disabled:opacity-40"
            >
              {loadingSceneId === scene.id ? "Loading…" : "Load"}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Render card ────────────────────────────────────────────────────────────────

interface PoseEditorProps {
  pose: NuRecCameraPose;
  onChange: (pose: NuRecCameraPose) => void;
}

function PoseEditor({ pose, onChange }: PoseEditorProps) {
  const setPos = (i: number, v: number) => {
    const p: [number, number, number] = [...pose.position] as [number, number, number];
    p[i] = v;
    onChange({ ...pose, position: p });
  };

  const setQuat = (i: number, v: number) => {
    const q: [number, number, number, number] = [...pose.quaternion] as [number, number, number, number];
    q[i] = v;
    onChange({ ...pose, quaternion: q });
  };

  const labelClass = "text-[10px] text-zinc-600 font-mono w-5";
  const inputClass =
    "w-16 bg-zinc-950 border border-zinc-700 rounded px-1.5 py-0.5 text-xs " +
    "font-mono text-zinc-200 focus:outline-none focus:border-amber-500";

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[10px] text-zinc-500 w-16">Position</span>
        {(["X", "Y", "Z"] as const).map((axis, i) => (
          <label key={axis} className="flex items-center gap-1">
            <span className={labelClass}>{axis}</span>
            <input
              type="number"
              step={0.1}
              value={pose.position[i]}
              onChange={(e) => setPos(i, parseFloat(e.currentTarget.value) || 0)}
              className={inputClass}
            />
          </label>
        ))}
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[10px] text-zinc-500 w-16">Quaternion</span>
        {(["W", "X", "Y", "Z"] as const).map((axis, i) => (
          <label key={axis} className="flex items-center gap-1">
            <span className={labelClass}>{axis}</span>
            <input
              type="number"
              step={0.01}
              min={-1}
              max={1}
              value={pose.quaternion[i]}
              onChange={(e) => setQuat(i, parseFloat(e.currentTarget.value) || 0)}
              className={inputClass}
            />
          </label>
        ))}
      </div>
    </div>
  );
}

interface RenderCardProps {
  sceneId: string | null;
  address: string;
  connected: boolean;
}

function RenderCard({ sceneId, address, connected }: RenderCardProps) {
  const [pose, setPose] = useState<NuRecCameraPose>({
    position: [0, 1, 3],
    quaternion: [1, 0, 0, 0],
  });
  const [width, setWidth] = useState(1280);
  const [height, setHeight] = useState(720);
  const [rendering, setRendering] = useState(false);
  const [imageData, setImageData] = useState<string | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  const handleRender = useCallback(async () => {
    if (!sceneId || !connected) return;
    setRendering(true);
    setRenderError(null);
    try {
      const res = await apiClient.nurecRender({
        scene_id: sceneId,
        pose,
        width,
        height,
        address,
      });
      setImageData(`data:image/png;base64,${res.image_base64}`);
    } catch (e) {
      setRenderError(e instanceof Error ? e.message : String(e));
    } finally {
      setRendering(false);
    }
  }, [sceneId, connected, pose, width, height, address]);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-4">
      <SectionHeading>Render Frame</SectionHeading>

      {!connected && (
        <p className="text-xs text-zinc-600 italic">
          Connect to a NuRec server and load a scene to render.
        </p>
      )}

      {connected && (
        <>
          <PoseEditor pose={pose} onChange={setPose} />

          <div className="flex items-center gap-4 flex-wrap">
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              Width
              <input
                type="number"
                min={64}
                max={3840}
                step={64}
                value={width}
                onChange={(e) =>
                  setWidth(parseInt(e.currentTarget.value) || 1280)
                }
                className="w-20 bg-zinc-950 border border-zinc-700 rounded px-2 py-0.5
                           text-xs font-mono text-zinc-200 focus:outline-none focus:border-amber-500"
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              Height
              <input
                type="number"
                min={64}
                max={2160}
                step={64}
                value={height}
                onChange={(e) =>
                  setHeight(parseInt(e.currentTarget.value) || 720)
                }
                className="w-20 bg-zinc-950 border border-zinc-700 rounded px-2 py-0.5
                           text-xs font-mono text-zinc-200 focus:outline-none focus:border-amber-500"
              />
            </label>
            <button
              type="button"
              onClick={handleRender}
              disabled={rendering || !sceneId}
              className="px-4 py-1.5 rounded bg-amber-500 hover:bg-amber-400
                         text-zinc-950 font-semibold text-sm transition-colors
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {rendering ? "Rendering…" : "Render"}
            </button>
          </div>

          {renderError && (
            <p className="text-xs text-red-400 font-mono break-all">
              {renderError}
            </p>
          )}

          {imageData && (
            <div className="rounded overflow-hidden border border-zinc-700">
              <img
                src={imageData}
                alt="NuRec render"
                className="w-full object-contain max-h-96"
              />
              <div className="flex items-center justify-between px-3 py-1.5 bg-zinc-950">
                <span className="text-[10px] text-zinc-600 font-mono">
                  {width} × {height} px
                </span>
                <a
                  href={imageData}
                  download={`nurec_render_${Date.now()}.png`}
                  className="text-[10px] text-amber-500 hover:text-amber-400 font-semibold"
                >
                  Download PNG
                </a>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── NuRecPanel ────────────────────────────────────────────────────────────────

export default function NuRecPanel() {
  const [address, setAddress] = useState(DEFAULT_ADDRESS);
  const [connState, setConnState] = useState<ConnectionState>("idle");
  const [scenes, setScenes] = useState<NuRecSceneInfo[]>([]);
  const [scenesLoading, setScenesLoading] = useState(false);
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [loadingSceneId, setLoadingSceneId] = useState<string | null>(null);
  const [connError, setConnError] = useState<string | null>(null);

  const handleConnect = useCallback(async () => {
    setConnState("checking");
    setConnError(null);
    try {
      const res = await apiClient.nurecHealth(address);
      if (res.reachable) {
        setConnState("connected");
        // auto-load scene list
        setScenesLoading(true);
        try {
          const ls = await apiClient.nurecListScenes(address);
          setScenes(ls.scenes);
        } catch {
          setScenes([]);
        } finally {
          setScenesLoading(false);
        }
      } else {
        setConnState("unreachable");
      }
    } catch (e) {
      setConnState("unreachable");
      setConnError(e instanceof Error ? e.message : String(e));
    }
  }, [address]);

  const handleRefreshScenes = useCallback(async () => {
    if (connState !== "connected") return;
    setScenesLoading(true);
    try {
      const ls = await apiClient.nurecListScenes(address);
      setScenes(ls.scenes);
    } catch {
      setScenes([]);
    } finally {
      setScenesLoading(false);
    }
  }, [connState, address]);

  const handleLoadScene = useCallback(
    async (sceneId: string) => {
      setLoadingSceneId(sceneId);
      try {
        const res = await apiClient.nurecLoadScene({ scene_id: sceneId, address });
        if (res.loaded) {
          setSelectedSceneId(sceneId);
        }
      } catch {
        // keep existing selection
      } finally {
        setLoadingSceneId(null);
      }
    },
    [address]
  );

  const connected = connState === "connected";

  return (
    <div className="space-y-4">
      {/* Connection card */}
      <ConnectionCard
        address={address}
        onAddressChange={setAddress}
        state={connState}
        onConnect={handleConnect}
      />

      {connError && (
        <p className="text-xs text-red-400 font-mono px-1">{connError}</p>
      )}

      {/* Server info when connected */}
      {connected && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-1">
          <SectionHeading>Server Info</SectionHeading>
          <InfoRow label="address" value={address} />
          <InfoRow label="scenes found" value={String(scenes.length)} />
          {selectedSceneId && (
            <InfoRow label="loaded scene" value={selectedSceneId} />
          )}
        </div>
      )}

      {/* Scene list */}
      <SceneListCard
        scenes={scenes}
        selectedId={selectedSceneId}
        loading={scenesLoading}
        onRefresh={handleRefreshScenes}
        onSelect={setSelectedSceneId}
        onLoad={handleLoadScene}
        loadingSceneId={loadingSceneId}
      />

      {/* Render */}
      <RenderCard
        sceneId={selectedSceneId}
        address={address}
        connected={connected}
      />
    </div>
  );
}

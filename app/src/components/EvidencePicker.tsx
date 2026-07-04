import { useState, useEffect, useCallback } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";

interface EvidencePickerProps {
  onSelectionChange: (selection: { imageDir: string; workspaceDir: string } | null) => void;
}

const isTauri = () => {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
};

export default function EvidencePicker({ onSelectionChange }: EvidencePickerProps) {
  const [imageDir, setImageDir] = useState("");
  const [workspaceDir, setWorkspaceDir] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [tauriAvailable] = useState(isTauri());

  const browseImageDir = useCallback(async () => {
    if (!tauriAvailable) return;
    try {
      const dir = await open({ directory: true, multiple: false });
      if (typeof dir === "string") setImageDir(dir);
    } catch (e) {
      console.error("Failed to pick evidence folder:", e);
    }
  }, [tauriAvailable]);

  const browseWorkspaceDir = useCallback(async () => {
    if (!tauriAvailable) return;
    try {
      const dir = await open({ directory: true, multiple: false });
      if (typeof dir === "string") setWorkspaceDir(dir);
    } catch (e) {
      console.error("Failed to pick workspace folder:", e);
    }
  }, [tauriAvailable]);

  useEffect(() => {
    if (imageDir && workspaceDir) {
      onSelectionChange({ imageDir, workspaceDir });
    } else {
      onSelectionChange(null);
    }
  }, [imageDir, workspaceDir, onSelectionChange]);

  useEffect(() => {
    if (!imageDir || !tauriAvailable) {
      setImages([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    invoke<string[]>("list_images", { dir: imageDir })
      .then((paths) => {
        if (!cancelled) setImages(paths ?? []);
      })
      .catch((err) => {
        console.error("Failed to list images:", err);
        if (!cancelled) setImages([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [imageDir, tauriAvailable]);

  const clear = useCallback(() => {
    setImageDir("");
    setWorkspaceDir("");
    setImages([]);
  }, []);

  return (
    <div className="bg-gray-900 text-gray-100 rounded-lg p-6 shadow-lg border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-amber-500 font-mono">Evidence Workspace</h2>
        <button
          type="button"
          onClick={clear}
          className="px-3 py-1 text-sm rounded bg-gray-800 hover:bg-gray-700 text-amber-500 border border-gray-700 transition"
        >
          Clear
        </button>
      </div>

      {!tauriAvailable && (
        <div className="mb-4 p-3 rounded bg-gray-800 border border-amber-500/30 text-amber-500 text-sm">
          Tauri not available. Folder browsing and image listing are disabled.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="space-y-2">
          <label htmlFor="evidence-dir" className="block text-sm font-medium text-gray-400">
            Evidence Folder
          </label>
          <div className="flex gap-2">
            <input
              id="evidence-dir"
              type="text"
              value={imageDir}
              onChange={(e) => setImageDir(e.currentTarget.value)}
              placeholder="Path to evidence images"
              className="flex-1 bg-gray-950 border border-gray-800 rounded px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
            />
            <button
              type="button"
              onClick={browseImageDir}
              disabled={!tauriAvailable}
              className="px-3 py-2 rounded bg-amber-500 hover:bg-amber-600 disabled:bg-gray-800 disabled:text-gray-500 text-gray-950 font-medium text-sm transition"
            >
              Browse
            </button>
          </div>
        </div>

        <div className="space-y-2">
          <label htmlFor="workspace-dir" className="block text-sm font-medium text-gray-400">
            Workspace
          </label>
          <div className="flex gap-2">
            <input
              id="workspace-dir"
              type="text"
              value={workspaceDir}
              onChange={(e) => setWorkspaceDir(e.currentTarget.value)}
              placeholder="Output directory"
              className="flex-1 bg-gray-950 border border-gray-800 rounded px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
            />
            <button
              type="button"
              onClick={browseWorkspaceDir}
              disabled={!tauriAvailable}
              className="px-3 py-2 rounded bg-amber-500 hover:bg-amber-600 disabled:bg-gray-800 disabled:text-gray-500 text-gray-950 font-medium text-sm transition"
            >
              Browse
            </button>
          </div>
        </div>
      </div>

      {imageDir && (
        <div className="relative">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Preview</h3>
            <span className="px-2 py-0.5 rounded-full bg-gray-800 text-amber-500 text-xs font-mono">
              {images.length} image{images.length !== 1 ? "s" : ""}
            </span>
          </div>
          {loading ? (
            <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
              Scanning for images...
            </div>
          ) : images.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
              No valid images found in selected folder.
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 max-h-96 overflow-y-auto p-1">
              {images.map((path) => (
                <div
                  key={path}
                  className="relative aspect-square bg-gray-950 rounded overflow-hidden border border-gray-800"
                  title={path}
                >
                  <img
                    src={convertFileSrc(path)}
                    alt={path}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

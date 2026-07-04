import { useState, useEffect, useCallback } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { Skeleton } from "./ui/Skeleton";
import { Badge } from "./ui/Badge";

interface EvidencePickerProps {
  onSelectionChange: (selection: { imageDir: string; workspaceDir: string } | null) => void;
  onImagesChange?: (images: string[]) => void;
}

const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

function FolderInput({
  id,
  label,
  value,
  placeholder,
  onBrowse,
  onChange,
  tauriAvailable,
}: {
  id: string;
  label: string;
  value: string;
  placeholder: string;
  onBrowse: () => void;
  onChange: (v: string) => void;
  tauriAvailable: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-xs font-medium text-zinc-500 uppercase tracking-widest">
        {label}
      </label>
      <div className="flex gap-2">
        <input
          id={id}
          type="text"
          value={value}
          onChange={(e) => onChange(e.currentTarget.value)}
          placeholder={placeholder}
          className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2
                     text-sm font-mono text-zinc-300 placeholder-zinc-700
                     focus:outline-none focus:border-amber-500 transition-colors"
        />
        <button
          type="button"
          onClick={onBrowse}
          disabled={!tauriAvailable}
          className="px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700
                     disabled:opacity-40 disabled:cursor-not-allowed
                     text-zinc-300 text-sm font-medium transition-colors border border-zinc-700"
        >
          Browse
        </button>
      </div>
    </div>
  );
}

export default function EvidencePicker({
  onSelectionChange,
  onImagesChange,
}: EvidencePickerProps) {
  const [imageDir, setImageDir] = useState("");
  const [workspaceDir, setWorkspaceDir] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [tauriAvailable] = useState(isTauri);

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
        const imgs = paths ?? [];
        if (!cancelled) {
          setImages(imgs);
          onImagesChange?.(imgs);
        }
      })
      .catch((err) => {
        console.error("Failed to list images:", err);
        if (!cancelled) {
          setImages([]);
          onImagesChange?.([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [imageDir, tauriAvailable, onImagesChange]);

  const clear = useCallback(() => {
    setImageDir("");
    setWorkspaceDir("");
    setImages([]);
    onImagesChange?.([]);
  }, [onImagesChange]);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-5 animate-fade-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100">Evidence Workspace</h2>
          <p className="text-[11px] text-zinc-600 mt-0.5">
            Select a folder of crime scene photographs and an output workspace.
          </p>
        </div>
        {(imageDir || workspaceDir) && (
          <button
            type="button"
            onClick={clear}
            className="px-2.5 py-1 text-xs rounded-md bg-zinc-800 hover:bg-zinc-700
                       text-zinc-400 hover:text-zinc-200 border border-zinc-700 transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {!tauriAvailable && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/25 text-xs text-amber-400">
          <span className="mt-0.5 shrink-0">⚠</span>
          <span>
            Running in browser mode — folder browsing is disabled.
            Type paths manually or launch via{" "}
            <code className="font-mono bg-amber-500/10 px-1 rounded">npm run tauri dev</code>.
          </span>
        </div>
      )}

      {/* Folder inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FolderInput
          id="evidence-dir"
          label="Evidence Folder"
          value={imageDir}
          placeholder="D:\evidence\photos"
          onBrowse={browseImageDir}
          onChange={setImageDir}
          tauriAvailable={tauriAvailable}
        />
        <FolderInput
          id="workspace-dir"
          label="Workspace Output"
          value={workspaceDir}
          placeholder="D:\forensim\workspace"
          onBrowse={browseWorkspaceDir}
          onChange={setWorkspaceDir}
          tauriAvailable={tauriAvailable}
        />
      </div>

      {/* Validation status */}
      {(imageDir || workspaceDir) && (
        <div className="flex items-center gap-3 flex-wrap">
          <Badge variant={imageDir ? "success" : "neutral"}>
            {imageDir ? "✓ Images" : "⊘ Images"}
          </Badge>
          <Badge variant={workspaceDir ? "success" : "neutral"}>
            {workspaceDir ? "✓ Workspace" : "⊘ Workspace"}
          </Badge>
          {imageDir && workspaceDir && (
            <Badge variant="amber">Ready to reconstruct</Badge>
          )}
        </div>
      )}

      {/* Image grid preview */}
      {imageDir && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 justify-between">
            <span className="text-xs text-zinc-500 font-medium">Preview</span>
            {!loading && (
              <Badge variant={images.length > 0 ? "amber" : "neutral"}>
                {images.length} image{images.length !== 1 ? "s" : ""}
              </Badge>
            )}
          </div>

          {loading ? (
            <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-5 gap-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="aspect-square rounded-lg" />
              ))}
            </div>
          ) : images.length === 0 ? (
            <div className="flex items-center justify-center h-24 rounded-lg
                            bg-zinc-950 border border-zinc-800 border-dashed">
              <p className="text-xs text-zinc-700">No images found in this folder.</p>
            </div>
          ) : (
            <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-5 gap-2 max-h-72 overflow-y-auto">
              {images.map((path) => (
                <div
                  key={path}
                  title={path}
                  className="relative aspect-square bg-zinc-950 rounded-lg overflow-hidden
                             border border-zinc-800 hover:border-amber-500/50 transition-colors group"
                >
                  <img
                    src={convertFileSrc(path)}
                    alt={path.split(/[/\\]/).pop()}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-zinc-950/60 to-transparent
                                  opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-end p-1">
                    <span className="text-[9px] font-mono text-zinc-300 truncate w-full leading-tight">
                      {path.split(/[/\\]/).pop()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

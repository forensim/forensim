import { useState, useEffect, useRef, useCallback } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { apiClient } from "../api/client";
import type { Annotation } from "../api/types";

interface AnnotationPanelProps {
  imagePaths: string[];
  workspaceDir: string;
  onAnnotationsChange?: (annotations: Annotation[]) => void;
}

type Shape = "rect" | "polygon";

type Point = [number, number];

const TAGS = [
  "blood_spatter",
  "bullet_casing",
  "impact_point",
  "entry_wound",
  "glass_fracture",
  "other",
];

export default function AnnotationPanel({
  imagePaths,
  workspaceDir,
  onAnnotationsChange,
}: AnnotationPanelProps) {
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [shape, setShape] = useState<Shape>("rect");
  const [tag, setTag] = useState<string>(TAGS[0]);
  const [description, setDescription] = useState("");
  const [confidence, setConfidence] = useState<number>(1.0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load annotations on mount / workspace change
  useEffect(() => {
    if (!workspaceDir) return;
    setLoading(true);
    setError(null);
    apiClient
      .loadAnnotations(workspaceDir)
      .then((res) => {
        setAnnotations(res.annotations);
        onAnnotationsChange?.(res.annotations);
      })
      .catch((err) => {
        // No annotations file yet is fine
        if (err instanceof Error && err.message.includes("404")) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setLoading(false));
  }, [workspaceDir, onAnnotationsChange]);

  // Select first image by default
  useEffect(() => {
    if (!selectedImage && imagePaths.length > 0) {
      setSelectedImage(imagePaths[0]);
    }
  }, [imagePaths, selectedImage]);

  const imageAnnotations = annotations.filter(
    (a) => a.image_path === selectedImage
  );

  const canvasToImage = useCallback(
    (canvasX: number, canvasY: number): Point => {
      const canvas = canvasRef.current;
      const img = imageRef.current;
      if (!canvas || !img || !img.naturalWidth) return [0, 0];
      const rect = canvas.getBoundingClientRect();
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      return [canvasX * scaleX, canvasY * scaleY];
    },
    []
  );

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match canvas internal size to displayed size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw existing annotations
    imageAnnotations.forEach((a, idx) => {
      const color = tagColor(a.tag);
      ctx.strokeStyle = color;
      ctx.fillStyle = `${color}33`;
      ctx.lineWidth = 2;

      if (a.shape === "rect" && a.coordinates.length === 2) {
        const [p1, p2] = a.coordinates;
        const [x1, y1] = imageToCanvas(p1, canvas, img);
        const [x2, y2] = imageToCanvas(p2, canvas, img);
        const w = x2 - x1;
        const h = y2 - y1;
        ctx.fillRect(x1, y1, w, h);
        ctx.strokeRect(x1, y1, w, h);
      } else if (a.shape === "polygon" && a.coordinates.length >= 3) {
        ctx.beginPath();
        const [first, ...rest] = a.coordinates.map((p) =>
          imageToCanvas(p, canvas, img)
        );
        ctx.moveTo(first[0], first[1]);
        rest.forEach((p) => ctx.lineTo(p[0], p[1]));
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Label
      const label = `${idx + 1}. ${a.tag}`;
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      const textWidth = ctx.measureText(label).width;
      const firstPoint = imageToCanvas(a.coordinates[0], canvas, img);
      ctx.fillStyle = color;
      ctx.fillRect(firstPoint[0], firstPoint[1] - 16, textWidth + 8, 16);
      ctx.fillStyle = "#000";
      ctx.fillText(label, firstPoint[0] + 4, firstPoint[1] - 3);
    });

    // Draw draft polygon
    if (draftPoints.length > 0) {
      ctx.strokeStyle = "#f59e0b";
      ctx.fillStyle = "#f59e0b33";
      ctx.lineWidth = 2;
      if (shape === "rect" && draftPoints.length === 2) {
        const [p1, p2] = draftPoints;
        const [x1, y1] = imageToCanvas(p1, canvas, img);
        const [x2, y2] = imageToCanvas(p2, canvas, img);
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      } else if (shape === "polygon") {
        ctx.beginPath();
        const [first, ...rest] = draftPoints.map((p) =>
          imageToCanvas(p, canvas, img)
        );
        ctx.moveTo(first[0], first[1]);
        rest.forEach((p) => ctx.lineTo(p[0], p[1]));
        ctx.stroke();
      }
    }
  }, [imageAnnotations, draftPoints, shape]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas, selectedImage, annotations, draftPoints]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!selectedImage) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const pt = canvasToImage(e.clientX - rect.left, e.clientY - rect.top);
      setIsDrawing(true);
      if (shape === "rect") {
        setDraftPoints([pt, pt]);
      } else {
        setDraftPoints([pt]);
      }
    },
    [canvasToImage, selectedImage, shape]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDrawing) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const pt = canvasToImage(e.clientX - rect.left, e.clientY - rect.top);
      if (shape === "rect") {
        setDraftPoints((prev) => (prev.length >= 2 ? [prev[0], pt] : [pt, pt]));
      }
      // Polygon updates only on click
    },
    [canvasToImage, isDrawing, shape]
  );

  const handleMouseUp = useCallback(() => {
    if (shape === "rect") {
      setIsDrawing(false);
      if (draftPoints.length === 2) {
        finalizeAnnotation(draftPoints);
      }
    }
  }, [shape, draftPoints]);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (shape !== "polygon") return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const pt = canvasToImage(e.clientX - rect.left, e.clientY - rect.top);
      setDraftPoints((prev) => {
        if (prev.length === 0) return [pt];
        // Close polygon with double-click near first point
        const first = prev[0];
        const dx = pt[0] - first[0];
        const dy = pt[1] - first[1];
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (prev.length >= 3 && dist < 20) {
          finalizeAnnotation(prev);
          return [];
        }
        return [...prev, pt];
      });
    },
    [canvasToImage, shape]
  );

  const finalizeAnnotation = (points: Point[]) => {
    if (!selectedImage) return;
    const newAnnotation: Annotation = {
      id: crypto.randomUUID(),
      image_path: selectedImage,
      shape,
      coordinates: points,
      tag,
      description,
      confidence,
    };
    const updated = [...annotations, newAnnotation];
    setAnnotations(updated);
    onAnnotationsChange?.(updated);
    setDraftPoints([]);
    setDescription("");
    setConfidence(1.0);
  };

  const handleDelete = (id: string) => {
    const updated = annotations.filter((a) => a.id !== id);
    setAnnotations(updated);
    onAnnotationsChange?.(updated);
  };

  const handleSave = async () => {
    if (!workspaceDir) return;
    setLoading(true);
    setError(null);
    setSaved(false);
    try {
      await apiClient.saveAnnotations({
        workspace_dir: workspaceDir,
        annotations,
      });
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  if (imagePaths.length === 0) {
    return (
      <div className="text-sm text-zinc-500">
        No evidence images available. Select an evidence folder first.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 h-full">
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={shape}
          onChange={(e) => {
            setShape(e.target.value as Shape);
            setDraftPoints([]);
            setIsDrawing(false);
          }}
          className="px-3 py-2 rounded bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm"
        >
          <option value="rect">Rectangle</option>
          <option value="polygon">Polygon</option>
        </select>

        <select
          value={tag}
          onChange={(e) => setTag(e.target.value)}
          className="px-3 py-2 rounded bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm"
        >
          {TAGS.map((t) => (
            <option key={t} value={t}>
              {t.replace(/_/g, " ")}
            </option>
          ))}
        </select>

        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Description"
          className="px-3 py-2 rounded bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm w-48"
        />

        <div className="flex items-center gap-2 text-sm text-zinc-400">
          <span>Confidence</span>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.1}
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="w-8 text-right">{confidence.toFixed(1)}</span>
        </div>

        <button
          type="button"
          onClick={handleSave}
          disabled={loading || annotations.length === 0}
          className="px-4 py-2 rounded bg-amber-500 hover:bg-amber-400 disabled:bg-zinc-800 disabled:text-zinc-500 text-zinc-950 font-medium text-sm transition"
        >
          {loading ? "Saving…" : "Save Annotations"}
        </button>
      </div>

      {error && (
        <div className="px-3 py-2 rounded bg-red-500/10 border border-red-500/30 text-xs text-red-400">
          {error}
        </div>
      )}
      {saved && (
        <div className="px-3 py-2 rounded bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-400">
          Annotations saved.
        </div>
      )}

      <div className="flex flex-1 gap-4 min-h-0">
        {/* Image list */}
        <div className="w-48 shrink-0 flex flex-col gap-2 overflow-hidden">
          <h4 className="text-xs font-semibold text-zinc-500 uppercase">
            Images
          </h4>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {imagePaths.map((path) => (
              <button
                key={path}
                type="button"
                onClick={() => {
                  setSelectedImage(path);
                  setDraftPoints([]);
                  setIsDrawing(false);
                }}
                className={`w-full text-left p-2 rounded border text-xs transition ${
                  selectedImage === path
                    ? "bg-amber-500/10 border-amber-500/50 text-amber-400"
                    : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <div className="truncate font-mono">{path.split(/[/\\]/).pop()}</div>
                <div className="text-[10px] text-zinc-600 mt-0.5">
                  {annotations.filter((a) => a.image_path === path).length} annotations
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Canvas viewer */}
        <div className="flex-1 flex flex-col min-h-0">
          <div
            ref={containerRef}
            className="relative flex-1 bg-zinc-950 rounded border border-zinc-800 overflow-hidden flex items-center justify-center"
          >
            {selectedImage && (
              <>
                <img
                  ref={imageRef}
                  src={convertFileSrc(selectedImage)}
                  alt={selectedImage}
                  className="max-w-full max-h-full object-contain"
                  onLoad={() => drawCanvas()}
                />
                <canvas
                  ref={canvasRef}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onClick={handleCanvasClick}
                  className="absolute inset-0 w-full h-full cursor-crosshair"
                />
              </>
            )}
          </div>
          <p className="text-[11px] text-zinc-600 mt-2">
            {shape === "rect"
              ? "Click and drag to draw a rectangle."
              : "Click points to draw a polygon; click near the first point to close it."}
          </p>
        </div>

        {/* Annotation list */}
        <div className="w-56 shrink-0 flex flex-col gap-2 overflow-hidden">
          <h4 className="text-xs font-semibold text-zinc-500 uppercase">
            Annotations
          </h4>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {imageAnnotations.length === 0 ? (
              <div className="text-xs text-zinc-600">
                No annotations on this image.
              </div>
            ) : (
              imageAnnotations.map((a, idx) => (
                <div
                  key={a.id}
                  className="p-2 rounded bg-zinc-900 border border-zinc-800 text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: tagColor(a.tag) }}
                    />
                    <span className="font-medium text-zinc-300">
                      {idx + 1}. {a.tag.replace(/_/g, " ")}
                    </span>
                  </div>
                  {a.description && (
                    <div className="text-zinc-500 mt-1 truncate">{a.description}</div>
                  )}
                  <div className="text-zinc-600 mt-1">c={(a.confidence ?? 1.0).toFixed(1)}</div>
                  <button
                    type="button"
                    onClick={() => handleDelete(a.id)}
                    className="mt-2 text-red-400 hover:text-red-300"
                  >
                    Delete
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function imageToCanvas(
  point: number[],
  canvas: HTMLCanvasElement,
  img: HTMLImageElement
): Point {
  const rect = canvas.getBoundingClientRect();
  const scaleX = rect.width / img.naturalWidth;
  const scaleY = rect.height / img.naturalHeight;
  return [point[0] * scaleX, point[1] * scaleY];
}

function tagColor(tag: string): string {
  const colors: Record<string, string> = {
    blood_spatter: "#ef4444",
    bullet_casing: "#f59e0b",
    impact_point: "#3b82f6",
    entry_wound: "#a855f7",
    glass_fracture: "#10b981",
    other: "#94a3b8",
  };
  return colors[tag] ?? "#94a3b8";
}

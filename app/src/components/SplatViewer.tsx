import { useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import * as THREE from "three";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - gaussian-splats-3d may lack type definitions in some environments
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import type { TrajectoryData } from "../api/types";

// ---------- Types ----------

export interface SplatViewerProps {
  plyPath: string | null;
  trajectories?: TrajectoryData[];
  className?: string;
}

// ---------- Component ----------

export default function SplatViewer({
  plyPath,
  trajectories,
  className,
}: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const trajectoryGroupRef = useRef<THREE.Group | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);

  // --- Main viewer lifecycle ---
  useEffect(() => {
    setError(null);
    setWarning(null);
    setLoading(false);

    // Cleanup previous viewer
    if (viewerRef.current) {
      try {
        viewerRef.current.dispose();
      } catch {
        // ignore dispose errors
      }
      viewerRef.current = null;
    }
    if (trajectoryGroupRef.current) {
      trajectoryGroupRef.current = null;
    }

    if (!plyPath) return;

    // Validate file extension
    const ext = plyPath.split(".").pop()?.toLowerCase();
    if (ext !== "ply" && ext !== "splat") {
      setWarning(
        `Unsupported file format ".${ext}". Expected .ply or .splat file.`
      );
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    let disposed = false;

    const init = async () => {
      setLoading(true);
      try {
        // Convert local filesystem path to asset:// URL for Tauri WebView
        const assetUrl = convertFileSrc(plyPath);

        const viewer = new GaussianSplats3D.Viewer({
          cameraUp: [0, 1, 0],
          initialCameraPosition: [0, 0, 5],
          initialCameraLookAt: [0, 0, 0],
          rootElement: container,
          sceneRevealMode: GaussianSplats3D.SceneRevealMode.Gradual,
          threeScene: undefined,
          selfDrivenMode: true,
        });

        if (disposed) {
          viewer.dispose();
          return;
        }

        viewerRef.current = viewer;

        await viewer.addSplatScene(assetUrl, {
          splatAlphaRemovalThreshold: 5,
        });

        if (disposed) {
          viewer.dispose();
          viewerRef.current = null;
          return;
        }

        viewer.start();

        // Create a group for trajectory overlays
        const group = new THREE.Group();
        group.name = "trajectories";
        if (viewer.threeScene) {
          viewer.threeScene.add(group);
        }
        trajectoryGroupRef.current = group;

        setLoading(false);
      } catch (err: unknown) {
        if (!disposed) {
          const message =
            err instanceof Error ? err.message : "Unknown error occurred";
          setError(message);
          setLoading(false);
        }
      }
    };

    init();

    return () => {
      disposed = true;
      if (viewerRef.current) {
        try {
          viewerRef.current.dispose();
        } catch {
          // ignore dispose errors
        }
        viewerRef.current = null;
      }
      trajectoryGroupRef.current = null;
    };
  }, [plyPath]);

  // --- Trajectory overlay updates ---
  useEffect(() => {
    const group = trajectoryGroupRef.current;
    if (!group) return;

    // Clear existing trajectory lines
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if (child instanceof THREE.Line) {
        child.geometry.dispose();
        if (child.material instanceof THREE.Material) {
          child.material.dispose();
        }
      }
    }

    // Add new trajectories
    if (trajectories && trajectories.length > 0) {
      for (const traj of trajectories) {
        if (traj.points.length < 2) continue;

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(traj.points.length * 3);
        for (let i = 0; i < traj.points.length; i++) {
          positions[i * 3] = traj.points[i][0];
          positions[i * 3 + 1] = traj.points[i][1];
          positions[i * 3 + 2] = traj.points[i][2];
        }
        geometry.setAttribute(
          "position",
          new THREE.BufferAttribute(positions, 3)
        );

        const material = new THREE.LineBasicMaterial({
          color: new THREE.Color(traj.color),
          linewidth: 2,
        });

        const line = new THREE.Line(geometry, material);
        line.name = `trajectory-${traj.id}`;
        group.add(line);
      }
    }
  }, [trajectories]);

  // --- Resize handling ---
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (viewerRef.current && viewerRef.current.renderer) {
          viewerRef.current.renderer.setSize(width, height);
        }
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // --- Render ---

  // Empty state
  if (!plyPath && !error && !warning) {
    return (
      <div
        className={`relative w-full h-full min-h-[400px] bg-zinc-950 flex items-center justify-center ${className ?? ""}`}
      >
        <p className="text-zinc-500 text-sm">No scene loaded</p>
      </div>
    );
  }

  // Warning state (invalid extension)
  if (warning) {
    return (
      <div
        className={`relative w-full h-full min-h-[400px] bg-zinc-950 flex items-center justify-center p-4 ${className ?? ""}`}
      >
        <div className="bg-yellow-950 border border-yellow-800 text-yellow-400 rounded-md p-4 max-w-md text-center text-sm">
          {warning}
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full min-h-[400px] bg-zinc-950 ${className ?? ""}`}
    >
      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/80 z-10">
          <div className="flex flex-col items-center gap-2">
            <div className="w-6 h-6 border-2 border-zinc-400 border-t-transparent rounded-full animate-spin" />
            <p className="text-zinc-400 text-sm">Loading splat...</p>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center p-4 z-10">
          <div className="bg-red-950 border border-red-800 text-red-400 rounded-md p-4 max-w-md text-center text-sm">
            {error}
          </div>
        </div>
      )}

      {/* Controls hint */}
      {!loading && !error && (
        <div className="absolute bottom-2 left-2 z-10">
          <p className="text-zinc-600 text-xs">
            Drag to orbit · Scroll to zoom · Right-drag to pan
          </p>
        </div>
      )}
    </div>
  );
}

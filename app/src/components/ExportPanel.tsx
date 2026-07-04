import { useState, useCallback } from "react";
import { apiClient } from "../api/client";
import type {
  InferResponse,
  ReconstructResponse,
  SimulateResponse,
  TrajectoryData,
} from "../api/types";

interface ExportPanelProps {
  imageDir?: string | null;
  workspaceDir?: string | null;
  reconstruction: ReconstructResponse | null;
  simulation: SimulateResponse | null;
  inference: InferResponse | null;
  trajectories: TrajectoryData[];
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function ExportPanel({
  imageDir,
  workspaceDir,
  reconstruction,
  simulation,
  inference,
  trajectories,
}: ExportPanelProps) {
  const [caseTitle, setCaseTitle] = useState("");
  const [examiner, setExaminer] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastPath, setLastPath] = useState<string | null>(null);

  const outputDir = workspaceDir ?? "D:/forensim/workspace";
  const safeCaseTitle = caseTitle.trim() || "forensim-report";
  const reportPath = `${outputDir}/${safeCaseTitle}.pdf`;
  const usdZipPath = `${outputDir}/${safeCaseTitle}-usd.zip`;
  const videoPath = `${outputDir}/${safeCaseTitle}-flythrough.mp4`;

  const handleGenerateReport = useCallback(async () => {
    setBusy("report");
    setError(null);
    try {
      const res = await apiClient.createReport({
        case_title: caseTitle.trim() || "ForenSim Report",
        examiner: examiner.trim() || "Forensic Analyst",
        notes,
        output_path: reportPath,
        reconstruction: reconstruction as Record<string, unknown> | null,
        simulation: simulation as Record<string, unknown> | null,
        inference: inference as Record<string, unknown> | null,
      });
      setLastPath(res.output_path);
      const blob = await apiClient.downloadFile(res.output_path);
      downloadBlob(blob, `${safeCaseTitle}.pdf`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }, [caseTitle, examiner, notes, reportPath, reconstruction, simulation, inference, safeCaseTitle]);

  const handleExportUsd = useCallback(async () => {
    if (!reconstruction?.usd_path) {
      setError("No USD scene available. Run reconstruction first.");
      return;
    }
    setBusy("usd");
    setError(null);
    try {
      const res = await apiClient.exportUsd({
        usd_path: reconstruction.usd_path,
        output_path: usdZipPath,
      });
      setLastPath(res.output_path);
      const blob = await apiClient.downloadFile(res.output_path);
      downloadBlob(blob, `${safeCaseTitle}-usd.zip`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }, [reconstruction, usdZipPath, safeCaseTitle]);

  const handleGenerateVideo = useCallback(async () => {
    if (!reconstruction?.ply_path && trajectories.length === 0) {
      setError("No PLY file or trajectories available for video generation.");
      return;
    }
    setBusy("video");
    setError(null);
    try {
      const res = await apiClient.createVideo({
        ply_path: reconstruction?.ply_path,
        trajectories,
        output_path: videoPath,
        duration_seconds: 5.0,
        fps: 30,
      });
      setLastPath(res.output_path);
      const blob = await apiClient.downloadFile(res.output_path);
      downloadBlob(blob, `${safeCaseTitle}-flythrough.mp4`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }, [reconstruction, trajectories, videoPath, safeCaseTitle]);

  return (
    <div className="space-y-5">
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-zinc-100">Case Metadata</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-[11px] font-medium text-zinc-500 mb-1">
              Case Title
            </label>
            <input
              type="text"
              value={caseTitle}
              onChange={(e) => setCaseTitle(e.target.value)}
              placeholder="e.g. Case 2026-001"
              className="w-full px-3 py-2 rounded bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm focus:outline-none focus:border-amber-500/50"
            />
          </div>
          <div>
            <label className="block text-[11px] font-medium text-zinc-500 mb-1">
              Examiner
            </label>
            <input
              type="text"
              value={examiner}
              onChange={(e) => setExaminer(e.target.value)}
              placeholder="e.g. Forensic Analyst"
              className="w-full px-3 py-2 rounded bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm focus:outline-none focus:border-amber-500/50"
            />
          </div>
        </div>
        <div>
          <label className="block text-[11px] font-medium text-zinc-500 mb-1">
            Notes
          </label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Additional case notes..."
            rows={3}
            className="w-full px-3 py-2 rounded bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm focus:outline-none focus:border-amber-500/50 resize-none"
          />
        </div>
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-zinc-100">Export Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <button
            type="button"
            onClick={handleGenerateReport}
            disabled={busy !== null}
            className="flex flex-col items-start gap-1 px-4 py-3 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-amber-500/40 transition-colors text-left disabled:opacity-50"
          >
            <span className="text-sm font-medium text-zinc-200">
              {busy === "report" ? "Generating…" : "Generate PDF Report"}
            </span>
            <span className="text-[11px] text-zinc-500">
              Case summary, evidence, hypotheses
            </span>
          </button>

          <button
            type="button"
            onClick={handleExportUsd}
            disabled={busy !== null || !reconstruction?.usd_path}
            className="flex flex-col items-start gap-1 px-4 py-3 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-amber-500/40 transition-colors text-left disabled:opacity-50"
          >
            <span className="text-sm font-medium text-zinc-200">
              {busy === "usd" ? "Packaging…" : "Export USD Scene"}
            </span>
            <span className="text-[11px] text-zinc-500">
              Zip archive of scene + assets
            </span>
          </button>

          <button
            type="button"
            onClick={handleGenerateVideo}
            disabled={busy !== null || (!reconstruction?.ply_path && trajectories.length === 0)}
            className="flex flex-col items-start gap-1 px-4 py-3 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-amber-500/40 transition-colors text-left disabled:opacity-50"
          >
            <span className="text-sm font-medium text-zinc-200">
              {busy === "video" ? "Rendering…" : "Generate Flythrough"}
            </span>
            <span className="text-[11px] text-zinc-500">
              MP4 video from point cloud + trajectories
            </span>
          </button>
        </div>
      </div>

      {error && (
        <div className="px-3 py-2 rounded bg-red-500/10 border border-red-500/30 text-xs text-red-400">
          {error}
        </div>
      )}

      {lastPath && !error && (
        <div className="px-3 py-2 rounded bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-400">
          Saved to backend: <span className="font-mono">{lastPath}</span>
        </div>
      )}

      <div className="space-y-2 text-[11px] text-zinc-600 border-t border-zinc-800 pt-3">
        <p>Export destinations are derived from the current workspace.</p>
        <p className="font-mono">{outputDir}</p>
        {imageDir && <p className="font-mono">Images: {imageDir}</p>}
      </div>
    </div>
  );
}

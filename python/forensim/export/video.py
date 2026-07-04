"""Generate an MP4 flythrough video from a 3D trajectory and optional point cloud."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _load_point_cloud_sample(ply_path: Path, max_points: int = 20_000) -> np.ndarray | None:
    """Load a subset of points from a PLY file using open3d."""
    try:
        import open3d as o3d  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("open3d not available; point cloud will not be rendered")
        return None

    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return None
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        return pts
    except Exception as exc:
        logger.warning("Failed to load point cloud: %s", exc)
        return None


def _build_trajectories(trajectories: list[dict[str, Any]]) -> list[tuple[np.ndarray, str]]:
    out: list[tuple[np.ndarray, str]] = []
    for t in trajectories:
        pts = t.get("points")
        if not pts:
            continue
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        out.append((arr, t.get("color", "#f59e0b")))
    return out


def generate_flythrough_video(
    output_path: Path,
    trajectories: list[dict[str, Any]] | None = None,
    ply_path: Path | None = None,
    duration_seconds: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Render a 3D scene flythrough MP4 to ``output_path``.

    Args:
        output_path: Destination MP4 file.
        trajectories: Optional list of trajectory objects with ``points`` and ``color``.
        ply_path: Optional point cloud PLY to render in the background.
        duration_seconds: Length of the output video.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Path to the generated MP4.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d import proj3d  # type: ignore[import-untyped]  # noqa: F401

    traj_data = _build_trajectories(trajectories or [])
    cloud = _load_point_cloud_sample(ply_path) if ply_path else None

    all_points_list: list[list[float]] = []
    for arr, _ in traj_data:
        all_points_list.extend(arr.tolist())
    if cloud is not None:
        all_points_list.extend(cloud.tolist())
    if not all_points_list:
        raise ValueError("No trajectory or point cloud data provided")

    all_points = np.array(all_points_list)
    center = all_points.mean(axis=0)
    max_extent = float(np.max(np.ptp(all_points, axis=0)))
    radius = max_extent * 1.2 if max_extent > 0 else 5.0

    n_frames = int(duration_seconds * fps)
    angle_step = 2 * np.pi / max(n_frames, 1)

    import imageio

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=5.0)

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    for i in range(n_frames):
        ax.clear()
        ax.set_facecolor("#111111")
        ax.set_axis_off()

        angle = i * angle_step
        elev = 20 + 10 * np.sin(angle * 2)
        azim = np.degrees(angle)
        ax.view_init(elev=elev, azim=azim)

        # Point cloud
        if cloud is not None:
            ax.scatter(
                cloud[:, 0],
                cloud[:, 1],
                cloud[:, 2],
                c="#52525b",
                s=1,
                alpha=0.4,
                depthshade=False,
            )

        # Trajectories
        for arr, color in traj_data:
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, linewidth=2)
            ax.scatter(arr[0, 0], arr[0, 1], arr[0, 2], color="white", s=40, marker="o")
            ax.scatter(arr[-1, 0], arr[-1, 1], arr[-1, 2], color=color, s=60, marker="x")

        # Set equal aspect limits
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        # Title overlay
        ax.text2D(
            0.02,
            0.98,
            "ForenSim Flythrough",
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            verticalalignment="top",
        )

        fig.canvas.draw()
        canvas = fig.canvas
        assert isinstance(canvas, FigureCanvasAgg)
        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[no-untyped-call]
        frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        writer.append_data(frame)

    writer.close()
    plt.close(fig)
    logger.info("Flythrough video written to %s", output_path)
    return output_path

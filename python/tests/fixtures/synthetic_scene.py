"""
Generate a small synthetic image dataset for reconstruction tests.

Creates textured images of a simple 3D scene from multiple camera viewpoints.
The scene is designed to produce enough SIFT feature matches for COLMAP to
reconstruct a small sparse model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _rotation_matrix(theta: float, axis: str) -> np.ndarray:
    """Return a 3D rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(axis)


def _build_scene_points(grid_size: int = 20, spread: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a structured point cloud representing the scene.

    Returns:
        xyz: (N, 3) float64 point positions.
        rgb: (N, 3) uint8 point colors.
    """
    points = []
    colors = []
    rng = np.random.default_rng(42)

    # Textured ground plane: grid of points.
    xs = np.linspace(-spread, spread, grid_size)
    ys = np.linspace(-spread, spread, grid_size)
    for x in xs:
        for y in ys:
            # Checkerboard-ish color.
            color = np.array([0, 0, 0], dtype=np.uint8)
            if (int((x + spread) * 3) + int((y + spread) * 3)) % 2 == 0:
                color = np.array([180, 120, 80], dtype=np.uint8)
            else:
                color = np.array([100, 140, 90], dtype=np.uint8)
            noise = rng.integers(-20, 20, size=3)
            color = np.clip(color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            points.append([x, y, 0.0])
            colors.append(color)

    # Random 3D feature points scattered in the volume.
    num_random = 400
    for _ in range(num_random):
        x = rng.uniform(-spread, spread)
        y = rng.uniform(-spread, spread)
        z = rng.uniform(0.1, 1.5)
        color = rng.integers(0, 256, size=3).astype(np.uint8)
        points.append([x, y, z])
        colors.append(color)

    # A few larger colored boxes as distinct objects.
    box_centers = [
        np.array([-1.0, -1.0, 0.6]),
        np.array([1.2, -0.8, 0.5]),
        np.array([0.0, 1.0, 0.7]),
        np.array([-1.5, 1.2, 0.4]),
        np.array([1.5, 1.5, 0.6]),
    ]
    box_colors = [
        np.array([200, 50, 50], dtype=np.uint8),
        np.array([50, 200, 50], dtype=np.uint8),
        np.array([50, 50, 200], dtype=np.uint8),
        np.array([200, 200, 50], dtype=np.uint8),
        np.array([200, 50, 200], dtype=np.uint8),
    ]
    for center, color in zip(box_centers, box_colors):
        for dx in np.linspace(-0.25, 0.25, 6):
            for dy in np.linspace(-0.25, 0.25, 6):
                for dz in np.linspace(-0.25, 0.25, 6):
                    points.append(center + np.array([dx, dy, dz]))
                    colors.append(color)

    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.uint8)


def _render_points(
    points: np.ndarray,
    colors: np.ndarray,
    camera_pos: np.ndarray,
    camera_rot: np.ndarray,
    focal: float,
    width: int,
    height: int,
) -> Image.Image:
    """Render the scene points into a single image."""
    # Camera look-at.
    forward = -camera_pos / (np.linalg.norm(camera_pos) + 1e-8)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    camera_rot = np.column_stack((right, up, forward))

    # Project all points.
    local = (camera_rot.T @ (points - camera_pos).T).T
    visible = local[:, 2] > 0.1
    local = local[visible]
    pts_colors = colors[visible]

    x = focal * local[:, 0] / local[:, 2] + width * 0.5
    y = focal * local[:, 1] / local[:, 2] + height * 0.5

    # Sort by depth (far first) for simple occlusion.
    order = np.argsort(local[:, 2])[::-1]
    x, y, pts_colors = x[order], y[order], pts_colors[order]

    img = Image.new("RGB", (width, height), (220, 220, 220))
    draw = ImageDraw.Draw(img)

    # Draw points as small squares with size based on depth.
    for xi, yi, ci, depth in zip(x, y, pts_colors, local[order, 2]):
        size = max(1, int(6 / depth))
        px, py = int(xi), int(yi)
        if 0 <= px < width and 0 <= py < height:
            draw.rectangle([px - size, py - size, px + size, py + size], fill=tuple(ci.tolist()))

    # Add extra high-frequency dots and lines for SIFT.
    rng = np.random.default_rng(abs(int((camera_pos[0] + camera_pos[1]) * 1000)) % (2**31))
    for _ in range(300):
        px = rng.integers(0, width)
        py = rng.integers(0, height)
        color = tuple(rng.integers(0, 256, size=3).tolist())
        draw.point((px, py), fill=color)

    return img


def generate_synthetic_dataset(
    output_dir: Path,
    num_images: int = 12,
    width: int = 800,
    height: int = 600,
    radius: float = 2.5,
    seed: int = 42,
) -> Path:
    """
    Generate a synthetic image dataset suitable for COLMAP.

    Args:
        output_dir: Directory where images will be written.
        num_images: Number of views to generate.
        width, height: Image resolution.
        radius: Camera orbit radius in meters.
        seed: Random seed.

    Returns:
        Path to the image directory.
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    points, colors = _build_scene_points()
    focal = width * 0.8

    for i in range(num_images):
        angle = 2 * math.pi * i / num_images
        camera_pos = np.array([radius * math.cos(angle), radius * math.sin(angle), 1.5])
        img = _render_points(points, colors, camera_pos, np.eye(3), focal, width, height)
        img.save(image_dir / f"frame_{i:03d}.jpg", quality=95)

    (output_dir / "cameras.txt").write_text(
        f"# Synthetic dataset generated by synthetic_scene.py\n"
        f"num_images: {num_images}\n"
        f"width: {width}\n"
        f"height: {height}\n"
        f"focal: {focal}\n",
        encoding="utf-8",
    )

    return image_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=12)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()
    generate_synthetic_dataset(args.output_dir, args.num_images, args.width, args.height)
    print(f"Generated dataset at {args.output_dir / 'images'}")

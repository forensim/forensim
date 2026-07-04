#!/usr/bin/env python3
"""
ForenSim — Sample Dataset Generator
=====================================
Generates synthetic forensic scene images for the built-in demo dataset.

Usage:
    python scripts/generate_sample_dataset.py

Requirements:
    numpy, Pillow  (both present in the project venv)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_DIR = REPO_ROOT / "assets" / "sample-scenes" / "crime-scene-01"
IMAGES_DIR = SCENE_DIR / "images"
WORKSPACE_DIR = SCENE_DIR / "workspace"

NUM_IMAGES = 8
IMG_W, IMG_H = 640, 480
JPEG_QUALITY = 85

# ---------------------------------------------------------------------------
# Evidence marker world positions (room-coordinate fractions 0..1)
#   EV-1  near east wall (right side)
#   EV-2  centre-floor
#   EV-3  on north wall  (top area)
# ---------------------------------------------------------------------------

EVIDENCE_MARKERS = [
    {"id": "EV-1", "wx": 0.72, "wy": 0.55, "color": (255, 191, 0)},   # amber
    {"id": "EV-2", "wx": 0.45, "wy": 0.68, "color": (255, 191, 0)},   # amber
    {"id": "EV-3", "wx": 0.30, "wy": 0.30, "color": (255, 191, 0)},   # amber
]

# Camera pans: (dx, dy) in pixel fractions — small translations per image
CAMERA_OFFSETS = [
    (0.00,  0.00),
    (0.03,  0.01),
    (-0.02, 0.02),
    (0.05,  0.03),
    (-0.04, 0.01),
    (0.02, -0.02),
    (-0.01,  0.04),
    (0.04, -0.03),
]

# Mild rotation angles in degrees per image
CAMERA_ROTATIONS = [0.0, 1.5, -1.0, 2.5, -2.0, 1.0, -3.0, 2.0]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _apply_noise(arr: np.ndarray, rng: np.random.Generator, sigma: float = 8.0) -> np.ndarray:
    """Add Gaussian noise to an RGB uint8 array."""
    noise = rng.normal(0.0, sigma, arr.shape)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_vignette(arr: np.ndarray) -> np.ndarray:
    """Darken the edges of an image to simulate a camera vignette."""
    h, w = arr.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    Y, X = np.ogrid[:h, :w]
    # Normalise distance to [0, 1]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # Smooth falloff: full bright at centre, darkened at corners
    mask = np.clip(1.0 - 0.55 * dist ** 1.6, 0.0, 1.0)
    return (arr * mask[:, :, np.newaxis]).astype(np.uint8)


def _world_to_pixel(wx: float, wy: float, dx: float = 0.0, dy: float = 0.0) -> tuple[int, int]:
    """Map world fractions → pixel coords, with camera pan offset."""
    # Room spans roughly the centre 80% of the image
    margin_x = 0.10
    margin_y = 0.15
    px = int((margin_x + wx * (1.0 - 2 * margin_x) + dx) * IMG_W)
    py = int((margin_y + wy * (1.0 - 2 * margin_y) + dy) * IMG_H)
    return (px, py)


def _draw_room(draw: ImageDraw.ImageDraw, floor_color: tuple, wall_color: tuple) -> None:
    """Draw the basic room geometry: floor + walls."""
    # Floor — fills the middle area
    floor_poly = [
        (50, 160),
        (590, 160),
        (590, 440),
        (50, 440),
    ]
    draw.polygon(floor_poly, fill=floor_color)

    # Back wall (top strip)
    back_wall = [
        (50, 80),
        (590, 80),
        (590, 160),
        (50, 160),
    ]
    draw.polygon(back_wall, fill=wall_color)

    # Left wall (trapezoid)
    left_wall = [
        (0, 0),
        (50, 80),
        (50, 440),
        (0, IMG_H),
    ]
    draw.polygon(left_wall, fill=(45, 45, 50))

    # Right wall (trapezoid)
    right_wall = [
        (IMG_W, 0),
        (590, 80),
        (590, 440),
        (IMG_W, IMG_H),
    ]
    draw.polygon(right_wall, fill=(40, 40, 45))

    # Ceiling (above back wall)
    ceiling = [
        (0, 0),
        (IMG_W, 0),
        (590, 80),
        (50, 80),
    ]
    draw.polygon(ceiling, fill=(35, 35, 40))

    # Baseboard line
    draw.line([(50, 440), (590, 440)], fill=(60, 60, 65), width=2)


def _draw_furniture(draw: ImageDraw.ImageDraw) -> None:
    """Draw a few dark-grey furniture silhouettes."""
    # Table (centre-left)
    draw.rectangle([80, 290, 220, 390], fill=(38, 38, 43), outline=(55, 55, 60), width=1)
    # Table legs
    for lx in [90, 210]:
        draw.rectangle([lx, 390, lx + 8, 430], fill=(32, 32, 36))

    # Shelving unit on east wall
    draw.rectangle([490, 175, 575, 375], fill=(40, 40, 45), outline=(60, 60, 65), width=1)
    for shelf_y in [220, 265, 310, 355]:
        draw.rectangle([495, shelf_y, 570, shelf_y + 4], fill=(55, 55, 60))

    # Chair (near table)
    draw.rectangle([145, 385, 195, 435], fill=(36, 36, 40), outline=(52, 52, 56), width=1)
    draw.rectangle([145, 360, 200, 395], fill=(42, 42, 47))

    # Small debris / item on floor (bottom-right)
    draw.ellipse([390, 400, 420, 420], fill=(34, 34, 38))


def _draw_evidence_marker(
    draw: ImageDraw.ImageDraw,
    px: int,
    py: int,
    label: str,
    color: tuple,
) -> None:
    """Draw a bright evidence marker circle with label."""
    r = 14
    # Outer glow (slightly larger, semi-transparent amber)
    draw.ellipse([px - r - 4, py - r - 4, px + r + 4, py + r + 4],
                 fill=(color[0], color[1], color[2], 80) if False else (
                     max(0, color[0] - 60),
                     max(0, color[1] - 60),
                     max(0, color[2] - 120),
                 ))
    # Main circle
    draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(255, 255, 255), width=2)
    # Inner cross-hair
    draw.line([(px - r + 3, py), (px + r - 3, py)], fill=(20, 20, 20), width=1)
    draw.line([(px, py - r + 3), (px, py + r - 3)], fill=(20, 20, 20), width=1)
    # Label text (using default font; no custom TTF required)
    draw.text((px + r + 3, py - 8), label, fill=(255, 220, 60))


def _draw_floor_tape(draw: ImageDraw.ImageDraw) -> None:
    """Draw yellow crime-scene tape effect along the floor boundary."""
    stripe_w = 12
    stripe_gap = 20
    y_tape = 440
    for x in range(0, IMG_W, stripe_w + stripe_gap):
        draw.rectangle([x, y_tape - 5, x + stripe_w, y_tape + 5], fill=(230, 190, 0))


def generate_scene_image(
    frame_index: int,
    rng: np.random.Generator,
) -> Image.Image:
    """Generate one synthetic forensic scene image."""
    dx_frac, dy_frac = CAMERA_OFFSETS[frame_index]
    angle = CAMERA_ROTATIONS[frame_index]

    # Vary floor / wall colours slightly per frame to mimic exposure changes
    brightness = int(rng.integers(-8, 8))
    floor_color = tuple(max(0, min(255, c + brightness)) for c in (52, 50, 48))
    wall_color = tuple(max(0, min(255, c + brightness // 2)) for c in (65, 63, 60))

    # Background (very dark — night / low-light crime scene)
    img_arr = np.full((IMG_H, IMG_W, 3), 18, dtype=np.uint8)
    img = Image.fromarray(img_arr, "RGB")
    draw = ImageDraw.Draw(img)

    _draw_room(draw, floor_color, wall_color)  # type: ignore[arg-type]
    _draw_furniture(draw)
    _draw_floor_tape(draw)

    # Draw evidence markers with pan offset
    for marker in EVIDENCE_MARKERS:
        px, py = _world_to_pixel(
            marker["wx"] + dx_frac,
            marker["wy"] + dy_frac,
        )
        # Clamp to image bounds
        px = max(20, min(IMG_W - 20, px))
        py = max(20, min(IMG_H - 20, py))
        _draw_evidence_marker(draw, px, py, marker["id"], marker["color"])  # type: ignore[arg-type]

    # Slight rotation to simulate hand-held camera
    if abs(angle) > 0.01:
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(18, 18, 18))

    # Add photographic grain
    arr = np.array(img)
    arr = _apply_noise(arr, rng, sigma=6.0)

    # Slight blur to simulate lens imperfection
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.4))

    # Vignette
    arr = np.array(img)
    arr = _apply_vignette(arr)

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Static file content
# ---------------------------------------------------------------------------

README_MD = """\
# Sample Scene: Crime Scene 01

A synthetic indoor crime scene dataset for testing ForenSim's reconstruction and inference pipeline.

## Contents
- 8 synthetic photographs (640×480, JPEG)
- Simulated indoor room with 3 evidence markers (EV-1, EV-2, EV-3)
- Camera positions approximate a real photogrammetry session

## Evidence Tags
| Marker | Tag | Description |
|---|---|---|
| EV-1 | blood_spatter | Simulated blood spatter pattern near east wall |
| EV-2 | shell_casing | Expended brass shell casing on floor |
| EV-3 | impact_mark | Projectile impact mark on north wall |

## Usage
Load the `images/` directory in ForenSim's Evidence tab,
set the workspace to `workspace/`, then run reconstruction.
"""

ANNOTATIONS_JSON = {
    "annotations": [
        {
            "id": "ann-ev1",
            "image_path": "img_001.jpg",
            "shape": "rect",
            "coordinates": [[120, 180], [200, 260]],
            "tag": "blood_spatter",
            "description": "Simulated blood spatter pattern, EV-1",
            "confidence": 0.9,
            "metadata": {"marker_id": "EV-1"},
        },
        {
            "id": "ann-ev2",
            "image_path": "img_003.jpg",
            "shape": "rect",
            "coordinates": [[310, 300], [360, 350]],
            "tag": "shell_casing",
            "description": "Brass shell casing on floor, EV-2",
            "confidence": 0.95,
            "metadata": {"marker_id": "EV-2"},
        },
        {
            "id": "ann-ev3",
            "image_path": "img_005.jpg",
            "shape": "polygon",
            "coordinates": [[400, 120], [450, 130], [445, 165], [395, 155]],
            "tag": "impact_mark",
            "description": "Projectile impact mark on wall, EV-3",
            "confidence": 0.85,
            "metadata": {"marker_id": "EV-3"},
        },
    ]
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("ForenSim — Sample Dataset Generator")
    print("=" * 44)

    # Create directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directories created under {SCENE_DIR}")

    # Generate images
    rng = np.random.default_rng(seed=42)
    for i in range(NUM_IMAGES):
        img = generate_scene_image(i, rng)
        out_path = IMAGES_DIR / f"img_{i + 1:03d}.jpg"
        img.save(str(out_path), "JPEG", quality=JPEG_QUALITY)
        print(f"[OK] Saved {out_path.name}  ({img.width}x{img.height})")

    # Write README
    readme_path = SCENE_DIR / "README.md"
    readme_path.write_text(README_MD, encoding="utf-8")
    print(f"[OK] {readme_path.name} written")

    # Write annotations.json
    ann_path = SCENE_DIR / "annotations.json"
    ann_path.write_text(json.dumps(ANNOTATIONS_JSON, indent=2), encoding="utf-8")
    print(f"[OK] annotations.json written")

    print()
    print(f"Dataset ready at: {SCENE_DIR}")
    print(f"  {NUM_IMAGES} images in {IMAGES_DIR.relative_to(REPO_ROOT)}")
    print(f"  Workspace:       {WORKSPACE_DIR.relative_to(REPO_ROOT)}")
    print()
    print("Run 'python scripts/run_demo.py' to execute the full pipeline demo.")


if __name__ == "__main__":
    main()

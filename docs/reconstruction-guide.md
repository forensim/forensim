# Reconstruction Guide

This guide walks through everything needed to go from a folder of crime-scene photographs to a physics-ready USD scene inside ForenSim.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Supported Reconstruction Methods](#2-supported-reconstruction-methods)
3. [Step-by-Step: Preparing Images](#3-step-by-step-preparing-images)
4. [Step-by-Step: Running Reconstruction in ForenSim](#4-step-by-step-running-reconstruction-in-forensim)
5. [Understanding the Output](#5-understanding-the-output)
6. [Troubleshooting](#6-troubleshooting)
7. [Advanced: Real gsplat Trainer vs. Fallback Exporter](#7-advanced-real-gsplat-trainer-vs-fallback-exporter)
8. [Sample Dataset Walkthrough](#8-sample-dataset-walkthrough)

---

## 1. Prerequisites

### Required for all reconstruction modes

| Dependency | Version | Install |
|---|---|---|
| Python | 3.12 | `uv python install 3.12` |
| PyTorch (CUDA build) | 2.7.1+cu126 | See below |
| COLMAP | 4.x | [github.com/colmap/colmap/releases](https://github.com/colmap/colmap/releases) |
| forensim Python package | current | `uv pip install -e ".[dev]"` |

### Required for Gaussian Splatting (real trainer)

| Dependency | Version | Notes |
|---|---|---|
| CUDA Toolkit | 12.6 | Provides `nvcc` for JIT compilation |
| VS 2022 Build Tools | 14.4x | MSVC `cl.exe`; required on Windows |
| gsplat | latest | Installed by `uv pip install -e ".[dev]"` |

### Required for NuRec gRPC (optional, high-quality)

| Dependency | Notes |
|---|---|
| Docker Desktop | To run the NuRec NGC container |
| NVIDIA NGC account | Free; needed to pull `nvcr.io/nvidia/nurec` |
| grpcio | Installed by `uv pip install -e ".[dev]"` |

### Installing PyTorch with CUDA support

```powershell
uv pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 `
    --index-url https://download.pytorch.org/whl/cu126
```

> **Note:** Do not install PyTorch from the default PyPI index — it ships a CPU-only build that cannot run gsplat.

### Installing COLMAP (Windows)

1. Download the latest CUDA-enabled binary from the [COLMAP releases page](https://github.com/colmap/colmap/releases).
2. Extract to a persistent location, e.g. `D:\Tools\COLMAP`.
3. Add the `bin` folder to your `PATH`, or set the environment variable:

```powershell
$env:COLMAP_PATH = "D:\Tools\COLMAP\bin\colmap.exe"
```

ForenSim searches `COLMAP_PATH` first, then falls back to the system `PATH`.

---

## 2. Supported Reconstruction Methods

### Method A: COLMAP + Gaussian Splatting (default)

This is the recommended path for most crime-scene photos.

```
Images → COLMAP (SfM) → sparse point cloud + camera poses
                                │
                                ▼
                        gsplat trainer → .ply (3D Gaussian Splat)
                                │
                                ▼
                omniverse-gsplat-converter → .usda scene
```

**Pros:** Runs entirely locally, no external service, well-tested.  
**Cons:** Real trainer requires CUDA Toolkit; fallback exporter produces a lower-quality scene.

### Method B: NuRec gRPC (optional)

NuRec is NVIDIA's neural reconstruction container. It produces higher-fidelity scenes with better relighting because of its Harmonizer network.

```
Images → NuRec Docker container (gRPC) → high-fidelity scene mesh/frames
                                │
                                ▼
                        omniverse-gsplat-converter → .usda scene
```

**Pros:** Best reconstruction quality, no CUDA Toolkit needed on the host.  
**Cons:** Requires a running Docker container and an NVIDIA NGC account.

To use NuRec, pull the container once:

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/nurec:latest
docker run -d -p 8080:8080 --gpus all nvcr.io/nvidia/nurec:latest
```

Then set:

```powershell
$env:NUREC_ADDRESS = "localhost:8080"
```

---

## 3. Step-by-Step: Preparing Images

Good input images are the single biggest factor in reconstruction quality.

### Image capture guidelines

- **Coverage:** Photograph the scene from all angles. Aim for at least 60–80% overlap between adjacent images.
- **Count:** 30–150 images covers a typical room. More is not always better — redundant nearly-identical shots slow COLMAP without adding information.
- **Lighting:** Consistent, diffuse lighting. Avoid hard shadows and strong reflections. If windows create bright spots, take photos both with and without window light.
- **Movement:** Nothing in the scene should move between shots. Tape off the area if working a live scene.
- **Resolution:** At least 3 MP per image (12 MP+ recommended). Higher resolution means more accurate COLMAP feature matches.
- **Format:** JPEG or PNG. RAW files should be converted before import.

### Organizing images for ForenSim

Create one directory per scene. Keep only scene photographs in it — do not mix orientation/calibration shots unless you add them as a separate camera group.

```
assets/
└── sample-scenes/
    └── crime-scene-01/
        ├── images/           ← all input photos go here
        │   ├── DSC_0001.jpg
        │   ├── DSC_0002.jpg
        │   └── ...
        └── metadata.json     ← optional: timestamps, GPS, camera model
```

The `metadata.json` schema:

```json
{
  "scene_id": "crime-scene-01",
  "capture_date": "2026-03-15T14:22:00Z",
  "camera_model": "Canon EOS R5",
  "focal_length_mm": 24,
  "images": [
    { "filename": "DSC_0001.jpg", "timestamp": "2026-03-15T14:22:05Z" }
  ]
}
```

Timestamps and GPS coordinates (if available in EXIF) are extracted automatically by ForenSim and attached to the reconstruction manifest.

---

## 4. Step-by-Step: Running Reconstruction in ForenSim

### Via the Desktop App

1. Open ForenSim. The Evidence Panel is on the left.
2. Click **New Scene** and enter a scene name.
3. Click **Import Images** and select the directory containing your photographs.
4. (Optional) Switch the reconstruction method in the dropdown from **Gaussian Splatting** to **NuRec** if the container is running.
5. Click **Reconstruct**. A progress bar streams COLMAP and gsplat output.
6. When complete, the 3D scene appears in the central viewport.

### Via the Python API directly

```python
from forensim.reconstruct import ReconstructionPipeline

pipeline = ReconstructionPipeline(
    image_dir="assets/sample-scenes/crime-scene-01/images",
    output_dir="output/crime-scene-01",
    method="gaussian_splatting",   # or "nurec"
    gsplat_iterations=30_000,
    colmap_matcher="exhaustive",   # or "sequential" for video frames
)

def on_progress(step: str, percent: float) -> None:
    print(f"[{percent:5.1f}%] {step}")

result = pipeline.run(on_progress=on_progress)
print(f"USD scene: {result.usd_path}")
print(f"Manifest:  {result.manifest_path}")
```

### Via the FastAPI sidecar (HTTP)

```bash
curl -X POST http://localhost:8008/api/reconstruct \
  -H "Content-Type: application/json" \
  -d '{
    "image_dir": "assets/sample-scenes/crime-scene-01/images",
    "output_dir": "output/crime-scene-01",
    "method": "gaussian_splatting"
  }'
```

The response is a `text/event-stream` of progress events:

```
data: {"step": "colmap_feature_extraction", "percent": 12.0}
data: {"step": "colmap_matching", "percent": 28.5}
data: {"step": "gsplat_training", "percent": 61.2}
data: {"step": "usd_conversion", "percent": 94.0}
data: {"step": "complete", "percent": 100.0, "manifest": "output/crime-scene-01/manifest.json"}
```

---

## 5. Understanding the Output

A completed reconstruction produces three artifacts:

### PLY point cloud — `output/<scene>/splat.ply`

The raw output of the Gaussian Splatting trainer. Each Gaussian is stored as a point with position (x, y, z), rotation quaternion, scale (sx, sy, sz), opacity, and spherical harmonic coefficients for view-dependent color. This file can be opened in any 3D Gaussian Splat viewer (e.g., [SuperSplat](https://supersplat.dev/)).

### USD scene — `output/<scene>/scene.usda`

The physics-ready scene in OpenUSD format, produced by `omniverse-gsplat-converter`. The USD file references the `.ply` file and wraps it in a `ParticleField3DGaussianSplat` prim. This file opens directly in NVIDIA Omniverse Composer.

After scene enrichment (Step 3 of the pipeline), the USD additionally contains:

- `PhysicsScene` prim with gravity and solver settings
- Per-object `PhysicsRigidBodyAPI` schemas with mass and friction properties
- `PhysicsMaterialAPI` bindings for each surface material
- Custom `forensim:annotation` attributes encoding ROI evidence

### Reconstruction manifest — `output/<scene>/manifest.json`

A machine-readable record of the full reconstruction run:

```json
{
  "scene_id": "crime-scene-01",
  "reconstruction_date": "2026-07-04T10:15:00Z",
  "method": "gaussian_splatting",
  "colmap": {
    "version": "4.1",
    "num_images": 72,
    "num_registered": 70,
    "num_points3d": 84321,
    "mean_reprojection_error": 0.42
  },
  "gsplat": {
    "iterations": 30000,
    "num_gaussians": 412089,
    "final_psnr": 31.7,
    "training_time_s": 847
  },
  "outputs": {
    "ply": "splat.ply",
    "usd": "scene.usda",
    "manifest": "manifest.json"
  }
}
```

The manifest is consumed by the inference pipeline to attach reconstruction metadata to the Bayesian analysis report.

---

## 6. Troubleshooting

### COLMAP not found

**Symptom:** `RuntimeError: COLMAP binary not found. Set COLMAP_PATH or add colmap to PATH.`

**Fix:** Set the environment variable pointing to the COLMAP executable:

```powershell
$env:COLMAP_PATH = "D:\Tools\COLMAP\bin\colmap.exe"
```

Or add `D:\Tools\COLMAP\bin` to your system `PATH` permanently.

---

### CUDA out of memory during gsplat training

**Symptom:** `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Fixes (try in order):**

1. Reduce the number of training iterations:
   ```python
   pipeline = ReconstructionPipeline(..., gsplat_iterations=10_000)
   ```

2. Reduce the initial number of Gaussians by increasing the densification interval:
   ```python
   pipeline = ReconstructionPipeline(..., densify_interval=500)
   ```

3. Switch to the fallback exporter (no training, lower quality but no VRAM pressure):
   ```powershell
   $env:FORENSIM_GSPLAT_FALLBACK = "1"
   ```

4. If on a 12 GB GPU, close other GPU-resident applications (games, other ML jobs) before running.

---

### gsplat JIT compilation fails (Windows)

**Symptom:** `RuntimeError: nvcc not found` or `error: unrecognized command line option '-Wno-attributes'`

**Diagnosis:** gsplat JIT-compiles its CUDA kernels on first import using `torch.utils.cpp_extension`. On Windows this requires:
- `nvcc` from the CUDA Toolkit
- `cl.exe` from **VS 2022 Build Tools** specifically (CUDA 12.6 does not support VS 2026+)

**Fix for `-Wno-attributes` error (MSVC):** Apply a one-time patch to the gsplat backend:

```python
# File: .venv/Lib/site-packages/gsplat/cuda/_backend.py  (line ~177)
# Change:
extra_cflags = [opt_level, "-Wno-attributes"]
# To:
import platform
if platform.system() == "Windows":
    extra_cflags = [opt_level]
else:
    extra_cflags = [opt_level, "-Wno-attributes"]
```

**Fix for `nvcc not found`:** Set the CUDA environment variables and ensure VS 2022 `cl.exe` is first in PATH:

```powershell
$msvc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$env:Path = "$msvc;$env:Path"
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CUDA_PATH = $env:CUDA_HOME
$env:TORCH_CUDA_ARCH_LIST = "8.6"   # adjust to your GPU's SM version
```

Compilation takes approximately 5 minutes on first run. The compiled `.pyd` is cached at:
`%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py312_cu126\gsplat_cuda\`

Subsequent runs load the cache instantly.

---

### NuRec gRPC connection refused

**Symptom:** `grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE`

**Fix:** Ensure the NuRec Docker container is running and listening on the configured port:

```bash
docker ps | grep nurec
# If not running:
docker run -d -p 8080:8080 --gpus all nvcr.io/nvidia/nurec:latest
```

Check the address ForenSim is using:

```powershell
$env:NUREC_ADDRESS = "localhost:8080"   # default
```

---

### COLMAP registers fewer images than expected

**Symptom:** Manifest shows `num_registered` significantly lower than `num_images`.

**Causes and fixes:**

- **Insufficient overlap:** Reshoot with more coverage; aim for 70%+ overlap between adjacent images.
- **Motion blur / out-of-focus shots:** Remove blurry images from the input directory.
- **Difficult surfaces:** Reflective floors, white walls, and featureless ceilings have few COLMAP feature matches. Try switching to the `exhaustive` matcher:
  ```python
  pipeline = ReconstructionPipeline(..., colmap_matcher="exhaustive")
  ```
- **Very large image count (>200):** Use `sequential` matcher if images were captured as a video walk-through; it assumes temporal ordering.

---

## 7. Advanced: Real gsplat Trainer vs. Fallback Exporter

ForenSim ships two gsplat paths, controlled by `FORENSIM_GSPLAT_FALLBACK`.

### Fallback exporter (`FORENSIM_GSPLAT_FALLBACK=1`, default)

Does **not** run the gsplat optimization loop. Instead, it initializes Gaussians directly from the COLMAP sparse point cloud and exports them to `.ply` without training. The result is a coarse splat that faithfully represents the point cloud geometry but lacks the smooth appearance of a trained model.

**Use when:**
- Running without the CUDA Toolkit installed
- CI / automated testing (fast, no GPU required beyond a CUDA-capable torch build)
- Rapid iteration on the pipeline without waiting for training

### Real trainer (`FORENSIM_GSPLAT_FALLBACK=0`)

Runs the full gsplat training loop: initialize from COLMAP sparse cloud, densify Gaussians, optimize appearance via photometric loss for N iterations. Produces a photo-realistic splat model.

**Requirements:**
- CUDA Toolkit 12.x (`nvcc`)
- VS 2022 Build Tools on Windows
- VRAM: 10 GB+ recommended for scenes with 30+ images

**Configuration:**

| Parameter | Default | Description |
|---|---|---|
| `gsplat_iterations` | 30 000 | Training iterations; more = better quality, slower |
| `densify_until_iter` | 15 000 | Stop adding new Gaussians after this iteration |
| `densify_interval` | 100 | Add/prune Gaussians every N iterations |
| `opacity_reset_interval` | 3 000 | Periodically reset low-opacity Gaussians |

```python
pipeline = ReconstructionPipeline(
    image_dir="assets/sample-scenes/crime-scene-01/images",
    output_dir="output/crime-scene-01",
    method="gaussian_splatting",
    gsplat_fallback=False,          # use real trainer
    gsplat_iterations=50_000,       # high quality
    densify_interval=200,
)
```

---

## 8. Sample Dataset Walkthrough

ForenSim ships a sample dataset at `assets/sample-scenes/crime-scene-01/`. This is a small public-domain set of 24 indoor scene photographs suitable for a quick end-to-end test.

### 1. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the demo script

```powershell
python scripts/run_demo.py
```

This calls `ReconstructionPipeline` with the sample images and the fallback exporter (so no CUDA Toolkit is needed), then runs a short Monte Carlo inference on synthetic trajectory data. Expected runtime: 2–5 minutes.

### 3. Inspect the output

```
output/crime-scene-01/
├── splat.ply            # point cloud (fallback — not photo-realistic)
├── scene.usda           # USD scene ready for PhysX enrichment
└── manifest.json        # reconstruction statistics
```

Open `scene.usda` in NVIDIA Omniverse Composer, or load it directly in ForenSim via **File → Open Scene**.

### 4. Run with the real trainer (optional)

If you have the CUDA Toolkit and VS 2022 Build Tools installed:

```powershell
$env:FORENSIM_GSPLAT_FALLBACK = "0"
python scripts/run_demo.py --iterations 30000
```

The scene will take 10–15 minutes to train but will be substantially more photo-realistic.

### 5. Verify reconstruction quality

The manifest reports `final_psnr` (peak signal-to-noise ratio). Typical values:

| PSNR | Quality |
|---|---|
| < 25 dB | Poor — likely sparse coverage or blurry images |
| 25–29 dB | Acceptable — usable for physics enrichment |
| 29–33 dB | Good — representative of most real-world scenes |
| > 33 dB | Excellent |

A PSNR below 25 dB suggests the image set needs more coverage or the COLMAP registration failed for too many images. Check `manifest.json → colmap.num_registered` first.

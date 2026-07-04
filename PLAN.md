# ForenSim — Forensic Scene Reconstruction & Probabilistic Event Analysis Platform
## Project Plan & Architecture

> **Organization:** https://github.com/forensim  
> **Local workspace:** `D:\forensim\`  
> **Status:** Pre-development / Planning  
> **Last updated:** 2026-07-04

---

## Table of Contents

1. [Vision & Goals](#1-vision--goals)
2. [Technology Stack](#2-technology-stack)
3. [System Architecture](#3-system-architecture)
4. [Module Breakdown](#4-module-breakdown)
5. [Data Pipeline: Images → 3D → Omniverse](#5-data-pipeline-images--3d--omniverse)
6. [Probabilistic Event Reconstruction Engine](#6-probabilistic-event-reconstruction-engine)
7. [Physics Simulation with NVIDIA PhysX](#7-physics-simulation-with-nvidia-physx)
8. [Rust/Python Bindings via PyO3 + Maturin](#8-rustpython-bindings-via-pyo3--maturin)
9. [Desktop UI — Tauri + React + TypeScript](#9-desktop-ui--tauri--react--typescript)
10. [Repository Structure](#10-repository-structure)
11. [Development Roadmap](#11-development-roadmap)
12. [Key Dependencies & Versions](#12-key-dependencies--versions)
13. [GitHub Organization Setup](#13-github-organization-setup)

---

## 1. Vision & Goals

**ForenSim** is an open-source forensic scene reconstruction platform that takes 2D evidence (images, video frames, photographs) and produces an interactive, probabilistically-annotated 3D simulation of the most likely sequence of events that occurred at a scene.

### Core Goals

- **Reconstruct** real-world scenes from 2D photographs and video using photogrammetry and neural rendering (Gaussian Splatting / NeRF)
- **Import** reconstructed geometry into NVIDIA Omniverse as USD scenes
- **Simulate** physical events within those scenes using NVIDIA PhysX
- **Infer** the probabilistic likelihood of event sequences from physical evidence using Bayesian / HMM / Markov chain models
- **Visualize** everything through a clean, modern, dark-themed desktop application

### Target Use Cases

- Crime scene reconstruction from photographs
- Accident reconstruction from dashcam / CCTV footage
- Physical evidence analysis (blood spatter, trajectory analysis, object dynamics)
- Courtroom-ready 3D presentations of reconstructed events

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **3D Reconstruction** | COLMAP + gsplat / instant-ngp | SfM + Gaussian Splatting / NeRF from images |
| **Neural Rendering** | NVIDIA NuRec (gRPC API) | Advanced scene reconstruction |
| **3D Scene Format** | OpenUSD (`pxr`) | Universal scene description |
| **Simulation Platform** | NVIDIA Omniverse / Isaac Sim 6.x | Physics-enabled 3D scene |
| **Physics Engine** | NVIDIA PhysX via `ovphysx` | Rigid body / soft body / fluid simulation |
| **Backend Language** | Rust (stable, MSRV 1.83) | Performance, safety, core logic |
| **Script/AI Language** | Python 3.12 | Omniverse APIs, ML, photogrammetry |
| **Rust↔Python Bridge** | PyO3 0.29 + Maturin 1.14 | FFI bindings between Rust and Python |
| **Desktop Framework** | Tauri v2 (Rust) | Cross-platform desktop shell |
| **Frontend** | React 19 + TypeScript + Vite | UI components and 3D viewport |
| **UI Components** | shadcn/ui + Tailwind CSS | Dark, modern forensic aesthetic |
| **3D Viewport (Web)** | Three.js + GaussianSplats3D | In-app 3D scene preview |
| **Data Visualization** | VisX + Recharts | Probability distributions, timelines |
| **Probabilistic Engine** | PyMC / Stan + custom Rust crates | Bayesian inference, HMM, sequence likelihood |
| **Rust Stats** | nalgebra, ndarray, statrs, bayes-rs | Numerical / statistical computing |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ForenSim Desktop App                         │
│                        (Tauri v2 Shell)                             │
├──────────────────────────┬──────────────────────────────────────────┤
│   React + TypeScript UI  │         Rust Core (Tauri Backend)        │
│  ┌────────────────────┐  │  ┌─────────────────────────────────────┐ │
│  │  Scene Viewer      │  │  │  forensim-core (Rust crate)         │ │
│  │  (Three.js / 3DGS) │◄─┼──┤  - File I/O, USD manipulation      │ │
│  ├────────────────────┤  │  │  - Physics state management         │ │
│  │  Evidence Panel    │  │  │  - Probabilistic sequence engine    │ │
│  │  (image upload,    │  │  │  - nalgebra / statrs / bayes-rs     │ │
│  │   metadata tags)   │  │  └──────────────┬──────────────────────┘ │
│  ├────────────────────┤  │                 │ PyO3 (GIL-aware)        │
│  │  Timeline View     │  │  ┌──────────────▼──────────────────────┐ │
│  │  (D3 event seq.)   │  │  │  Python Bridge Layer                │ │
│  ├────────────────────┤  │  │  (forensim-py, Maturin wheel)       │ │
│  │  Probability Panel │  │  │  - COLMAP, gsplat, nerfstudio       │ │
│  │  (VisX dist. plots)│  │  │  - Omniverse / Isaac Sim APIs       │ │
│  ├────────────────────┤  │  │  - ovphysx (PhysX simulation)       │ │
│  │  Scenario Engine   │  │  │  - PyMC (Bayesian inference)        │ │
│  │  (hypothesis list) │  │  │  - pxr (OpenUSD)                    │ │
│  └────────────────────┘  │  └──────────────┬──────────────────────┘ │
│            ▲             │                 │                         │
│            │ HTTP/IPC    │                 │ gRPC / USD / HTTP       │
└────────────┼─────────────┴─────────────────┼─────────────────────────┘
             │                               │
             │                    ┌──────────▼────────────┐
             │                    │  NVIDIA Omniverse      │
             │                    │  (Local / Nucleus)     │
             │                    │  - USD Scene           │
             │                    │  - PhysX Simulation    │
             │                    │  - RTX Rendering       │
             │                    └──────────┬────────────┘
             │                               │
             │                    ┌──────────▼────────────┐
             │                    │  NVIDIA NuRec          │
             │                    │  (Docker / NGC)        │
             │                    │  - gRPC API            │
             │                    │  - Gaussian Splatting  │
             └────────────────────┴────────────────────────┘
```

---

## 4. Module Breakdown

### 4.1 `forensim-core` (Rust crate — the heart of the system)

Responsible for:
- Scene graph representation (wrapping USD concepts in safe Rust types)
- Physics state ingestion from `ovphysx` outputs via Python bridge
- Probabilistic sequence modeling engine:
  - Markov chain transition matrices
  - Hidden Markov Model (Viterbi algorithm)
  - Bayesian evidence aggregation
  - Monte Carlo simulation for uncertainty
- Statistical utilities (nalgebra, ndarray, statrs, bayes-rs)
- IPC with Tauri frontend via `tauri::command`

### 4.2 `forensim-py` (Python package — Maturin wheel)

Responsible for:
- Wrapping `forensim-core` Rust functions for Python consumption
- Orchestrating the full reconstruction pipeline
- Calling Omniverse / Isaac Sim / pxr APIs
- Running `ovphysx` physics simulations
- Running PyMC Bayesian models
- Conversion: Gaussian Splat PLY → USD

### 4.3 `forensim-reconstruct` (Python sub-package)

Responsible for:
- Ingesting raw images or video frames
- Running COLMAP for Structure-from-Motion (SfM)
- Running `gsplat` for 3D Gaussian Splatting
- Optionally calling NVIDIA NuRec gRPC API for advanced reconstruction
- Exporting `.ply` → `.usdz` via `omniverse-gsplat-converter`
- Outputting a ready-to-simulate USD scene

### 4.4 `forensim-app` (Tauri v2 + React)

Responsible for:
- Desktop window, navigation, dark theme
- Evidence upload (images, video, metadata)
- 3D viewport (Three.js + GaussianSplats3D viewer)
- Probability distribution panels (VisX)
- Event timeline visualization (D3.js)
- Scenario hypothesis manager (ranked list of event sequences)
- Sending reconstruction / simulation jobs to Python backend
- Displaying results from Rust probabilistic engine

---

## 5. Data Pipeline: Images → 3D → Omniverse

```
Raw Evidence Input
(images / video / CCTV frames)
        │
        ▼
┌──────────────────────────────────┐
│  Step 1: Structure-from-Motion   │
│  Tool: COLMAP                    │
│  Output: sparse point cloud +    │
│          camera poses            │
└────────────────┬─────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌──────────────────────────┐
│ Traditional   │  │  Neural / Gaussian        │
│ Route:        │  │  Route:                   │
│ OpenMVS       │  │  gsplat (3D Gaussian      │
│ → Dense mesh  │  │  Splatting)               │
│ → PLY/OBJ     │  │  OR instant-ngp (NeRF)    │
│               │  │  → .ply splat file        │
└───────┬───────┘  └──────────┬───────────────┘
        │                     │
        ▼                     ▼
┌───────────────────────────────────┐
│  Step 2: USD Conversion           │
│  Tool: omniverse-gsplat-converter │
│  OR: Omniverse Asset Converter    │
│  Output: .usdz / .usda scene      │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  Step 3: Scene Enrichment         │
│  Tool: pxr Python API +           │
│        omni.kit / Isaac Sim       │
│  - Add PhysX properties           │
│  - Tag objects (rigid/soft/fluid) │
│  - Set material properties        │
│  - Define initial conditions      │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  Step 4: Physics Simulation       │
│  Tool: ovphysx (Python) or        │
│        omni.physx (in-Kit)        │
│  - Run N simulation scenarios     │
│  - Vary initial conditions        │
│  - Record state per timestep      │
│  Output: JSON trajectory data     │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  Step 5: Probabilistic Analysis   │
│  Tool: forensim-core (Rust) +     │
│        PyMC (Python)              │
│  - Compare trajectories vs        │
│    physical evidence              │
│  - Bayesian inference             │
│  - Rank hypotheses by P(H|E)      │
│  Output: ranked event sequence    │
│          probability distribution │
└───────────────────────────────────┘
```

---

## 6. Probabilistic Event Reconstruction Engine

### Mathematical Foundation

The core inference problem: given evidence E = {e₁, e₂, ..., eₙ} (physical observations), rank competing hypotheses H about what sequence of events occurred.

**Bayesian Update:**
```
P(H | E) ∝ P(E | H) × P(H)
```

**For event sequences**, the likelihood of an ordered sequence S = {s₁, s₂, ..., sₖ}:
```
P(S | H) = P(s₁|H) × ∏ P(sᵢ | sᵢ₋₁, H)    [Markov assumption]
```

**Integrating PhysX simulation outputs:**
- Run M parallel PhysX simulations with varied initial conditions
- Each simulation produces trajectory Tⱼ
- Compute likelihood: `P(E | Tⱼ)` = how well Tⱼ matches observed evidence
- Use Monte Carlo averaging: `P(E | H) ≈ (1/M) Σ P(E | Tⱼ)`

### Rust Implementation Targets (`forensim-core`)

- **Markov Chain:** Transition matrix T[i][j], state sequence scoring
- **HMM:** Forward-backward algorithm, Viterbi for best path
- **Monte Carlo:** Parallel simulation sampling via Rayon
- **Statistics:** nalgebra (matrix ops), statrs (distributions), ndarray-stats (summary stats)
- **Bayesian:** bayes-rs or fugue-ppl for posterior sampling

### Python Implementation (`forensim-py` / PyMC)

- Complex Bayesian networks with latent variables
- Gaussian Process surrogates for expensive PhysX calls
- NUTS/HMC sampler via PyMC
- Likelihood Ratio calculations for competing hypotheses
- Temporal Hawkes processes for crime linkage analysis

### Output

A **ranked hypothesis list** where each entry contains:
- Description of event sequence (e.g., "Object A fell from height H at time T, struck object B...")
- Posterior probability `P(H|E)` with credible intervals
- Visualization-ready trajectory data
- Sensitivity analysis (which evidence most changes the ranking)

---

## 7. Physics Simulation with NVIDIA PhysX

### Primary Interface: `ovphysx` (Python)

```python
from ovphysx import PhysX
import numpy as np

# Load reconstructed USD scene
physx = PhysX()
physx.add_usd("scene/reconstructed.usda")

# Set initial conditions derived from evidence
physx.set_initial_state("object_A", velocity=np.array([vx, vy, vz]))

# Run N scenario variations (Monte Carlo over initial conditions)
results = []
for scenario in scenario_variations:
    physx_clone = physx.clone()
    physx_clone.set_initial_state("object_A", velocity=scenario.velocity)
    trajectory = []
    for _ in range(steps):
        physx_clone.step(1.0 / 60.0, 0.0)
        trajectory.append(physx_clone.get_transforms())
    results.append(trajectory)
    physx_clone.release()

physx.release()
```

### Key Simulation Types for Forensics

| Simulation Type | PhysX Feature | Forensic Use Case |
|---|---|---|
| Rigid body dynamics | `RigidBody`, articulations | Object impact, vehicle collisions |
| Soft body simulation | FEM soft bodies | Human body, deformable materials |
| Fluid dynamics | SPH particles | Blood spatter, liquid spills |
| Cloth simulation | Cloth solver | Fabric evidence |
| Particle systems | GPU particles | Debris, dust, fragments |
| Contact reporting | `ContactReport` | Impact detection, force analysis |

### PhysX → Probabilistic Bridge

State extracted per simulation step:
```python
{
    "timestamp": float,
    "transforms": [...],   # position + orientation of each object
    "velocities": [...],   # linear + angular velocity
    "contacts": [...],     # contact pairs, force magnitude, contact point
    "energy": float        # total kinetic energy (energy dissipation = damage)
}
```

This data feeds directly into the `forensim-core` Rust probabilistic engine via PyO3 bindings.

---

## 8. Rust/Python Bindings via PyO3 + Maturin

### Strategy

- **Rust is the brain:** probabilistic math, sequence scoring, data structures
- **Python is the hands:** calls Omniverse, gsplat, ovphysx, PyMC
- **PyO3 bridges both** — Python calls into Rust (fast math), Rust calls into Python (ecosystem access)

### Project setup

```toml
# Cargo.toml
[package]
name = "forensim-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "forensim_core"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.29", features = ["extension-module"] }
nalgebra = "0.33"
ndarray = "0.17"
statrs = "0.18"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rayon = "1"
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.14,<2.0"]
build-backend = "maturin"

[project]
name = "forensim"
requires-python = ">=3.12"

[tool.maturin]
python-source = "python"
module-name = "forensim._core"
features = ["pyo3/extension-module"]
```

### Example: Viterbi HMM in Rust, called from Python

```rust
// src/hmm.rs
use pyo3::prelude::*;
use ndarray::Array2;

#[pyclass]
pub struct HiddenMarkovModel {
    transition: Array2<f64>,   // state × state
    emission: Array2<f64>,     // state × obs
    initial: Vec<f64>,
}

#[pymethods]
impl HiddenMarkovModel {
    #[new]
    fn new(transition: Vec<Vec<f64>>, emission: Vec<Vec<f64>>, initial: Vec<f64>) -> Self {
        // ...
    }

    fn viterbi(&self, observations: Vec<usize>) -> Vec<usize> {
        // Fast Viterbi implementation in Rust
        // Returns most likely hidden state sequence
    }

    fn forward_log_likelihood(&self, observations: Vec<usize>) -> f64 {
        // Log-likelihood of observation sequence
    }
}
```

```python
# Python usage
from forensim._core import HiddenMarkovModel

hmm = HiddenMarkovModel(
    transition=transition_matrix,
    emission=emission_matrix,
    initial=initial_probs
)

best_sequence = hmm.viterbi(observed_events)
log_prob = hmm.forward_log_likelihood(observed_events)
```

### Build workflow

```bash
# Development (installs into active venv)
maturin develop --release

# Production wheel
maturin build --release

# CI (GitHub Actions)
# Uses: PyO3/maturin-action
```

---

## 9. Desktop UI — Tauri + React + TypeScript

### Architecture

```
Tauri v2 App
├── WebView (React frontend)
│   ├── shadcn/ui + Tailwind (dark forensic theme)
│   ├── Three.js viewport (3D scene preview + Gaussian Splat viewer)
│   ├── VisX probability distribution panels
│   ├── D3.js forensic event timeline
│   └── Recharts for simulation result graphs
│
├── Rust Backend (Tauri commands)
│   ├── File system operations (evidence import/export)
│   ├── forensim-core bindings (probabilistic engine)
│   ├── Sidecar management (spawn Python backend)
│   └── Window management
│
└── Python Sidecar (FastAPI — bundled as executable)
    ├── /api/reconstruct  — run COLMAP + gsplat pipeline
    ├── /api/simulate     — run PhysX simulation scenarios
    ├── /api/infer        — run PyMC Bayesian inference
    └── /api/export       — export USD, reports
```

### UI Design Principles

- **Always dark mode** — deep blue-gray palette (`#0a0f1e` background)
- **Monospace accents** — forensic data feels technical
- **Color semantics:**
  - Blue (`#3b82f6`) — neutral information, timelines
  - Green (`#22c55e`) — high-probability scenarios
  - Amber (`#f59e0b`) — medium probability / warnings
  - Red (`#ef4444`) — low probability / critical evidence flags
- **Panels** — resizable split-pane layout (evidence | viewport | analysis)
- **3D Viewport** — dark background, wireframe overlays, probability heat maps
- **Timeline** — horizontal scrollable event sequence with probability bands

### Key React Components

```
src/
├── components/
│   ├── viewport/
│   │   ├── SceneViewer.tsx         # Three.js + 3DGS renderer
│   │   ├── CameraControls.tsx      # Orbit, pan, zoom
│   │   └── PhysicsOverlay.tsx      # Trajectory visualization
│   ├── evidence/
│   │   ├── EvidencePanel.tsx       # Image upload + tagging
│   │   ├── MetadataEditor.tsx      # Camera EXIF, timestamps
│   │   └── AnnotationLayer.tsx     # Draw regions of interest
│   ├── analysis/
│   │   ├── HypothesisList.tsx      # Ranked event sequences
│   │   ├── ProbabilityChart.tsx    # VisX distribution plots
│   │   ├── EventTimeline.tsx       # D3.js temporal view
│   │   └── LikelihoodRatioCard.tsx # P(H|E) display
│   ├── simulation/
│   │   ├── ScenarioBuilder.tsx     # Initial conditions editor
│   │   ├── SimulationControls.tsx  # Run / pause / step
│   │   └── ResultsTable.tsx        # Per-scenario outcomes
│   └── ui/                         # shadcn/ui primitives
```

### Tauri IPC Example

```typescript
// Frontend: trigger reconstruction job
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// Stream progress from Python backend
const unlisten = await listen<ReconstructionProgress>("reconstruction-progress", (event) => {
    setProgress(event.payload.percent);
    setCurrentStep(event.payload.step);
});

// Start reconstruction
const result = await invoke<ReconstructionResult>("start_reconstruction", {
    imageDir: selectedDir,
    method: "gaussian_splatting",
    outputPath: projectDir,
});
```

```rust
// Tauri command
#[tauri::command]
async fn start_reconstruction(
    app: AppHandle,
    image_dir: String,
    method: String,
    output_path: String,
) -> Result<ReconstructionResult, String> {
    // Spawn Python sidecar task
    // Stream progress events back to frontend
    todo!()
}
```

---

## 10. Repository Structure

```
forensim/                          # D:\forensim\ (local) / github.com/forensim/forensim
│
├── PLAN.md                        # This file
├── README.md                      # Project overview
├── LICENSE                        # MIT or Apache-2.0
├── Cargo.toml                     # Workspace root
├── Cargo.lock
├── pyproject.toml                 # Maturin config + Python package metadata
│
├── crates/
│   └── forensim-core/             # Core Rust library
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs             # PyO3 module root
│           ├── hmm.rs             # Hidden Markov Model
│           ├── bayes.rs           # Bayesian inference
│           ├── markov.rs          # Markov chain scoring
│           ├── monte_carlo.rs     # MC sampling
│           ├── usd.rs             # USD scene types (Rust-side)
│           └── stats.rs           # Utility wrappers (statrs, ndarray)
│
├── python/
│   └── forensim/                  # Python package (Maturin source)
│       ├── __init__.py
│       ├── _core.pyi              # Type stubs for Rust extension
│       ├── reconstruct/           # forensim-reconstruct sub-package
│       │   ├── __init__.py
│       │   ├── colmap.py          # COLMAP wrapper
│       │   ├── gsplat.py          # Gaussian Splatting pipeline
│       │   ├── nurec.py           # NuRec gRPC client
│       │   └── usd_export.py      # PLY/mesh → USD converter
│       ├── simulate/              # Physics simulation
│       │   ├── __init__.py
│       │   ├── physx_runner.py    # ovphysx wrapper
│       │   ├── scene_builder.py   # USD scene enrichment
│       │   └── scenario.py        # Scenario variation logic
│       ├── infer/                 # Probabilistic inference
│       │   ├── __init__.py
│       │   ├── bayesian.py        # PyMC models
│       │   ├── sequence.py        # Event sequence likelihoods
│       │   └── evidence.py        # Evidence scoring
│       └── api/                   # FastAPI sidecar
│           ├── __init__.py
│           ├── main.py            # FastAPI app entry point
│           ├── routes/
│           │   ├── reconstruct.py
│           │   ├── simulate.py
│           │   └── infer.py
│           └── build.spec         # PyInstaller spec
│
├── app/                           # Tauri + React desktop app
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── components/            # React components (see §9)
│   │   ├── hooks/
│   │   ├── lib/
│   │   │   ├── api.ts             # FastAPI client (auto-gen from OpenAPI)
│   │   │   └── tauri.ts           # Tauri IPC wrappers
│   │   └── styles/
│   │       └── globals.css        # Tailwind + dark theme vars
│   └── src-tauri/                 # Tauri Rust app
│       ├── Cargo.toml
│       ├── tauri.conf.json
│       ├── capabilities/
│       │   └── default.json
│       ├── binaries/              # Bundled Python sidecar EXE
│       └── src/
│           ├── lib.rs
│           ├── commands/
│           │   ├── mod.rs
│           │   ├── reconstruct.rs
│           │   ├── simulate.rs
│           │   └── project.rs
│           └── sidecar.rs         # Python sidecar lifecycle
│
├── assets/
│   ├── icons/
│   └── sample-scenes/            # Test evidence images
│
├── docs/
│   ├── architecture.md
│   ├── reconstruction-guide.md
│   └── probability-model.md
│
└── scripts/
    ├── setup-dev.ps1              # Windows dev environment setup
    ├── build-sidecar.ps1          # Build Python EXE
    └── build-app.ps1              # Full app build
```

---

## 11. Development Roadmap

### Phase 0 — Setup (Week 1)
- [x] Create GitHub organization: `forensim`
- [x] Create local workspace: `D:\forensim\`
- [ ] Initialize Cargo workspace (`Cargo.toml`)
- [ ] Initialize Tauri v2 app (`npm create tauri-app`)
- [ ] Initialize Python package with Maturin (`maturin new`)
- [ ] Set up CI skeleton (GitHub Actions)
- [ ] Write `AGENTS.md` with dev commands

### Phase 1 — Reconstruction Pipeline (Weeks 2-4)
- [x] Wrap COLMAP CLI in Python (`forensim.reconstruct.colmap`)
- [x] Integrate `gsplat` for Gaussian Splatting (fallback exporter + real trainer hook)
- [x] Implement PLY → USD conversion (`omniverse-gsplat-converter`)
- [x] Build end-to-end pipeline orchestrator with progress callbacks and manifest
- [x] Add unit + integration tests for the reconstruction pipeline
- [ ] Build evidence ingestion UI (file picker, image grid) — Phase 1.5
- [ ] Connect frontend → FastAPI → reconstruction pipeline — Phase 1.5
- [ ] Display reconstructed 3DGS in Three.js viewport — Phase 2

### Phase 2 — Physics Simulation (Weeks 5-7)
- [ ] Integrate `ovphysx` for standalone PhysX simulation
- [ ] Build `forensim.simulate.scene_builder` (USD scene enrichment)
- [ ] Build `forensim.simulate.physx_runner` (scenario Monte Carlo)
- [ ] Implement scenario builder UI
- [ ] Stream simulation progress to frontend
- [ ] Render simulation trajectories in 3D viewport

### Phase 3 — Probabilistic Engine (Weeks 8-10)
- [x] Implement Markov chain sequence scorer in Rust (`forensim-core::markov`)
- [x] Implement Viterbi HMM in Rust (`forensim-core::hmm`)
- [x] Implement Monte Carlo Bayesian engine in Rust
- [x] Build PyMC Bayesian model for evidence integration
- [x] Build hypothesis ranking and P(H|E) output
- [x] Connect to UI: probability panels, likelihood ratio cards

### Phase 4 — Advanced Features (Weeks 11-14)
- [x] NVIDIA NuRec gRPC client integration (health-check, list/load scenes, render frames)
- [ ] Soft body / fluid simulation for blood spatter analysis
- [x] Evidence annotation layer (ROI drawing on images)
- [x] Export: PDF report, USD scene, video flythrough
- [x] Sensitivity analysis (LOO re-ranking, per-evidence impact bars)
- [ ] Omniverse Nucleus integration (optional collaborative mode)

### Phase 5 — Polish & Portfolio (Weeks 15-16)
- [ ] Full dark theme polish
- [ ] Sample dataset: public domain crime scene photos
- [ ] Documentation site
- [ ] Demo video
- [ ] README with badges, screenshots, architecture diagram

---

## 12. Key Dependencies & Versions

### Rust
| Crate | Version | Purpose |
|---|---|---|
| pyo3 | 0.29.0 | Python ↔ Rust bindings |
| nalgebra | 0.33 | Linear algebra |
| ndarray | 0.17 | N-dimensional arrays |
| statrs | 0.18 | Probability distributions |
| rayon | 1.x | Parallel computation |
| serde / serde_json | 1.x | Serialization |
| tauri | 2.x | Desktop framework |
| tokio | 1.x | Async runtime |

### Python
| Package | Version | Purpose |
|---|---|---|
| maturin | >=1.14 | Build Rust extension wheels |
| pxr (usd-core) | latest | OpenUSD manipulation |
| gsplat | latest | Gaussian Splatting |
| ovphysx | 0.4.13+ | PhysX Python bindings |
| pymc | 5.x | Bayesian inference |
| open3d | latest | Point cloud processing |
| trimesh | latest | Mesh processing |
| omniverse-gsplat-converter | latest | PLY → USD |
| fastapi | latest | REST API sidecar |
| uvicorn | latest | ASGI server |
| pyinstaller | latest | Bundle Python to EXE |
| grpcio | latest | NuRec gRPC client |
| numpy | latest | Numerical computing |
| torch | latest (CUDA) | ML backend |

### Frontend
| Package | Version | Purpose |
|---|---|---|
| react | 19.x | UI framework |
| typescript | 5.x | Type safety |
| vite | 6.x | Build tool |
| tailwindcss | 4.x | Styling |
| shadcn/ui | latest | UI components |
| three | latest | 3D viewport |
| @tauri-apps/api | 2.x | Tauri IPC |
| d3 | 7.x | Timeline visualization |
| @visx/\* | latest | Probability plots |
| recharts | latest | Data charts |

### System Requirements
| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX 3060 12GB | NVIDIA RTX 4080 16GB+ |
| CUDA | 12.x | 13.x |
| NVIDIA Driver | 525+ | 580+ |
| Python | 3.12 | 3.12 |
| Rust | 1.83 (MSRV) | stable latest |
| OS | Windows 11 / Ubuntu 22.04 | Windows 11 |
| RAM | 16GB | 32GB+ |

---

## 13. GitHub Organization Setup

**Organization:** https://github.com/forensim  
**Status:** Created, no public repos yet.

### Planned Repositories

| Repo | Description |
|---|---|
| `forensim/forensim` | Main monorepo (this project) |
| `forensim/.github` | Org-level GitHub Actions, issue templates, code of conduct |
| `forensim/docs` | Documentation site (Astro or Docusaurus) |
| `forensim/datasets` | Public domain sample forensic scene datasets |

### Repository Initialization Checklist

```bash
# From D:\forensim\, once repo is created at github.com/forensim/forensim
git init
git add .
git commit -m "Initial commit: project plan and workspace setup"
git remote add origin https://github.com/forensim/forensim.git
git push -u origin main
```

### Recommended Labels for Issues
- `reconstruction` — photogrammetry pipeline
- `simulation` — physics / PhysX
- `inference` — probabilistic engine
- `ui` — frontend / Tauri
- `rust-core` — forensim-core crate
- `python-bridge` — PyO3 / Maturin
- `omniverse` — NVIDIA Omniverse integration

---

## Notes & Key Research Findings

### On NVIDIA NuRec
NuRec is available via NVIDIA NGC as a Docker container with a gRPC API. It is primarily designed for autonomous vehicle simulation but the underlying Gaussian Splatting + Harmonizer pipeline is directly applicable to forensic scene reconstruction. Licensing requires agreement with NVIDIA for the dataset; the container itself is available on NGC.

### On physx-rs (Rust)
The `physx-rs` crate (EmbarkStudios) was **archived in May 2026**. For Rust integration with PhysX, the recommended path is: use `ovphysx` (Python, officially maintained by NVIDIA) and call it from Rust via PyO3. This is simpler and more maintainable than maintaining C++ FFI bindings manually.

### On Omniverse Python Version
Isaac Sim 6.x requires **Python 3.12**. All Python code in this project must target 3.12. PyO3 0.29 fully supports Python 3.12.

### On the forensim name
The name `forensim` on GitHub is taken as a CRAN-mirrored R package for DNA mixture analysis (archived). However, the **organization** `forensim` is controlled by you and the R package is under `cran/forensim`, not `forensim/forensim`. The org name is clean for this project.

### On Gaussian Splatting → USD
Use `omniverse-gsplat-converter` (`pip install omniverse-gsplat-converter[usd]`). NVIDIA Omniverse natively supports the `ParticleField3DGaussianSplat` USD schema with full RTX path-tracing including shadows, reflections, and motion blur.

### On Probabilistic Math
The most academically sound approach for forensic sequence reconstruction combines:
1. **Bayesian inference** (PyMC) for evidence integration
2. **HMM with Viterbi** (Rust) for most-likely event sequence
3. **Monte Carlo PhysX** for physics likelihood `P(E|H)`
4. **Likelihood Ratios** for competing hypothesis comparison
This matches the current (2025) forensic science literature.

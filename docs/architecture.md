# ForenSim — Architecture Overview

## Table of Contents

1. [System Diagram](#1-system-diagram)
2. [Layer Interactions](#2-layer-interactions)
3. [Data Flow](#3-data-flow)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Module Descriptions](#5-module-descriptions)
6. [NuRec gRPC Integration](#6-nurec-grpc-integration)
7. [Annotation Layer and Inference Feedback](#7-annotation-layer-and-inference-feedback)

---

## 1. System Diagram

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
│  │  (upload, tags,    │  │  │  - nalgebra / statrs / bayes-rs     │ │
│  │   annotation ROIs) │  │  └──────────────┬──────────────────────┘ │
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
             │                    │  - Neural Rendering    │
             └────────────────────┴────────────────────────┘
```

The desktop app runs as a single Tauri v2 process. The React frontend communicates with the Rust backend over Tauri IPC (`invoke` / `listen`). The Rust backend spawns and manages a Python **sidecar** process (a PyInstaller-bundled FastAPI server) and communicates with it over localhost HTTP. The Python layer owns all NVIDIA toolchain calls; the Rust layer owns all probabilistic computation.

---

## 2. Layer Interactions

### React Frontend → Tauri Rust Backend

Communication uses Tauri's typed IPC:

```typescript
// Trigger a reconstruction job and stream progress events
const result = await invoke<ReconstructionResult>("start_reconstruction", {
    imageDir: selectedDir,
    method: "gaussian_splatting",
    outputPath: projectDir,
});

// Listen for streaming progress from the Rust layer
const unlisten = await listen<ReconstructionProgress>(
    "reconstruction-progress",
    (event) => setProgress(event.payload.percent)
);
```

The Rust backend validates parameters, dispatches the job to the Python sidecar, and re-emits progress events back to the WebView. All file system access goes through Tauri's capability-gated APIs — the React frontend never touches the filesystem directly.

### Rust Backend → Python Sidecar (FastAPI)

The Python sidecar is a FastAPI application bundled to a standalone EXE via PyInstaller and placed in `app/src-tauri/binaries/`. Tauri's `sidecar` API spawns it on app start and terminates it on app exit.

```
Rust (Tauri command)
    │  HTTP POST  localhost:8008/api/reconstruct
    ▼
Python FastAPI sidecar
    │  subprocess / library call
    ▼
forensim.reconstruct  ─►  COLMAP, gsplat, omniverse-gsplat-converter
forensim.simulate     ─►  ovphysx (PhysX Monte Carlo)
forensim.infer        ─►  PyMC, forensim._core (Rust via PyO3)
```

Long-running jobs stream `text/event-stream` (SSE) responses back to the Rust layer, which converts them to Tauri events for the frontend.

### Python → Rust (`forensim-core` via PyO3)

The `forensim-core` Rust crate is built as a Python extension module (`forensim._core`) using Maturin. The Python `forensim.infer` sub-package calls into Rust for the computationally intensive steps — Viterbi, Markov scoring, and Monte Carlo sampling — while keeping PyMC model definition in Python where it is most expressive.

```python
from forensim._core import MarkovChain, HiddenMarkovModel, BayesEngine

chain = MarkovChain(transition_matrix=T, states=state_labels)
score = chain.score_sequence(["entry", "confrontation", "impact", "exit"])

hmm = HiddenMarkovModel(transition=T, emission=E, initial=pi)
best_path = hmm.viterbi(observed_events)

engine = BayesEngine(prior=prior_vec)
posterior = engine.update(likelihoods)
```

The PyO3 bindings are GIL-aware; long Rust computations release the GIL so Python threads remain responsive.

---

## 3. Data Flow

```
Raw Evidence Input
(images / video / CCTV frames)
        │
        ▼
┌──────────────────────────────────┐
│  Step 1: Structure-from-Motion   │
│  Tool: COLMAP                    │
│  Output: sparse point cloud +    │
│          camera poses (COLMAP DB)│
└────────────────┬─────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌──────────────────────────┐
│ Traditional   │  │  Gaussian / Neural        │
│ Route:        │  │  Route (default):         │
│ OpenMVS       │  │  gsplat trainer           │
│ → Dense mesh  │  │  → .ply splat file        │
│ → PLY/OBJ     │  │                           │
│               │  │  NuRec gRPC (optional):   │
└───────┬───────┘  │  → rendered frames / mesh │
        │          └──────────┬───────────────┘
        │                     │
        ▼                     ▼
┌───────────────────────────────────┐
│  Step 2: USD Conversion           │
│  Tool: omniverse-gsplat-converter │
│  Output: .usdz / .usda scene      │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  Step 3: Scene Enrichment         │
│  Tool: pxr Python API             │
│  - Add PhysX properties           │
│  - Tag objects (rigid/soft/fluid) │
│  - Apply ROI evidence annotations │
│  - Define initial conditions      │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  Step 4: Physics Simulation       │
│  Tool: ovphysx (Python)           │
│  - Run N Monte Carlo scenarios    │
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
│  - Markov chain sequence scoring  │
│  - Viterbi HMM best path          │
│  - Bayesian posterior update      │
│  - Sensitivity analysis (LOO)     │
│  Output: ranked hypotheses +      │
│          P(H|E), Bayes factors,   │
│          Shannon entropy          │
└───────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  Step 6: Export                   │
│  - PDF forensic report            │
│  - USD scene package              │
│  - MP4 flythrough video           │
│  - Reconstruction manifest JSON   │
└───────────────────────────────────┘
```

### Key Artifacts at Each Stage

| Stage | Artifact | Format |
|---|---|---|
| SfM | Sparse point cloud + camera poses | COLMAP database / `.ply` |
| Gaussian Splatting | Trained 3DGS model | `.ply` (splat format) |
| NuRec (optional) | High-fidelity reconstruction | gRPC frames / mesh |
| USD conversion | Scene ready for PhysX | `.usda` / `.usdz` |
| Physics simulation | Per-step state trajectories | JSON |
| Inference | Ranked hypothesis posterior | JSON / in-memory |
| Export | Deliverables | PDF, USD, MP4 |

---

## 4. Key Design Decisions

### Why Rust for the probabilistic engine?

The core inference loop — Viterbi over event sequences, Markov scoring over hundreds of hypothesis chains, Monte Carlo integration over thousands of PhysX trajectories — is numerically intensive and benefits from:

- **Zero-cost abstractions**: tight inner loops compile to SIMD-optimized machine code via LLVM.
- **Rayon parallelism**: data-parallel Monte Carlo sampling with no thread-safety ceremony.
- **Memory safety without GC pauses**: no stop-the-world GC during live probability updates.
- **`nalgebra` + `ndarray`**: ergonomic matrix operations without a Python FFI round-trip.

Python's PyMC is retained for complex Bayesian network definition (NUTS/HMC sampler, Gaussian Process surrogates) where expressiveness matters more than raw throughput.

### Why Maturin + PyO3?

Maturin is the canonical build system for mixed Rust/Python projects. It produces a standard wheel (`forensim_core-*.whl`) that installs like any Python package. PyO3 provides idiomatic Rust bindings — Python types map directly to Rust types, errors propagate naturally as Python exceptions, and GIL handling is explicit rather than implicit. The alternative (CFFI or ctypes) would require manual memory management and lose type safety at the boundary.

### Why Tauri v2 over Electron?

| Concern | Electron | Tauri v2 |
|---|---|---|
| Binary size | ~150 MB (ships Chromium) | ~5–15 MB (uses OS WebView) |
| Memory footprint | ~300 MB baseline | ~50 MB baseline |
| Native file access | Requires IPC anyway | First-class Rust commands |
| Sidecar management | Node child_process | Built-in sidecar API |
| GPU access for forensics | WebGL only | Full native via sidecar |

Tauri also allows the forensim-core Rust crate to be used directly in the Tauri backend without any FFI boundary — they share the same Cargo workspace.

### Why FastAPI as the Python sidecar?

The Python layer must be bundled as a standalone process (PyInstaller EXE) because NVIDIA Omniverse, Isaac Sim, and ovphysx have complex dependency trees that cannot be imported into the Tauri Rust process. FastAPI gives:

- Auto-generated OpenAPI schema (TypeScript client generated from it)
- Server-Sent Events for streaming long reconstruction jobs
- Async task queue via `asyncio` / `BackgroundTasks`
- Clean separation of concerns: Rust owns the desktop shell; Python owns the NVIDIA stack

### Why OpenUSD as the scene format?

USD is NVIDIA's canonical scene format across Omniverse, Isaac Sim, and NuRec. Using it as ForenSim's native scene format means:

- Reconstructed scenes open directly in Omniverse Composer for inspection
- PhysX material properties are stored as USD metadata alongside geometry
- Evidence annotations can be encoded as USD custom attributes
- The scene is self-contained and portable for courtroom presentation

---

## 5. Module Descriptions

### `forensim-core` — Rust crate (`crates/forensim-core/`)

The computational heart of the system. Exposed to Python as the `forensim._core` extension module.

| Source file | Responsibility |
|---|---|
| `lib.rs` | PyO3 module root; registers all Python-visible classes |
| `markov.rs` | `MarkovChain` struct — transition matrix, sequence scoring, steady-state |
| `hmm.rs` | `HiddenMarkovModel` — forward-backward, Viterbi, log-likelihood |
| `bayes.rs` | `BayesEngine` — prior/likelihood/posterior cycle, Bayes factor computation |
| `monte_carlo.rs` | `MonteCarloSampler` — Rayon-parallel trajectory integration |
| `usd.rs` | Rust-side USD type wrappers (scene graph nodes, prim references) |
| `stats.rs` | Utility wrappers: `statrs` distribution helpers, `ndarray-stats` summary |

### `forensim-py` — Python package (`python/forensim/`)

The Python orchestration layer and PyO3 consumer. Also contains the FastAPI sidecar.

```
python/forensim/
├── _core.pyi          # Type stubs for the Rust extension module
├── __init__.py
├── reconstruct/       # forensim-reconstruct sub-package
├── simulate/          # forensim-simulate sub-package
├── infer/             # forensim-infer sub-package
└── api/               # forensim-api (FastAPI sidecar)
```

### `forensim-reconstruct` — `python/forensim/reconstruct/`

Orchestrates the image-to-3D pipeline.

| Module | Role |
|---|---|
| `colmap.py` | Subprocess wrapper for COLMAP CLI; feature extraction, matching, SfM |
| `gsplat.py` | Gaussian Splatting pipeline — real trainer (requires nvcc) or fallback exporter |
| `nurec.py` | gRPC client for NVIDIA NuRec; health-check, scene listing, frame rendering |
| `usd_export.py` | Converts `.ply` splat output to `.usda` via `omniverse-gsplat-converter` |

The orchestrator (`__init__.py`) calls these in sequence, emitting progress callbacks consumed by the FastAPI SSE stream.

### `forensim-simulate` — `python/forensim/simulate/`

Owns physics simulation and scene enrichment.

| Module | Role |
|---|---|
| `scene_builder.py` | Reads USD scene; adds `PhysicsRigidBodyAPI`, material properties, and initial conditions derived from evidence |
| `physx_runner.py` | Wraps `ovphysx.PhysX`; runs M scenario variations in sequence, collects `transforms` per timestep |
| `scenario.py` | Generates scenario parameter space (initial velocities, masses, friction coefficients) for Monte Carlo sweep |

### `forensim-infer` — `python/forensim/infer/`

Connects physics trajectories to probabilistic hypotheses.

| Module | Role |
|---|---|
| `evidence.py` | Converts annotated ROIs and tags to log-likelihood vectors |
| `sequence.py` | Builds event vocabulary and transition matrix from domain knowledge |
| `bayesian.py` | PyMC model: Gaussian Process likelihood surface, NUTS sampler, posterior summary |

This layer calls `forensim._core` (Rust) for Viterbi and Markov scoring, then uses PyMC for the full Bayesian network where latent variables are needed.

### `forensim-api` — `python/forensim/api/`

FastAPI application bundled as a sidecar EXE.

| Endpoint | Method | Description |
|---|---|---|
| `/api/reconstruct` | POST | Start reconstruction; streams SSE progress |
| `/api/simulate` | POST | Run PhysX Monte Carlo scenarios |
| `/api/infer` | POST | Run full probabilistic inference pipeline |
| `/api/export` | POST | Export PDF report, USD package, MP4 video |
| `/api/health` | GET | Sidecar health check |

The Tauri backend calls `/api/health` at startup to confirm the sidecar is ready before enabling UI controls.

---

## 6. NuRec gRPC Integration

NVIDIA NuRec is a neural reconstruction service available as a Docker container on NVIDIA NGC. It implements Gaussian Splatting augmented with a Harmonizer network for relighting.

### Architecture

```
ForenSim Python layer
    │
    │  gRPC (grpcio)
    ▼
NuRec gRPC Server  (localhost:8080 by default; NUREC_ADDRESS env var)
    │  Docker container / NGC service
    ▼
NuRec Gaussian Splatting + Harmonizer pipeline
    │
    ▼
Rendered frames / reconstructed scene mesh
```

### `forensim.reconstruct.nurec` client

The NuRec client (`nurec.py`) exposes three operations:

| Method | gRPC call | Description |
|---|---|---|
| `health_check()` | `NuRecService/HealthCheck` | Verify server is available |
| `list_scenes()` | `NuRecService/ListScenes` | Enumerate loaded scene IDs |
| `load_scene(path)` | `NuRecService/LoadScene` | Upload images and trigger reconstruction |
| `render_frame(scene_id, pose)` | `NuRecService/RenderFrame` | Render a novel view from a camera pose |

### When to use NuRec vs. local gsplat

| Criterion | Local gsplat | NuRec gRPC |
|---|---|---|
| Setup complexity | CUDA Toolkit + nvcc | Docker + NGC account |
| Speed (first run) | ~5 min JIT compilation | Instant (pre-compiled container) |
| Image quality | Good | Higher (Harmonizer relighting) |
| Offline capability | Yes | Requires running container |
| Recommended for | Development, CI | Production reconstruction |

NuRec is selected by passing `method="nurec"` to the reconstruction API or toggling the method in the UI.

---

## 7. Annotation Layer and Inference Feedback

The annotation layer is a React component (`AnnotationLayer.tsx`) that renders on top of the evidence image panel. It provides:

- **Rectangle tool** — draw bounding boxes around objects of interest
- **Polygon tool** — trace irregular regions (bloodstain patterns, scuff marks)
- **Tag assignment** — attach semantic tags from a controlled vocabulary (`blood`, `impact`, `trajectory`, `footprint`, `weapon`, `entry-point`, `exit-point`)
- **Confidence slider** — analyst-supplied confidence for each annotation (0–1)

### How annotations feed the inference pipeline

Each annotation ROI is serialized to a JSON evidence record:

```json
{
  "id": "ann-001",
  "image_id": "img-003",
  "type": "polygon",
  "coords": [[120, 340], [155, 340], [160, 390], [115, 395]],
  "tag": "blood",
  "analyst_confidence": 0.85
}
```

The `forensim.infer.evidence` module maps each tag to a log-likelihood contribution `log P(e_i | H)` for each competing hypothesis:

```python
TAG_LIKELIHOOD_TABLE = {
    #  tag           H_entry  H_exit   H_struggle  H_accident
    "blood":        [-0.5,    -1.2,    -0.3,       -0.8],
    "impact":       [-0.6,    -0.9,    -0.4,       -0.5],
    "footprint":    [-0.3,    -0.4,    -0.7,       -1.1],
    "trajectory":   [-0.4,    -0.4,    -0.8,       -0.6],
    "entry-point":  [-0.2,    -2.0,    -1.5,       -1.8],
    "exit-point":   [-2.1,    -0.2,    -1.4,       -1.7],
}
```

The analyst confidence acts as a weight: the contribution of annotation `i` is scaled by `confidence_i` before being added to the log-likelihood sum. This lets analysts flag uncertain or ambiguous evidence without dropping it entirely.

The aggregated log-likelihood vector feeds directly into `BayesEngine.update()` in `forensim-core`, producing the updated posterior over hypotheses. The sensitivity analysis step (`LOO re-ranking`) then systematically removes one annotation at a time and re-runs the update, reporting which annotation has the largest effect on the top-ranked hypothesis rank order. See [probability-model.md](probability-model.md) for the full mathematical treatment.

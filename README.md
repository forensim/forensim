# ForenSim

**Forensic Scene Reconstruction & Probabilistic Event Analysis**

[![CI](https://github.com/forensim/forensim/actions/workflows/ci.yml/badge.svg)](https://github.com/forensim/forensim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.83+-orange.svg)](https://rustup.rs)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)

ForenSim takes 2D photographic evidence — crime scene photos, CCTV frames, dashcam footage — and produces an interactive 3D simulation of the most likely sequence of events. Photographs are reconstructed into a Gaussian Splat scene, physics-simulated with NVIDIA PhysX, and run through a Bayesian / HMM inference engine to produce ranked, quantified hypotheses with Bayes factors and sensitivity analysis.

- Reconstructs scenes from photographs using COLMAP Structure-from-Motion + 3D Gaussian Splatting
- Enriches the resulting USD scene with PhysX rigid-body properties for Monte Carlo simulation
- Scores competing event-sequence hypotheses using Markov chains, Viterbi HMM, and Bayesian updating
- Surfaces the most influential pieces of evidence via leave-one-out sensitivity analysis
- Delivers results through a dark-themed Tauri v2 desktop app with a live 3D viewport

---

## Features

- **3D Gaussian Splatting reconstruction** from crime scene photographs (gsplat trainer + fallback exporter)
- **NVIDIA NuRec gRPC integration** for high-fidelity neural radiance field rendering
- **Monte Carlo PhysX simulation** of forensic events over hundreds of scenario variations
- **Bayesian / HMM hypothesis ranking** with posterior probabilities, Bayes factors, and Shannon entropy
- **Evidence ROI annotation layer** (rectangle + polygon tools) with tag-driven likelihoods
- **Sensitivity analysis** — leave-one-out re-ranking shows which annotation most drives the conclusion
- **Export:** PDF forensic report, USD scene package, MP4 flythrough video
- **Dark forensic UI** built with Tauri v2 + React 19 + Tailwind CSS

---

## Architecture

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
             │                    └──────────┬────────────┘
             │                               │
             │                    ┌──────────▼────────────┐
             │                    │  NVIDIA NuRec          │
             │                    │  (Docker / NGC)        │
             └────────────────────┴────────────────────────┘
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| 3D Reconstruction | COLMAP + gsplat (3D Gaussian Splatting) |
| Neural Rendering | NVIDIA NuRec (gRPC API) |
| 3D Scene Format | OpenUSD (`pxr`) |
| Physics Engine | NVIDIA PhysX via `ovphysx` |
| Probabilistic Core | Rust — `forensim-core` (nalgebra, statrs, rayon) |
| Python Bridge | PyO3 0.29 + Maturin 1.14 |
| Python Toolchain | Python 3.12 — PyMC, gsplat, ovphysx, FastAPI |
| Desktop Framework | Tauri v2 (Rust) |
| Frontend | React 19 + TypeScript + Vite |
| UI Components | shadcn/ui + Tailwind CSS 4 |
| 3D Viewport | Three.js + GaussianSplats3D |
| Data Visualization | VisX + D3.js + Recharts |

---

## Quick Start

Requirements: Windows 11 / Ubuntu 22.04+, NVIDIA RTX GPU, CUDA 12.x, Rust 1.83+, Node.js 24, Python 3.12.

```powershell
# 1. Set up Python environment
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
maturin develop --release

# 2. Run the sample demo (no CUDA Toolkit required)
python scripts/run_demo.py

# 3. Start the desktop app
cd app
npm install
npm run tauri dev
```

The demo script runs the full pipeline on `assets/sample-scenes/crime-scene-01/` using the fallback Gaussian Splatting exporter (no CUDA Toolkit needed). See [AGENTS.md](AGENTS.md) for the complete setup guide including CUDA Toolkit, real gsplat trainer, and NuRec Docker configuration.

---

## Project Structure

```
forensim/
├── crates/
│   └── forensim-core/       # Rust: HMM, Markov chain, Bayes, Monte Carlo
├── python/
│   └── forensim/
│       ├── reconstruct/     # COLMAP, gsplat, NuRec gRPC, USD export
│       ├── simulate/        # PhysX Monte Carlo runner, scene builder
│       ├── infer/           # Bayesian inference, evidence scoring
│       └── api/             # FastAPI sidecar (bundled as EXE)
├── app/
│   ├── src/                 # React 19 frontend
│   └── src-tauri/           # Tauri v2 Rust backend + sidecar binaries
├── docs/
│   ├── architecture.md      # System design and layer interactions
│   ├── reconstruction-guide.md  # COLMAP + gsplat setup and troubleshooting
│   └── probability-model.md     # Markov / HMM / Bayesian math reference
├── assets/
│   └── sample-scenes/       # Public domain test evidence images
├── scripts/
│   └── run_demo.py          # End-to-end demo pipeline
├── PLAN.md                  # Full architecture plan
└── AGENTS.md                # Developer commands and environment guide
```

---

## Roadmap

- [x] Phase 0 — Project scaffold, workspace, CI skeleton, AGENTS.md
- [x] Phase 1 — Reconstruction pipeline: COLMAP + gsplat + USD conversion + manifest
- [x] Phase 2 — Probabilistic engine: Markov chain, Viterbi HMM, Bayesian updating (Rust)
- [x] Phase 3 — Inference integration: PyMC models, evidence scoring, hypothesis ranking
- [x] Phase 4 — Advanced features: NuRec gRPC, annotation layer, export (PDF/USD/MP4), sensitivity analysis
- [ ] Phase 1.5 — Evidence ingestion UI + frontend ↔ FastAPI reconstruction connection
- [ ] Phase 2 (sim) — PhysX Monte Carlo runner + simulation trajectory UI
- [ ] Phase 5 — Dark theme polish, public sample dataset, documentation site, demo video

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System diagram, layer interactions, design decisions, module descriptions |
| [docs/reconstruction-guide.md](docs/reconstruction-guide.md) | Prerequisites, step-by-step reconstruction, troubleshooting, gsplat modes |
| [docs/probability-model.md](docs/probability-model.md) | Markov chain, HMM, Bayesian inference, sensitivity analysis — math reference |
| [AGENTS.md](AGENTS.md) | Developer environment setup, common commands, CUDA notes |
| [PLAN.md](PLAN.md) | Full architecture plan with all design decisions |

---

## Contributing

Contributions are welcome. Please open an issue before starting significant work so we can discuss the approach.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Ensure CI passes: `cargo check --workspace && ruff check python/ && cd app && npm run build`
4. Submit a pull request against `main`

See [AGENTS.md](AGENTS.md) for the complete developer setup and command reference.

---

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

**GitHub:** [github.com/forensim/forensim](https://github.com/forensim/forensim)

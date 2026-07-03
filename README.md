# ForenSim

**Forensic Scene Reconstruction & Probabilistic Event Analysis Platform**

ForenSim takes 2D photographic evidence (crime scene photos, CCTV frames, dashcam footage) and reconstructs an interactive 3D simulation of the most likely sequence of events that occurred — complete with ranked probabilistic hypotheses backed by physics simulation.

---

## What it does

```
2D Images / Video Frames
        │
        ▼  COLMAP + Gaussian Splatting
Reconstructed 3D Scene (USD / USDZ)
        │
        ▼  NVIDIA PhysX (ovphysx)
Monte Carlo Physics Simulations
        │
        ▼  Bayesian Inference + HMM (Rust core)
Ranked Event Sequence Hypotheses
        │
        ▼  Tauri Desktop App
Interactive 3D Visualization + Probability Dashboard
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| 3D Reconstruction | COLMAP + gsplat (Gaussian Splatting) |
| Neural Rendering | NVIDIA NuRec (gRPC) |
| 3D Scene Format | OpenUSD (`pxr`) |
| Simulation | NVIDIA Omniverse / Isaac Sim 6.x |
| Physics Engine | NVIDIA PhysX via `ovphysx` |
| Performance Core | Rust (PyO3 0.29 + Maturin 1.14) |
| Python Bridge | Python 3.12 (Omniverse APIs, ML, gsplat) |
| Desktop App | Tauri v2 + React 19 + TypeScript |
| UI Components | shadcn/ui + Tailwind CSS |
| Visualization | Three.js, VisX, D3.js, Recharts |
| Probabilistic Math | PyMC, Bayesian networks, HMM, Markov chains |

---

## Architecture

```
┌─────────────────────── Tauri Desktop App ──────────────────────────┐
│  React UI (shadcn/ui + Tailwind)  │  Rust Backend (Tauri v2)        │
│  • 3D viewport (Three.js / 3DGS)  │  • forensim-core (HMM, Bayes)   │
│  • Probability panels (VisX)      │  • IPC commands                  │
│  • Event timeline (D3.js)         │  • Python sidecar management     │
│  • Hypothesis ranker              │                                   │
└──────────────────────────────────────────────────────────────────────┘
                         │ HTTP (FastAPI sidecar)
┌──────────────────────── Python Layer ───────────────────────────────┐
│  forensim.reconstruct   │  forensim.simulate  │  forensim.infer      │
│  • COLMAP               │  • ovphysx (PhysX)  │  • PyMC (Bayes)      │
│  • gsplat               │  • Monte Carlo      │  • Sequence scoring  │
│  • NuRec gRPC           │  • USD enrichment   │  • LR calculation    │
│  • USD export           │                     │                      │
└─────────────────────────────────────────────────────────────────────┘
                         │ USD / gRPC
        NVIDIA Omniverse / Isaac Sim / NuRec
```

---

## Project Structure

```
forensim/
├── crates/forensim-core/    # Rust: HMM, Markov, Bayesian, Monte Carlo
├── python/forensim/         # Python package (Maturin wheel)
│   ├── reconstruct/         # COLMAP, gsplat, NuRec, USD export
│   ├── simulate/            # PhysX simulation
│   ├── infer/               # Probabilistic inference
│   └── api/                 # FastAPI REST sidecar
├── app/                     # Tauri v2 + React + TypeScript
│   ├── src/                 # React frontend
│   └── src-tauri/           # Rust Tauri backend
├── PLAN.md                  # Full architecture plan
└── AGENTS.md                # Developer guide & commands
```

---

## Getting Started

See [AGENTS.md](AGENTS.md) for full setup instructions.

**Quick start (after prerequisites):**

```powershell
# Set up Python environment
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

# Build Rust extension
maturin develop --release

# Start the desktop app
cd app
npm install
npm run tauri dev
```

---

## Requirements

- Windows 11 / Ubuntu 22.04+
- NVIDIA RTX GPU (RTX 3060 12GB minimum, RTX 4080+ recommended)
- CUDA 12.x+
- Rust 1.83+
- Node.js 24.x
- Python 3.12 (via uv)

---

## Roadmap

- [x] Phase 0 — Project scaffold & workspace setup
- [ ] Phase 1 — Reconstruction pipeline (COLMAP + gsplat + USD)
- [ ] Phase 2 — Physics simulation (PhysX Monte Carlo)
- [ ] Phase 3 — Probabilistic engine (HMM + Bayes + sequence scoring)
- [ ] Phase 4 — Advanced features (NuRec, blood spatter, export)
- [ ] Phase 5 — Polish, docs, demo

---

## License

MIT OR Apache-2.0

---

## Organization

**GitHub:** https://github.com/forensim  
**Project repo:** https://github.com/forensim/forensim

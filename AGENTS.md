# ForenSim — Developer Reference

## Environment Requirements

| Tool | Version | Install |
|---|---|---|
| Rust | 1.83+ (stable) | https://rustup.rs |
| Node.js | 24.x | https://nodejs.org |
| npm | 11.x | bundled with Node |
| uv | latest | `irm https://astral.sh/uv/install.ps1 \| iex` |
| Python | 3.12 (managed by uv) | `uv python install 3.12` |
| CUDA Toolkit | 12.x+ | https://developer.nvidia.com/cuda-downloads |
| NVIDIA Driver | 525+ | Windows Update or NVIDIA website |

## First-Time Setup

```powershell
# 1. Install uv (Python manager)
irm https://astral.sh/uv/install.ps1 | iex
$env:Path = "C:\Users\$env:USERNAME\.local\bin;$env:Path"

# 2. Install Python 3.12
uv python install 3.12

# 3. Create virtualenv and install Python deps
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

# 4. Install Maturin and build the Rust extension
uv pip install maturin
maturin develop --release

# 5. Install Tauri CLI and frontend deps
cd app
npm install
cd ..

# 6. Install Tauri CLI globally (optional, also available via npm)
cargo install tauri-cli --version "^2"
```

## Common Commands

### Rust (forensim-core)

```powershell
# Check compilation
cargo check --workspace

# Run tests
cargo test --workspace

# Build release
cargo build --release --workspace

# Build Python extension (installs into active venv)
maturin develop --release

# Build distributable wheel
maturin build --release
```

### Python

```powershell
# Activate venv (always do this first)
.venv\Scripts\Activate.ps1

# Run FastAPI dev server
uvicorn forensim.api.main:app --host 127.0.0.1 --port 8008 --reload

# Run tests
pytest python/tests/

# Lint
ruff check python/

# Type check
mypy python/forensim/
```

### Tauri / Frontend

```powershell
cd app

# Install dependencies
npm install

# Start dev server (hot reload — opens desktop window)
npm run tauri dev

# Build production app
npm run tauri build

# Frontend only (no Tauri)
npm run dev
```

### Full Build Pipeline

```powershell
# 1. Build Rust core + Python extension
maturin build --release

# 2. Build Python FastAPI sidecar to EXE
cd python
pyinstaller forensim/api/build.spec --onefile
copy dist\forensim-api.exe ..\app\src-tauri\binaries\

# 3. Build Tauri app (bundles everything)
cd ..\app
npm run tauri build
```

## Project Layout

```
forensim/
├── Cargo.toml               # Workspace root
├── pyproject.toml           # Maturin + Python package
├── AGENTS.md                # This file
├── PLAN.md                  # Architecture plan
├── README.md
│
├── crates/forensim-core/    # Rust: HMM, Markov, Bayes, Monte Carlo
├── python/forensim/         # Python: reconstruct, simulate, infer, api
├── app/                     # Tauri v2 + React + TypeScript
│   ├── src/                 # React frontend
│   └── src-tauri/           # Rust Tauri backend
│
├── .github/workflows/       # CI
├── docs/
└── scripts/
```

## Key Environment Variables

```powershell
# Optional: point to local Omniverse Kit installation
$env:ISAAC_SIM_PATH = "C:\path\to\isaac_sim"

# NuRec server address (default: localhost:8080)
$env:NUREC_ADDRESS = "localhost:8080"

# ForenSim API port (default: 8008)
$env:FORENSIM_API_PORT = "8008"
```

## Dependency Notes

- **physx-rs is archived** — use `ovphysx` (Python) instead, called from Rust via PyO3
- **Isaac Sim 6.x requires Python 3.12** — this project is locked to 3.12
- **COLMAP must be installed separately** — not available via pip
- **gsplat requires CUDA** — GPU mandatory for Gaussian Splatting training
- `omniverse-gsplat-converter` handles PLY → USD conversion without needing a full Omniverse install

## Running Tests

```powershell
# Rust unit tests
cargo test --workspace

# Python unit tests (after maturin develop)
pytest python/tests/ -v

# Integration test: reconstruction pipeline (requires COLMAP + CUDA)
pytest python/tests/integration/ -v -m integration
```

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
| COLMAP | 4.x | https://github.com/colmap/colmap/releases |

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

# 5. Install COLMAP (Windows)
# Download the latest CUDA-enabled Windows binary from:
#   https://github.com/colmap/colmap/releases
# Extract to a persistent location, e.g. D:\Tools\COLMAP, and add its bin folder to PATH:
$env:Path = "D:\Tools\COLMAP\bin;$env:Path"

# 6. Install CUDA Toolkit 12.6 (required for real Gaussian Splatting trainer)
# Download: https://developer.nvidia.com/cuda-12-6-0-download-archive
#   → Windows → x86_64 → 11 → exe (local)
# During install choose "Custom" and check: CUDA Toolkit + Development Tools.
# Uncheck "Display Driver" (yours is already newer).
# After install, set CUDA_HOME to enable gsplat JIT compilation:
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
# Add nvcc to PATH permanently via System Properties → Environment Variables.

# 7. Install PyTorch CUDA 12.6 build (replaces the CPU-only version)
uv pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 `
    --index-url https://download.pytorch.org/whl/cu126

# 8. Install Gaussian Splatting trainer and viewer deps
uv pip install gaussian-splatting>=2.3 viser>=1.0 nerfview>=0.1 imageio>=2.33 `
    tyro>=0.8 tensorboard>=2.14

# 9. Install Tauri CLI and frontend deps
cd app
npm install
cd ..

# 10. Install Tauri CLI globally (optional, also available via npm)
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

# Optional: override the COLMAP binary path
$env:COLMAP_PATH = "D:\Tools\COLMAP\bin\colmap.exe"

# NuRec server address (default: localhost:8080)
$env:NUREC_ADDRESS = "localhost:8080"

# ForenSim API port (default: 8008)
$env:FORENSIM_API_PORT = "8008"

# CUDA Toolkit root — required for gsplat JIT compilation (both must be set)
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

# RTX 3060 Ti is sm_86; avoids compiling for all archs (faster JIT)
$env:TORCH_CUDA_ARCH_LIST = "8.6"

# Gaussian Splatting trainer mode (default: 1 = use fallback exporter)
# Set to 0 to force the real gaussian-splatting trainer (requires CUDA_HOME):
$env:FORENSIM_GSPLAT_FALLBACK = "0"
```

## CUDA / gsplat JIT Compilation Notes (Windows)

gsplat's CUDA kernels are JIT-compiled on first import using `torch.utils.cpp_extension`.
On Windows, this requires the **VS 2022 Build Tools** `cl.exe` — not the newer VS 2026 Community
edition which CUDA 12.6 doesn't support. These env vars must be set in the shell before running:

```powershell
# Required: VS 2022 Build Tools MSVC compiler must be first in PATH
# (CUDA 12.6 only supports MSVC up to VS 2022 / cl.exe v19.4x)
$msvc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$env:Path = "$msvc;D:\forensim\.venv\Scripts;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" + $env:Path
# Strip VS 2026 entries from PATH if present (they cause "unsupported MSVC version" errors)
$env:Path = ($env:Path -split ";" | Where-Object { $_ -notlike "*Visual Studio\18*" }) -join ";"

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:TORCH_CUDA_ARCH_LIST = "8.6"
```

The compiled extension is cached at `%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py312_cu126\gsplat_cuda\`.
Compilation takes ~5 min on first run; subsequent runs load the cached `.pyd` instantly.

**One-time patch** applied to `gsplat/cuda/_backend.py` in the venv to remove the GCC-only
`-Wno-attributes` flag that breaks MSVC. If you recreate the venv, re-apply it:
```python
# In .venv/Lib/site-packages/gsplat/cuda/_backend.py, line ~177:
# Change:  extra_cflags = [opt_level, "-Wno-attributes"]
# To:
import platform
if platform.system() == "Windows":
    extra_cflags = [opt_level]
else:
    extra_cflags = [opt_level, "-Wno-attributes"]
```

## Dependency Notes

- **physx-rs is archived** — use `ovphysx` (Python) instead, called from Rust via PyO3
- **Isaac Sim 6.x requires Python 3.12** — this project is locked to 3.12
- **COLMAP must be installed separately** — not available via pip; set `COLMAP_PATH` or add its `bin` folder to PATH
- **gsplat requires CUDA** — GPU mandatory for Gaussian Splatting training
  - The fallback exporter (`FORENSIM_GSPLAT_FALLBACK=1`) works with any CUDA torch build (no nvcc needed)
  - The real trainer (`FORENSIM_GSPLAT_FALLBACK=0`) additionally requires the CUDA Toolkit (nvcc) for JIT compilation
  - gsplat prebuilt wheels at https://docs.gsplat.studio/whl only cover Python 3.10; for Python 3.12 the JIT path is the only option
- `omniverse-gsplat-converter` handles PLY → USD conversion without needing a full Omniverse install
- **nerfstudio is split into a separate `nerf` extra** because it pins protobuf 3.20, which conflicts with modern grpcio-tools
- **torch must be the CUDA build** — install from `https://download.pytorch.org/whl/cu126`, not from PyPI default

## Running Tests

```powershell
# Rust unit tests
cargo test --workspace

# Python unit tests (after maturin develop)
pytest python/tests/ -v

# Integration test: reconstruction pipeline (requires COLMAP + CUDA)
pytest python/tests/integration/ -v -m integration
```

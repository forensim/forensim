# ForenSim Development Environment Setup
# Run from repo root: .\scripts\setup-dev.ps1
#
# Prerequisites:
#   - Rust (https://rustup.rs)
#   - Node.js 24.x (https://nodejs.org)
#   - uv (https://astral.sh/uv)
#
# Optional (for full pipeline):
#   - COLMAP 4.x  (https://github.com/colmap/colmap/releases)
#   - CUDA Toolkit 12.x (https://developer.nvidia.com/cuda-downloads)

param(
    [switch]$SkipRust,
    [switch]$SkipNpm,
    [switch]$SkipMaturin
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n>> $Message" -ForegroundColor Cyan
}

function Write-OK {
    param([string]$Message)
    Write-Host "  [OK]  $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "  [WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "  [FAIL] $Message" -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "ForenSim Dev Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Forensic Scene Reconstruction - Development Environment" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Verify we're at the repo root
# ---------------------------------------------------------------------------
if (-not (Test-Path ".\pyproject.toml") -or -not (Test-Path ".\Cargo.toml")) {
    Write-Fail "Must be run from the forensim repo root."
    Write-Fail "  cd D:\forensim"
    Write-Fail "  .\scripts\setup-dev.ps1"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 1 - Check uv
# ---------------------------------------------------------------------------
Write-Step "Step 1: Checking uv (Python manager)"

$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Warn "uv not found in PATH."
    Write-Warn "Install it with:"
    Write-Warn "  irm https://astral.sh/uv/install.ps1 | iex"
    Write-Warn "Then re-open your terminal and re-run this script."
    exit 1
}

$uvVersion = (uv --version 2>&1) -replace "uv ", ""
Write-OK "uv $uvVersion found at $($uvPath.Source)"

# ---------------------------------------------------------------------------
# Step 2 - Create virtualenv
# ---------------------------------------------------------------------------
Write-Step "Step 2: Creating Python 3.12 virtual environment (.venv)"

if (Test-Path ".\.venv\Scripts\python.exe") {
    Write-OK ".venv already exists — skipping creation"
} else {
    Write-Host "  Running: uv venv --python 3.12 .venv" -ForegroundColor DarkGray
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "uv venv failed. Ensure Python 3.12 is available: uv python install 3.12"
        exit 1
    }
    Write-OK ".venv created with Python 3.12"
}

# ---------------------------------------------------------------------------
# Step 3 - Install Python dependencies
# ---------------------------------------------------------------------------
Write-Step "Step 3: Installing Python dependencies"

Write-Host "  Running: uv pip install -e `".[dev]`"" -ForegroundColor DarkGray
uv pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Fail "uv pip install failed. Check pyproject.toml for errors."
    exit 1
}
Write-OK "Python dependencies installed"

# ---------------------------------------------------------------------------
# Step 4 - Build Rust extension with maturin
# ---------------------------------------------------------------------------
if (-not $SkipMaturin) {
    Write-Step "Step 4: Building Rust extension (forensim._core) with maturin"

    # Ensure maturin is available
    $maturinCmd = Get-Command maturin -ErrorAction SilentlyContinue
    if (-not $maturinCmd) {
        Write-Host "  maturin not in PATH — installing via uv pip ..." -ForegroundColor DarkGray
        uv pip install maturin
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to install maturin."
            exit 1
        }
    }

    # Check Rust is available
    $cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargoCmd) {
        Write-Fail "cargo not found. Install Rust from https://rustup.rs"
        exit 1
    }
    $rustVersion = (rustup show active-toolchain 2>&1) -replace "stable-.*", "stable"
    Write-OK "Rust toolchain: $rustVersion"

    Write-Host "  Running: maturin develop --release" -ForegroundColor DarkGray
    maturin develop --release
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "maturin develop --release failed. Check crates/forensim-core for errors."
        exit 1
    }
    Write-OK "forensim._core Rust extension built and installed into .venv"
} else {
    Write-Warn "Step 4 skipped (--SkipMaturin)"
}

# ---------------------------------------------------------------------------
# Step 5 - Install npm dependencies
# ---------------------------------------------------------------------------
if (-not $SkipNpm) {
    Write-Step "Step 5: Installing frontend npm dependencies"

    if (-not (Test-Path ".\app\package.json")) {
        Write-Fail "app\package.json not found. Is this the right repo root?"
        exit 1
    }

    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        Write-Fail "npm not found. Install Node.js 24.x from https://nodejs.org"
        exit 1
    }
    $npmVersion = (npm --version 2>&1)
    Write-OK "npm $npmVersion found"

    Write-Host "  Running: npm install (in app/)" -ForegroundColor DarkGray
    Push-Location app
    npm install
    $npmRC = $LASTEXITCODE
    Pop-Location

    if ($npmRC -ne 0) {
        Write-Fail "npm install failed. Check app/package.json for errors."
        exit 1
    }
    Write-OK "Frontend npm dependencies installed"
} else {
    Write-Warn "Step 5 skipped (--SkipNpm)"
}

# ---------------------------------------------------------------------------
# Step 6 - Generate sample dataset
# ---------------------------------------------------------------------------
Write-Step "Step 6: Generating sample dataset"

$sampleDir = ".\assets\sample-scenes\crime-scene-01\images"
if ((Test-Path $sampleDir) -and ((Get-ChildItem $sampleDir -Filter "img_*.jpg").Count -eq 8)) {
    Write-OK "Sample dataset already exists"
} else {
    Write-Host "  Running: python scripts\generate_sample_dataset.py" -ForegroundColor DarkGray
    & ".venv\Scripts\python.exe" "scripts\generate_sample_dataset.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Sample dataset generation failed — run manually later:"
        Write-Warn "  .venv\Scripts\python.exe scripts\generate_sample_dataset.py"
    } else {
        Write-OK "Sample dataset generated"
    }
}

# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Run the full demo pipeline:" -ForegroundColor White
Write-Host "       .venv\Scripts\python.exe scripts\run_demo.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Start the FastAPI backend:" -ForegroundColor White
Write-Host "       .venv\Scripts\uvicorn forensim.api.main:app --host 127.0.0.1 --port 8008 --reload" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Start the Tauri desktop app (opens a window):" -ForegroundColor White
Write-Host "       cd app; npm run tauri dev" -ForegroundColor Cyan
Write-Host ""
Write-Host "  4. Run tests:" -ForegroundColor White
Write-Host "       cargo test --workspace" -ForegroundColor Cyan
Write-Host "       .venv\Scripts\pytest python/tests/ -v" -ForegroundColor Cyan
Write-Host ""
Write-Host "Optional GPU setup (Gaussian Splatting):" -ForegroundColor DarkGray
Write-Host "  See AGENTS.md -> CUDA / gsplat JIT Compilation Notes" -ForegroundColor DarkGray
Write-Host ""

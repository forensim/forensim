#!/usr/bin/env python3
"""
ForenSim Demo Script
====================
Runs the full forensic reconstruction and inference pipeline
on the built-in sample crime scene dataset.

Usage:
    python scripts/run_demo.py [--skip-reconstruction]

Requirements:
    - ForenSim Python package installed (uv pip install -e ".[dev]")
    - maturin develop --release (builds the Rust extension)
    - Optional: COLMAP in PATH for real 3D reconstruction
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional rich support — use it when available, fall back to plain print
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    _console = Console()

    def _print(msg: str = "", markup: bool = True) -> None:
        _console.print(msg, markup=markup)

    def _print_rule(title: str = "") -> None:
        _console.rule(title)

    def _ok(msg: str) -> None:
        _console.print(f"  [bold green][OK][/bold green]   {msg}")

    def _warn(msg: str) -> None:
        _console.print(f"  [bold yellow][WARN][/bold yellow]  {msg}")

    def _fail(msg: str) -> None:
        _console.print(f"  [bold red][FAIL][/bold red]  {msg}")

    def _step(n: int, title: str) -> None:
        _console.print(f"\n[bold cyan]Step {n}: {title}[/bold cyan]")
        _console.print("  " + "-" * (len(title) + 8))

    def _rich_table(columns: list[str], rows: list[list[str]], title: str = "") -> None:
        table = Table(title=title, box=rich_box.SIMPLE_HEAVY, show_header=True,
                      header_style="bold magenta")
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        _console.print(table)

    _HAS_RICH = True

except ImportError:
    _HAS_RICH = False

    def _print(msg: str = "", markup: bool = True) -> None:  # type: ignore[misc]
        # Strip rich markup tags from the message
        import re
        clean = re.sub(r"\[/?[a-zA-Z_ ]+\]", "", msg)
        print(clean)

    def _print_rule(title: str = "") -> None:  # type: ignore[misc]
        print(f"\n{'=' * 60}")
        if title:
            print(f"  {title}")
        print('=' * 60)

    def _ok(msg: str) -> None:  # type: ignore[misc]
        print(f"  [OK]   {msg}")

    def _warn(msg: str) -> None:  # type: ignore[misc]
        print(f"  [WARN]  {msg}")

    def _fail(msg: str) -> None:  # type: ignore[misc]
        print(f"  [FAIL]  {msg}")

    def _step(n: int, title: str) -> None:  # type: ignore[misc]
        print(f"\nStep {n}: {title}")
        print("  " + "-" * (len(title) + 8))

    def _rich_table(columns: list[str], rows: list[list[str]], title: str = "") -> None:  # type: ignore[misc]
        if title:
            print(f"\n  {title}")
        col_widths = [max(len(c), max((len(r[i]) for r in rows), default=0))
                      for i, c in enumerate(columns)]
        header = "  " + "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(columns))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in rows:
            print("  " + "  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(columns))))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_DIR = REPO_ROOT / "assets" / "sample-scenes" / "crime-scene-01"
IMAGES_DIR = SCENE_DIR / "images"
WORKSPACE_DIR = SCENE_DIR / "workspace"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _banner() -> None:
    _print()
    if _HAS_RICH:
        _console.print("[bold white]ForenSim Demo[/bold white]")
        _console.print("[cyan]" + "=" * 60 + "[/cyan]")
        _console.print("[white]Forensic Scene Reconstruction & Probabilistic Event Analysis[/white]")
        _console.print("[cyan]" + "=" * 60 + "[/cyan]")
    else:
        print("ForenSim Demo")
        print("=" * 60)
        print("Forensic Scene Reconstruction & Probabilistic Event Analysis")
        print("=" * 60)
    _print()


# ---------------------------------------------------------------------------
# Step 0 — Dependency check
# ---------------------------------------------------------------------------

def _check_dependencies() -> bool:
    """Return True if forensim + _core are both importable."""
    _print_rule("Dependency Check")
    all_ok = True

    # forensim package
    try:
        import forensim  # noqa: F401
        _ok(f"forensim package  (version {forensim.__version__})")
    except ImportError as exc:
        _fail(f"forensim package not importable: {exc}")
        _fail("  → Run: uv pip install -e \".[dev]\" from the repo root")
        all_ok = False

    # _core Rust extension
    try:
        import forensim._core as core  # type: ignore[import-untyped]
        _ = core.MarkovChain  # sanity-check symbol exists
        _ok("forensim._core Rust extension (MarkovChain, HMM, BayesianUpdater)")
    except ImportError as exc:
        _warn(f"forensim._core not built: {exc}")
        _warn("  → Run: maturin develop --release")
        # Warn only — can still demo the Python-side logic
        all_ok = False

    # numpy / pillow
    try:
        import numpy as np  # noqa: F401
        _ok(f"numpy {np.__version__}")
    except ImportError:
        _fail("numpy not installed")
        all_ok = False

    try:
        from PIL import Image  # noqa: F401
        import PIL
        _ok(f"Pillow {PIL.__version__}")
    except ImportError:
        _fail("Pillow not installed")
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Ensure sample dataset exists
# ---------------------------------------------------------------------------

def _ensure_dataset() -> bool:
    """Return True if dataset is ready (generating it if needed)."""
    _print_rule("Sample Dataset")

    if IMAGES_DIR.exists() and len(list(IMAGES_DIR.glob("img_*.jpg"))) == 8:
        _ok(f"Sample dataset found at {SCENE_DIR.relative_to(REPO_ROOT)}")
        return True

    _warn("Sample dataset not found — generating it now …")
    gen_script = REPO_ROOT / "scripts" / "generate_sample_dataset.py"
    if not gen_script.exists():
        _fail(f"Generator script not found: {gen_script}")
        return False

    result = subprocess.run(
        [sys.executable, str(gen_script)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _fail("Dataset generation failed:")
        _fail(result.stderr)
        return False

    _ok("Dataset generated successfully")
    return True


# ---------------------------------------------------------------------------
# Step 1 — Evidence loading
# ---------------------------------------------------------------------------

def step1_load_evidence() -> tuple[list[Path], list]:
    """Load images and annotations from the sample scene."""
    _step(1, "Evidence Loading")

    images = sorted(IMAGES_DIR.glob("img_*.jpg"))
    _print(f"  Found [bold]{len(images)}[/bold] scene images in "
           f"[dim]{IMAGES_DIR.relative_to(REPO_ROOT)}[/dim]:")
    for img_path in images:
        size_kb = img_path.stat().st_size // 1024
        _print(f"    • {img_path.name}  ({size_kb} KB)")

    # Load annotations
    ann_path = SCENE_DIR / "annotations.json"
    annotations: list = []
    if ann_path.exists():
        try:
            from forensim.annotate.manager import Annotation
            raw = json.loads(ann_path.read_text(encoding="utf-8"))
            annotations = [Annotation.from_dict(a) for a in raw.get("annotations", [])]
            _ok(f"Loaded {len(annotations)} annotations from annotations.json")
            for ann in annotations:
                _print(f"    • [{ann.id}] {ann.tag}  ({ann.description[:55]})")
        except Exception as exc:  # noqa: BLE001
            _warn(f"Could not load annotations via forensim: {exc}")
            raw = json.loads(ann_path.read_text(encoding="utf-8"))
            annotations = raw.get("annotations", [])
            _ok(f"Loaded {len(annotations)} annotations (raw JSON)")
    else:
        _warn("annotations.json not found — skipping annotation load")

    return images, annotations


# ---------------------------------------------------------------------------
# Step 2 — Probabilistic inference
# ---------------------------------------------------------------------------

def step2_probabilistic_inference(
    annotations: list,
) -> list:
    """Run Markov chain ranking and return scored hypotheses."""
    _step(2, "Probabilistic Inference")

    # ── Hypotheses ────────────────────────────────────────────────────────
    sequences = [
        ["entry_east", "shot_fired", "casing_ejected", "exit_north"],
        ["entry_east", "shot_fired", "return_fire", "casing_ejected", "impact"],
        ["approach", "shot_fired", "casing_ejected", "escape"],
    ]
    descriptions = [
        "H1: Single shooter, entered east",
        "H2: Exchange of fire",
        "H3: Drive-by trajectory",
    ]

    event_vocab = [
        "entry_east",
        "shot_fired",
        "casing_ejected",
        "exit_north",
        "return_fire",
        "impact",
        "approach",
        "escape",
    ]
    n = len(event_vocab)

    # ── Transition matrix ─────────────────────────────────────────────────
    # Slightly favours H1 (single-shooter) by making exit_north more likely
    # after casing_ejected, and making return_fire less likely overall.
    T = [
        # entry_east  shot_fired  casing_ej  exit_north  ret_fire  impact  approach  escape
        [0.05,        0.60,       0.05,       0.10,       0.05,    0.05,   0.05,     0.05],  # entry_east
        [0.05,        0.05,       0.50,       0.05,       0.15,    0.10,   0.05,     0.05],  # shot_fired
        [0.05,        0.05,       0.05,       0.50,       0.05,    0.05,   0.05,     0.20],  # casing_ej
        [0.10,        0.05,       0.05,       0.05,       0.05,    0.05,   0.55,     0.10],  # exit_north
        [0.05,        0.30,       0.40,       0.05,       0.05,    0.10,   0.05,     0.00],  # return_fire
        [0.05,        0.05,       0.05,       0.05,       0.05,    0.10,   0.05,     0.60],  # impact
        [0.05,        0.55,       0.05,       0.05,       0.05,    0.05,   0.05,     0.15],  # approach
        [0.10,        0.05,       0.05,       0.10,       0.05,    0.05,   0.50,     0.10],  # escape
    ]
    # Row-normalise to guarantee proper stochastic matrix
    T_norm = []
    for row in T:
        s = sum(row)
        T_norm.append([v / s for v in row])

    initial_probs = [1.0 / n] * n

    try:
        from forensim.infer.sequence import rank_event_sequences, integrate_annotation_scores
        from forensim.annotate.manager import Annotation

        hypotheses = rank_event_sequences(
            sequences, descriptions, T_norm, initial_probs, event_vocab
        )
        _ok(f"MarkovChain scored {len(hypotheses)} hypotheses")

        # Integrate annotations
        if annotations:
            ann_objects: list[Annotation] = []
            if annotations and hasattr(annotations[0], "tag"):
                # Already Annotation dataclass objects
                ann_objects = annotations  # type: ignore[assignment]
            else:
                # Raw dicts from JSON fallback
                ann_objects = [Annotation.from_dict(a) for a in annotations]

            hypotheses = integrate_annotation_scores(hypotheses, ann_objects, strength=1.0)
            _ok(f"Integrated {len(ann_objects)} annotation scores (strength=1.0)")

    except ImportError as exc:
        _warn(f"forensim._core unavailable — using pure-Python fallback: {exc}")
        # Pure-Python fallback: compute log-prob by summing log transition probabilities
        from forensim.infer.sequence import ScoredHypothesis
        import math as _math_local

        vocab_index = {name: i for i, name in enumerate(event_vocab)}
        scored: list[ScoredHypothesis] = []
        for idx, (seq, desc) in enumerate(zip(sequences, descriptions)):
            lp = 0.0
            for s_idx in range(len(seq) - 1):
                i = vocab_index.get(seq[s_idx], 0)
                j = vocab_index.get(seq[s_idx + 1], 0)
                p = T_norm[i][j]
                lp += _math_local.log(max(p, 1e-12))
            scored.append(ScoredHypothesis(
                index=idx, description=desc, log_probability=lp,
                posterior=float("nan"), events=seq,
            ))

        # Normalise
        max_lp = max(h.log_probability for h in scored)
        weights = [_math_local.exp(h.log_probability - max_lp) for h in scored]
        total = sum(weights)
        for h, w in zip(scored, weights):
            h.posterior = w / total
        scored.sort(key=lambda h: h.posterior, reverse=True)
        for rank, h in enumerate(scored):
            h.bayes_factor = h.posterior / scored[0].posterior if scored[0].posterior > 0 else 1.0
        hypotheses = scored

    # Print ranked table
    rows = []
    for rank, h in enumerate(hypotheses, 1):
        posterior_pct = f"{h.posterior * 100:.1f}%" if not math.isnan(h.posterior) else "N/A"
        bf = f"{h.bayes_factor:.3f}" if not math.isnan(h.bayes_factor) else "1.000"
        rows.append([str(rank), h.description, posterior_pct, bf])

    _rich_table(
        columns=["Rank", "Hypothesis", "Posterior", "Bayes Factor"],
        rows=rows,
        title="Ranked Hypotheses",
    )

    return hypotheses


# ---------------------------------------------------------------------------
# Step 3 — Sensitivity analysis
# ---------------------------------------------------------------------------

def step3_sensitivity(hypotheses: list) -> None:
    """Run leave-one-out sensitivity analysis."""
    _step(3, "Evidence Sensitivity Analysis")

    # Three evidence sources matching the annotation tags
    evidence_defs = [
        {
            "name": "blood_spatter",
            # Per-hypothesis log-likelihood delta: H1 benefits most (spatter on east wall
            # supports single-shooter entry from east), H2 less, H3 least.
            "deltas": [0.80, 0.45, 0.20],
        },
        {
            "name": "shell_casing",
            # Casing on floor — consistent with all hypotheses but most with H1/H2.
            "deltas": [0.60, 0.55, 0.40],
        },
        {
            "name": "impact_mark",
            # Impact on north wall — mostly consistent with H2 (exchange of fire) and H3.
            "deltas": [0.30, 0.70, 0.55],
        },
    ]

    # Re-order deltas to match hypothesis ordering returned from step 2
    # (hypotheses are sorted by posterior, so we must map them correctly)
    # Build description → index mapping in the original sequence order
    h1_idx, h2_idx, h3_idx = None, None, None
    for i, h in enumerate(hypotheses):
        if "H1" in h.description:
            h1_idx = i
        elif "H2" in h.description:
            h2_idx = i
        elif "H3" in h.description:
            h3_idx = i

    order = [h1_idx, h2_idx, h3_idx]

    try:
        from forensim.infer.sensitivity import EvidenceSource, compute_sensitivity

        evidence_sources = []
        for ev in evidence_defs:
            # Reorder deltas to match sorted hypothesis order
            deltas_ordered = [ev["deltas"][order.index(i)] if i in order else 0.0
                              for i in range(len(hypotheses))]
            evidence_sources.append(
                EvidenceSource(
                    name=ev["name"],
                    log_likelihood_delta=deltas_ordered,
                    weight=1.0,
                )
            )

        results = compute_sensitivity(hypotheses, evidence_sources)
        _ok(f"Sensitivity computed for {len(results)} evidence sources")

        rows = []
        for r in results:
            rows.append([
                r.evidence_name,
                f"{r.impact_pct:+.1f}%",
                str(r.rank_change),
                f"{r.baseline_top_posterior * 100:.1f}%",
                f"{r.loo_top_posterior * 100:.1f}%",
            ])

        _rich_table(
            columns=["Evidence", "Impact %", "Rank Chg", "Baseline P(H*)", "LOO P(H*)"],
            rows=rows,
            title="Leave-One-Out Sensitivity",
        )

    except ImportError as exc:
        _warn(f"Sensitivity module unavailable: {exc}")


# ---------------------------------------------------------------------------
# Step 4 — Export summary
# ---------------------------------------------------------------------------

def step4_export_summary(hypotheses: list) -> None:
    """Print would-be export paths (no actual file generation)."""
    _step(4, "Export Summary")

    workspace = WORKSPACE_DIR
    report_pdf = workspace / "forensim_report.pdf"
    usd_export = workspace / "scene_reconstruction.usdc"
    video_path = workspace / "scene_walkthrough.mp4"

    # Show relative paths to keep output compact
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    _print("  [dim](Export paths - no files generated in demo mode)[/dim]")
    _print()
    _print(f"  [bold]PDF Report:[/bold]        {_rel(report_pdf)}")
    _print(f"  [bold]USD Scene:[/bold]         {_rel(usd_export)}")
    _print(f"  [bold]Video Walkthrough:[/bold]  {_rel(video_path)}")
    _print()
    _ok("In full mode, run via the API:")
    _print("     POST /export/report?format=pdf&workspace=<path>")
    _print("     POST /export/usd?workspace=<path>")


# ---------------------------------------------------------------------------
# Completion banner
# ---------------------------------------------------------------------------

def _completion_banner(hypotheses: list) -> None:
    import math as _math

    top = hypotheses[0] if hypotheses else None
    top_label = top.description if top else "unknown"
    top_pct = f"{top.posterior * 100:.1f}%" if top and not _math.isnan(top.posterior) else "N/A"

    # Shorten H1 label to just "H1"
    short_label = top_label.split(":")[0].strip() if top else "H?"

    _print()
    if _HAS_RICH:
        _console.print("[bold green]" + "=" * 60 + "[/bold green]")
        _console.print("[bold white]Demo complete![/bold white]")
        _console.print(
            f"[green]Results: {short_label} ranked #1 with posterior {top_pct}[/green]"
        )
        _console.print(
            "[dim]Run 'npm run tauri dev' from the app/ directory to explore interactively.[/dim]"
        )
        _console.print("[bold green]" + "=" * 60 + "[/bold green]")
    else:
        print("=" * 60)
        print("Demo complete!")
        print(f"Results: {short_label} ranked #1 with posterior {top_pct}")
        print("Run 'npm run tauri dev' from the app/ directory to explore interactively.")
        print("=" * 60)
    _print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="ForenSim end-to-end pipeline demo")
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Skip the 3D reconstruction step (COLMAP not required)",
    )
    args = parser.parse_args()

    _banner()

    # Step 0 — deps
    deps_ok = _check_dependencies()
    if not deps_ok:
        _warn("Some dependencies missing — continuing with best-effort fallbacks …")

    # Ensure dataset
    if not _ensure_dataset():
        _fail("Cannot continue without the sample dataset.")
        return 1

    # Step 1 — load evidence
    images, annotations = step1_load_evidence()

    # Step 2 — inference
    hypotheses = step2_probabilistic_inference(annotations)

    # Step 3 — sensitivity
    step3_sensitivity(hypotheses)

    # Step 4 — export
    step4_export_summary(hypotheses)

    # Completion
    _completion_banner(hypotheses)

    return 0


if __name__ == "__main__":
    sys.exit(main())

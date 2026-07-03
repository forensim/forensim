"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Return a fresh temporary workspace directory."""
    return tmp_path

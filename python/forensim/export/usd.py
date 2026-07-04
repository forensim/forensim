"""Package a USD scene and its referenced assets into a zip archive."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def package_usd_scene(usd_path: Path, output_path: Path) -> Path:
    """Create a zip archive containing the USD scene and any PLY siblings.

    Args:
        usd_path: Path to the USD scene file.
        output_path: Path for the generated zip file.

    Returns:
        Path to the generated zip file.
    """
    usd_path = Path(usd_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not usd_path.exists():
        raise FileNotFoundError(f"USD scene not found: {usd_path}")

    files_to_archive = [usd_path]
    # Include common sibling assets if they exist
    for ext in (".ply", ".png", ".jpg", ".jpeg", ".usda", ".usdc"):
        for sibling in usd_path.parent.glob(f"*{ext}"):
            if sibling not in files_to_archive:
                files_to_archive.append(sibling)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files_to_archive:
            arcname = file.name
            zf.write(file, arcname)
            logger.info("Archived %s as %s", file, arcname)

    logger.info("USD scene packaged: %s", output_path)
    return output_path

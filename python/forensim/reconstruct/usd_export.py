"""
USD export utilities.

Converts Gaussian Splat `.ply` files and polygon meshes to OpenUSD format
for import into NVIDIA Omniverse.

Requires: pip install omniverse-gsplat-converter[usd]
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ply_to_usd(
    ply_path: Path,
    output_path: Path,
    up_axis: str = "Y",
    validate: bool = True,
) -> Path:
    """
    Convert a 3DGS .ply file to a USDZ/USDA Gaussian Splat scene.

    Uses ``omniverse-gsplat-converter`` which writes the
    ``ParticleField3DGaussianSplat`` USD schema understood by Omniverse RTX.

    Args:
        ply_path: Input .ply Gaussian Splat file.
        output_path: Output path (.usdz or .usda).
        up_axis: Scene up-axis ('Y' or 'Z').
        validate: If True, raise an error if the output file is empty or missing.

    Returns:
        Path to the written USD file.
    """
    try:
        from omniverse_gsplat_converter import (  # type: ignore[import-untyped]
            read_ply,
            write_gaussian_splat_usd,
        )
    except ImportError as exc:
        raise ImportError(
            "omniverse-gsplat-converter not installed. "
            "Run: uv pip install 'omniverse-gsplat-converter[usd]'"
        ) from exc

    if not ply_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {ply_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Converting %s to USD %s", ply_path, output_path)
    splat_data = read_ply(str(ply_path))
    write_gaussian_splat_usd(
        splat_data,
        str(output_path),
        source_file=str(ply_path),
        up_axis=up_axis,
    )

    if validate:
        if not output_path.exists():
            raise RuntimeError(f"USD export did not create {output_path}")
        if output_path.stat().st_size == 0:
            raise RuntimeError(f"USD export created an empty file: {output_path}")

    logger.info("USD export complete: %s", output_path)
    return output_path


def mesh_to_usd(mesh_path: Path, output_path: Path) -> Path:
    """
    Convert an OBJ/PLY/FBX/glTF mesh to USD using the pxr library.

    For simple meshes. For complex conversions with materials,
    the Omniverse Asset Converter (``omni.kit.asset_converter``) is preferred.

    Args:
        mesh_path: Input mesh file (.obj, .ply, .glb, .gltf, .fbx).
        output_path: Output .usda path.

    Returns:
        Path to the written USD file.
    """
    try:
        import trimesh
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("trimesh and usd-core are required") from exc

    mesh = trimesh.load(str(mesh_path), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), "mesh_path must load a Trimesh object"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    xform = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(xform.GetPrim())

    geo = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/Mesh"))
    verts = [Gf.Vec3f(*v) for v in mesh.vertices.tolist()]
    faces = mesh.faces.flatten().tolist()
    face_counts = [3] * len(mesh.faces)

    geo.GetPointsAttr().Set(Vt.Vec3fArray(verts))
    geo.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces))
    geo.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))

    stage.Save()
    return output_path

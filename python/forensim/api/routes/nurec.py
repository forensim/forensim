"""NuRec gRPC proxy routes."""

from __future__ import annotations

import base64
import io
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────


class NuRecHealthResponse(BaseModel):
    status: str
    address: str
    reachable: bool


class CameraPoseModel(BaseModel):
    position: list[float]
    quaternion: list[float]


class SceneInfoModel(BaseModel):
    name: str
    id: str
    asset_path: str
    description: str = ""


class ListScenesResponse(BaseModel):
    status: str
    scenes: list[SceneInfoModel]


class LoadSceneRequest(BaseModel):
    scene_id: str
    address: str = "localhost:8080"


class LoadSceneResponse(BaseModel):
    status: str
    loaded: bool
    scene_id: str


class RenderRequest(BaseModel):
    scene_id: str
    pose: CameraPoseModel
    width: int = 1920
    height: int = 1080
    address: str = "localhost:8080"


class RenderResponse(BaseModel):
    status: str
    width: int
    height: int
    image_base64: str


# ── NuRec imports (graceful when grpcio is absent) ───────────────────────────

try:
    from forensim.reconstruct.nurec import CameraPose, NuRecClient, SceneInfo
except ImportError:
    CameraPose = None  # type: ignore[assignment, misc]
    NuRecClient = None  # type: ignore[assignment, misc]
    SceneInfo = None  # type: ignore[assignment, misc]


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/health", response_model=NuRecHealthResponse)
async def health(address: str = "localhost:8080") -> NuRecHealthResponse:
    """Check whether the NuRec gRPC server is reachable."""
    if NuRecClient is None:
        return NuRecHealthResponse(
            status="unavailable",
            address=address,
            reachable=False,
        )
    try:
        client = NuRecClient(address)
        reachable = client.health_check()
        return NuRecHealthResponse(
            status="ok" if reachable else "unreachable",
            address=address,
            reachable=reachable,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/scenes", response_model=ListScenesResponse)
async def list_scenes(address: str = "localhost:8080") -> ListScenesResponse:
    """List scenes available on the NuRec server."""
    if NuRecClient is None or SceneInfo is None:
        return ListScenesResponse(status="unavailable", scenes=[])
    try:
        client = NuRecClient(address)
        scenes = client.list_scenes()
        return ListScenesResponse(
            status="ok",
            scenes=[
                SceneInfoModel(
                    name=s.name,
                    id=s.id,
                    asset_path=s.asset_path,
                    description=s.description,
                )
                for s in scenes
            ],
        )
    except ImportError:
        return ListScenesResponse(status="unavailable", scenes=[])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/load", response_model=LoadSceneResponse)
async def load_scene(req: LoadSceneRequest) -> LoadSceneResponse:
    """Ask the NuRec server to load a scene."""
    if NuRecClient is None:
        return LoadSceneResponse(
            status="unavailable",
            loaded=False,
            scene_id=req.scene_id,
        )
    try:
        client = NuRecClient(req.address)
        loaded = client.load_scene(req.scene_id)
        return LoadSceneResponse(
            status="ok" if loaded else "failed",
            loaded=loaded,
            scene_id=req.scene_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/render", response_model=RenderResponse)
async def render(req: RenderRequest) -> RenderResponse:
    """Render an RGB image from a camera pose via NuRec."""
    if NuRecClient is None or CameraPose is None:
        raise HTTPException(
            status_code=503,
            detail="NuRec gRPC client is unavailable (grpcio not installed?)",
        )
    try:
        client = NuRecClient(req.address)
        image = client.render_rgb(
            pose=CameraPose(
                position=tuple(req.pose.position),  # type: ignore[arg-type]
                quaternion=tuple(req.pose.quaternion),  # type: ignore[arg-type]
            ),
            width=req.width,
            height=req.height,
        )

        try:
            from PIL import Image as PILImage

            rgb = PILImage.frombytes(
                mode="RGB",
                data=image.data,
                size=(image.width, image.height),
            )
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except ImportError:
            # PIL not available; fall back to base64-encoding the raw bytes.
            image_base64 = base64.b64encode(image.data).decode("ascii")

        return RenderResponse(
            status="ok",
            width=image.width,
            height=image.height,
            image_base64=image_base64,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Re-export the model classes so they can be imported from the route module.
__all__: list[Any] = [
    "router",
    "NuRecHealthResponse",
    "CameraPoseModel",
    "SceneInfoModel",
    "ListScenesResponse",
    "LoadSceneRequest",
    "LoadSceneResponse",
    "RenderRequest",
    "RenderResponse",
]

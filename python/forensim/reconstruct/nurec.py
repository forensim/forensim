"""
NVIDIA NuRec gRPC client.

NuRec runs as a Docker container exposing a gRPC server on port 8080.
Launch with:
    docker run --gpus all -p 8080:8080 nvcr.io/nvidia/nre-ga:latest

Requires: pip install grpcio grpcio-tools
Docs: https://docs.nvidia.com/nurec/api/grpc_api_guide.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    """6-DOF camera pose."""

    position: tuple[float, float, float]      # (x, y, z) in metres
    quaternion: tuple[float, float, float, float]  # (qw, qx, qy, qz)


@dataclass
class RGBImage:
    """Rendered RGB image buffer."""

    width: int
    height: int
    data: bytes  # raw RGB bytes


@dataclass
class SceneInfo:
    """Metadata for a scene available on the NuRec server."""

    name: str
    id: str
    asset_path: str  # path to USD/PLY on the server
    description: str = ""


@dataclass
class NuRecRenderResult:
    """A single frame rendered by NuRec plus the camera pose that produced it."""

    image: RGBImage
    pose: CameraPose
    scene_id: str
    frame_index: int = 0


class NuRecClient:
    """
    Thin Python client for the NuRec gRPC API.

    Example:
        client = NuRecClient("localhost:8080")
        image = client.render_rgb(pose, width=1920, height=1080)
    """

    def __init__(self, address: str = "localhost:8080") -> None:
        self.address = address
        self._stub: Any | None = None
        self._scene_stub: Any | None = None

    def _get_stub(self) -> Any:
        """Lazy-load the NuRec gRPC service stub."""
        if self._stub is None:
            try:
                import grpc
                from nre.grpc.protos.sensorsim_pb2_grpc import (  # type: ignore[import-not-found]
                    SensorsimServiceStub,
                )
            except ImportError as e:
                raise ImportError(
                    "grpcio and NuRec protos required. "
                    "Run: pip install grpcio and ensure NuRec container is running."
                ) from e
            channel = grpc.insecure_channel(self.address)
            self._stub = SensorsimServiceStub(channel)
        return self._stub

    def _get_scene_stub(self) -> Any:
        """Lazy-load the NuRec scene-management gRPC stub."""
        if self._scene_stub is None:
            try:
                import grpc
                from nre.grpc.protos.sensorsim_pb2_grpc import (
                    SceneServiceStub,
                )
            except ImportError as e:
                raise ImportError(
                    "grpcio and NuRec protos required. "
                    "Run: pip install grpcio and ensure NuRec container is running."
                ) from e
            channel = grpc.insecure_channel(self.address)
            self._scene_stub = SceneServiceStub(channel)
        return self._scene_stub

    def health_check(self) -> bool:
        """Return True if the NuRec server is reachable."""
        try:
            import grpc

            channel = grpc.insecure_channel(self.address)
            grpc.channel_ready_future(channel).result(timeout=3)
            return True
        except Exception:
            return False

    def render_rgb(
        self,
        pose: CameraPose,
        width: int = 1920,
        height: int = 1080,
    ) -> RGBImage:
        """Render an RGB image from the given camera pose."""
        try:
            from nre.grpc.protos.sensorsim_pb2 import (  # type: ignore[import-not-found]
                ImageFormat,
                PosePair,
                RGBRenderRequest,
            )
        except ImportError as e:
            raise ImportError(
                "grpcio and NuRec protos required. "
                "Run: pip install grpcio and ensure NuRec container is running."
            ) from e

        stub = self._get_stub()
        request = RGBRenderRequest(
            pose=PosePair(
                position=list(pose.position),
                quaternion=list(pose.quaternion),
            ),
            width=width,
            height=height,
            format=ImageFormat.RGB,
        )
        try:
            response = stub.render_rgb(request)
        except Exception as exc:
            logger.warning("NuRec render_rgb failed: %s", exc)
            raise
        return RGBImage(width=width, height=height, data=response.image_data)

    def list_scenes(self) -> list[SceneInfo]:
        """List scenes available on the NuRec server."""
        try:
            import grpc
            from google.protobuf.empty_pb2 import Empty
        except ImportError as e:
            raise ImportError(
                "grpcio and NuRec protos required. "
                "Run: pip install grpcio and ensure NuRec container is running."
            ) from e

        stub = self._get_scene_stub()
        try:
            response = stub.ListScenes(Empty())
        except grpc.RpcError as exc:
            logger.warning("NuRec ListScenes failed: %s", exc)
            return []

        scenes: list[SceneInfo] = []
        for scene in response.scenes:
            scenes.append(
                SceneInfo(
                    name=scene.name,
                    id=scene.id,
                    asset_path=scene.asset_path,
                    description=getattr(scene, "description", ""),
                )
            )
        return scenes

    def load_scene(self, scene_id: str) -> bool:
        """Ask the NuRec server to load the specified scene."""
        try:
            import grpc
            from nre.grpc.protos.sensorsim_pb2 import (
                LoadSceneRequest,
            )
        except ImportError as e:
            raise ImportError(
                "grpcio and NuRec protos required. "
                "Run: pip install grpcio and ensure NuRec container is running."
            ) from e

        stub = self._get_scene_stub()
        request = LoadSceneRequest(scene_id=scene_id)
        try:
            stub.LoadScene(request)
            return True
        except grpc.RpcError as exc:
            logger.warning("NuRec LoadScene failed for scene_id=%s: %s", scene_id, exc)
            return False

    def render_flythrough(
        self,
        scene_id: str,
        poses: list[CameraPose],
        width: int = 1920,
        height: int = 1080,
    ) -> list[NuRecRenderResult]:
        """Render a sequence of camera poses as a flythrough."""
        frames: list[NuRecRenderResult] = []
        for frame_index, pose in enumerate(poses):
            image = self.render_rgb(pose, width=width, height=height)
            frames.append(
                NuRecRenderResult(
                    image=image,
                    pose=pose,
                    scene_id=scene_id,
                    frame_index=frame_index,
                )
            )
        return frames

    def render_flythrough_to_frames(
        self,
        scene_id: str,
        poses: list[CameraPose],
        width: int = 1280,
        height: int = 720,
    ) -> list[bytes]:
        """Render a flythrough and return only the raw image bytes per frame."""
        return [
            result.image.data
            for result in self.render_flythrough(
                scene_id=scene_id,
                poses=poses,
                width=width,
                height=height,
            )
        ]

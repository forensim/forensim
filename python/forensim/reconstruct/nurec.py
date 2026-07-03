"""
NVIDIA NuRec gRPC client.

NuRec runs as a Docker container exposing a gRPC server on port 8080.
Launch with:
    docker run --gpus all -p 8080:8080 nvidia/nre-ga

Requires: pip install grpcio grpcio-tools
Docs: https://docs.nvidia.com/nurec/api/grpc_api_guide.html
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraPose:
    """6-DOF camera pose."""
    position: tuple[float, float, float]      # (x, y, z) in metres
    quaternion: tuple[float, float, float, float]  # (qw, qx, qy, qz)


@dataclass
class RGBImage:
    width: int
    height: int
    data: bytes  # raw RGB bytes


class NuRecClient:
    """
    Thin Python client for the NuRec gRPC API.

    Example:
        client = NuRecClient("localhost:8080")
        image = client.render_rgb(pose, width=1920, height=1080)
    """

    def __init__(self, address: str = "localhost:8080") -> None:
        self.address = address
        self._stub: object | None = None

    def _get_stub(self) -> object:
        if self._stub is None:
            try:
                import grpc  # type: ignore[import-untyped]
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

    def render_rgb(
        self,
        pose: CameraPose,
        width: int = 1920,
        height: int = 1080,
    ) -> RGBImage:
        """Render an RGB image from the given camera pose."""
        from nre.grpc.protos.sensorsim_pb2 import (  # type: ignore[import-not-found]
            ImageFormat,
            PosePair,
            RGBRenderRequest,
        )
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
        response = stub.render_rgb(request)  # type: ignore[attr-defined]
        return RGBImage(width=width, height=height, data=response.image_data)

    def health_check(self) -> bool:
        """Return True if the NuRec server is reachable."""
        try:
            import grpc
            channel = grpc.insecure_channel(self.address)
            grpc.channel_ready_future(channel).result(timeout=3)
            return True
        except Exception:
            return False

"""
ForenSim FastAPI sidecar.

Exposes the reconstruction and inference pipeline over HTTP so the
Tauri/Rust desktop app can call it via REST.

Run directly:
    uvicorn forensim.api.main:app --host 127.0.0.1 --port 8008

Or via the Tauri sidecar bundler (see app/src-tauri/src/sidecar.rs).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import annotate, export, infer, reconstruct, simulate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    print("[forensim-api] started on http://127.0.0.1:8008")
    yield
    # Shutdown
    print("[forensim-api] shutting down")


app = FastAPI(
    title="ForenSim API",
    description="Forensic scene reconstruction and probabilistic event analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow Tauri WebView (tauri://localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://localhost:1420", "http://127.0.0.1:1420"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reconstruct.router, prefix="/api/reconstruct", tags=["reconstruct"])
app.include_router(simulate.router, prefix="/api/simulate", tags=["simulate"])
app.include_router(infer.router, prefix="/api/infer", tags=["infer"])
app.include_router(annotate.router, prefix="/api/annotate", tags=["annotate"])
app.include_router(export.router, prefix="/api/export", tags=["export"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)

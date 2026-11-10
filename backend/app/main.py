"""API TouchFree : detection de gestes de la main a partir d'une image.

Lancement local :
    uvicorn app.main:app --reload            (depuis le dossier backend)
Documentation interactive (Swagger) : http://localhost:8000/docs
"""

import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from app.config import get_settings, load_metadata
from app.schemas import (
    Detection,
    HealthResponse,
    MetadataResponse,
    PredictResponse,
)

logger = logging.getLogger("touchfree")

REQUESTS = Counter("touchfree_predict_requests_total", "Requetes /predict", ["status"])
LATENCY = Histogram(
    "touchfree_inference_seconds",
    "Temps d'inference (pre-traitement + modele + NMS)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
)
GESTURES = Counter("touchfree_gestures_total", "Gestes detectes", ["label"])


def load_detector(settings, metadata):
    """Charge le modele ONNX s'il existe, sinon l'API demarre sans modele."""
    if not settings.model_path.exists():
        logger.warning("Modele introuvable : %s", settings.model_path)
        return None
    from app.inference import OnnxDetector

    return OnnxDetector(
        settings.model_path,
        metadata["names"],
        imgsz=int(metadata.get("imgsz", 512)),
        conf=settings.conf_threshold,
        iou=settings.iou_threshold,
    )


def create_app(detector=None):
    """Fabrique l'application. Les tests passent un faux detecteur."""
    settings = get_settings()
    metadata = load_metadata(settings.metadata_path)

    @asynccontextmanager
    async def lifespan(app):
        app.state.detector = detector or load_detector(settings, metadata)
        yield

    app = FastAPI(
        title="TouchFree API",
        description="Detection de 18 gestes de la main + no_gesture (YOLO11, ONNX).",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.metadata = metadata

    def require_detector(request: Request):
        det = request.app.state.detector
        if det is None:
            raise HTTPException(503, "Modele non charge : lancez `dvc pull models`.")
        return det

    @app.get("/health", response_model=HealthResponse, tags=["systeme"])
    async def health(request: Request):
        loaded = request.app.state.detector is not None
        return HealthResponse(status="ok", model_loaded=loaded)

    @app.get("/model/metadata", response_model=MetadataResponse, tags=["modele"])
    async def model_metadata(request: Request):
        require_detector(request)
        meta = request.app.state.metadata
        return MetadataResponse(
            names=meta["names"],
            imgsz=int(meta.get("imgsz", 512)),
            variant=str(meta.get("variant", "inconnu")),
            version=str(meta.get("version", "local")),
            metrics=meta.get("metrics", {}),
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["modele"])
    async def predict(request: Request, file: UploadFile = File(...)):
        det = require_detector(request)
        if file.content_type and not file.content_type.startswith("image/"):
            REQUESTS.labels(status="400").inc()
            raise HTTPException(400, "Le fichier doit etre une image (JPEG ou PNG).")
        raw = await file.read()
        if len(raw) > settings.max_upload_mb * 1024 * 1024:
            REQUESTS.labels(status="413").inc()
            raise HTTPException(413, "Image trop lourde.")
        try:
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except (UnidentifiedImageError, OSError):
            REQUESTS.labels(status="400").inc()
            raise HTTPException(400, "Image illisible.")

        detections, elapsed_ms = await run_in_threadpool(det.predict, image)
        LATENCY.observe(elapsed_ms / 1000)
        REQUESTS.labels(status="200").inc()
        for d in detections:
            GESTURES.labels(label=d["label"]).inc()
        return PredictResponse(
            detections=[Detection(**d) for d in detections],
            inference_ms=round(elapsed_ms, 2),
            image_width=image.shape[1],
            image_height=image.shape[0],
            model_version=str(request.app.state.metadata.get("version", "local")),
        )

    @app.get("/metrics", tags=["systeme"], include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()

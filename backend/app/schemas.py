"""Formats des reponses de l'API (valides et documentes dans Swagger)."""

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(examples=["palm"])
    class_id: int = Field(examples=[8])
    confidence: float = Field(ge=0, le=1, examples=[0.91])
    box: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] en pixels de l'image envoyee",
    )


class PredictResponse(BaseModel):
    detections: list[Detection]
    inference_ms: float = Field(description="Temps pre-traitement + modele + NMS")
    image_width: int
    image_height: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    names: list[str]
    imgsz: int
    variant: str
    version: str
    metrics: dict = Field(default_factory=dict)
    conf_threshold: float
    iou_threshold: float

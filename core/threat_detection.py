"""YOLO evidence for weapons, fire, smoke, and bag candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

from core.logging_utils import get_logger
from core.gpu import cuda_device_name
from core.model_security import download_verified, file_sha256, load_yolo
from core.paths import ROOT

log = get_logger("threats")
FIRE_CATEGORIES = ("fire", "smoke")


@dataclass(frozen=True)
class ThreatModelArtifact:
    repo: str
    filename: str
    revision: str
    sha256: str


AUXILIARY_MODELS = {
    "surveillance": ThreatModelArtifact(
        repo="xDevine/yolo11s-multiclass-surveillance-detection",
        filename="weights/best.pt",
        revision="9a58a5a9da9513c23a315cdd668c0a33093eb177",
        sha256="b95e99b25c954177fbbb97029a57d2c8e1450135873240ea2a39dda7aff4375f",
    ),
    "fire": ThreatModelArtifact(
        repo="SalahALHaismawi/yolov26-fire-detection",
        filename="best.pt",
        revision="a15cd847ddd587e16ca1c77ae21bd6a352c972ef",
        sha256="52684ddffd4797f1bc5ff1bd2b72af3ec34eabf0c1646c37f518bb766b27b79f",
    ),
}


@dataclass
class ThreatDetection:
    source: str
    label: str
    category: str
    confidence: float
    bbox: List[float]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "label": self.label,
            "category": self.category,
            "confidence": round(float(self.confidence), 3),
            "bbox": [round(float(v), 1) for v in self.bbox],
        }


@dataclass
class ThreatFrame:
    detections: List[ThreatDetection] = field(default_factory=list)
    alerts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SurveillanceThreatDetector:
    """Confirm detections across frames using dedicated fire/smoke evidence."""

    WEAPONS = {
        "short-gun",
        "long-gun",
        "gun",
        "pistol",
        "rifle",
        "firearm",
        "knife",
    }
    FIRE = {"fire"}
    SMOKE = {"smoke"}
    BAGS = {"bag", "backpack", "handbag", "suitcase", "purse"}

    def __init__(
        self,
        surveillance_weights: Optional[str] = None,
        threat_weights: Optional[str] = None,
        download_missing: bool = True,
        conf: float = 0.35,
        high_conf: float = 0.85,
        confirm_frames: int = 3,
        imgsz: int = 640,
        device: str = "cuda",
        bag_proximity_px: float = 150.0,
        enable_weapon_alerts: bool = False,
        enable_fire_alerts: bool = True,
        enable_bag_alerts: bool = False,
    ):
        if threat_weights:
            log.warning(
                "event=legacy_threat_weights_ignored use_surveillance_weights=true"
            )
        self.weights = {"surveillance": surveillance_weights}
        if enable_fire_alerts:
            self.weights["fire"] = None
        self.download_missing = bool(download_missing)
        self.conf = float(conf)
        self.high_conf = float(high_conf)
        self.confirm_frames = max(1, int(confirm_frames))
        self.imgsz = int(imgsz)
        self.device = cuda_device_name(device)
        self.bag_proximity_px = float(bag_proximity_px)
        self.enable_weapon_alerts = bool(enable_weapon_alerts)
        self.enable_fire_alerts = bool(enable_fire_alerts)
        self.enable_bag_alerts = bool(enable_bag_alerts)
        self.models: Dict[str, Any] = {}
        self.status: Dict[str, Dict[str, Any]] = {}
        self.streaks: Dict[str, int] = {}
        self._load_attempted = False
        self._fire_source = None

    def reset(self) -> None:
        self.streaks.clear()
        self._load_attempted = False
        self._fire_source = None

    def _resolve_weight(self, source: str) -> Optional[str]:
        artifact = AUXILIARY_MODELS[source]
        configured = self.weights[source]
        if configured:
            path = Path(configured)
            if path.is_file():
                return str(path)
            if not self.download_missing:
                return None
        destination = (
            ROOT
            / "models"
            / source
            / f"{artifact.revision}_{Path(artifact.filename).name}"
        )
        if destination.is_file() and file_sha256(destination) == artifact.sha256:
            return str(destination)
        if not self.download_missing:
            return None
        filename = quote(artifact.filename, safe="/")
        url = f"https://huggingface.co/{artifact.repo}/resolve/{artifact.revision}/{filename}?download=true"
        return download_verified(url, destination, artifact.sha256)

    def _load_models(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        for source in self.weights:
            if source in self.models:
                continue
            try:
                path = self._resolve_weight(source)
                if path is None:
                    self.status[source] = {
                        "ready": False,
                        "error": "weights unavailable",
                    }
                    continue
                self.models[source] = load_yolo(path)
                self.status[source] = {"ready": True, "path": path}
                log.info("event=threat_model_ready source=%s path=%s", source, path)
            except Exception as exc:
                self.status[source] = {"ready": False, "error": type(exc).__name__}
                log.warning(
                    "event=threat_model_load_failed source=%s error=%r", source, exc
                )

    @staticmethod
    def _label(names: Any, index: int) -> str:
        if isinstance(names, dict):
            return str(names.get(index, index)).strip().lower()
        if isinstance(names, (list, tuple)) and 0 <= index < len(names):
            return str(names[index]).strip().lower()
        return str(index)

    def _category(self, label: str) -> Optional[str]:
        normalized = label.replace("_", "-").replace(" ", "-")
        if normalized in self.WEAPONS:
            return "weapon"
        if normalized in self.FIRE:
            return "fire"
        if normalized in self.SMOKE:
            return "smoke"
        if normalized in self.BAGS:
            return "bag"
        return None

    def _predict(self, frame, source: str) -> List[ThreatDetection]:
        model = self.models[source]
        kwargs = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        kwargs["device"] = self.device
        results = model.predict(frame, **kwargs)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        out = []
        for class_id, confidence, bbox in zip(classes, confidences, boxes):
            label = self._label(result.names, int(class_id))
            category = self._category(label)
            if category is not None and (
                source != "fire" or category in FIRE_CATEGORIES
            ):
                out.append(
                    ThreatDetection(
                        source=source,
                        label=label,
                        category=category,
                        confidence=float(confidence),
                        bbox=list(bbox),
                    )
                )
        return out

    @staticmethod
    def _best(detections: Iterable[ThreatDetection], category: str):
        candidates = [d for d in detections if d.category == category]
        return max(candidates, key=lambda d: d.confidence, default=None)

    def _bag_is_separated(self, detection: ThreatDetection, persons) -> bool:
        if not persons:
            return True
        bx = (detection.bbox[0] + detection.bbox[2]) / 2.0
        by = (detection.bbox[1] + detection.bbox[3]) / 2.0
        nearest = min(
            (
                (bx - (p.bbox[0] + p.bbox[2]) / 2.0) ** 2
                + (by - (p.bbox[1] + p.bbox[3]) / 2.0) ** 2
            )
            ** 0.5
            for p in persons
        )
        return nearest > self.bag_proximity_px

    def detect(self, frame, persons=None, timestamp: float = 0.0) -> ThreatFrame:
        self._load_models()
        detections = []
        for source in self.models:
            path = self.status.get(source, {}).get("path")
            try:
                detections.extend(self._predict(frame, source))
                self.status[source] = {"ready": True, "path": path}
            except Exception as exc:
                self.status[source] = {
                    "ready": False,
                    "path": path,
                    "error": type(exc).__name__,
                }
                log.warning(
                    "event=threat_inference_failed source=%s error=%r", source, exc
                )

        # Use surveillance fire/smoke evidence during dedicated model outages.
        # A model change starts a fresh confirmation streak.
        fire_source = (
            "fire" if self.status.get("fire", {}).get("ready") else "surveillance"
        )
        if fire_source != self._fire_source:
            for category in FIRE_CATEGORIES:
                self.streaks.pop(category, None)
            self._fire_source = fire_source
        detections = [
            detection
            for detection in detections
            if detection.category not in FIRE_CATEGORIES
            or detection.source == fire_source
        ]

        best = {
            category: self._best(detections, category)
            for category in ("weapon", "fire", "smoke", "bag")
        }
        for category, detection in best.items():
            threshold = self.conf if category == "weapon" else self.high_conf
            if detection is not None and detection.confidence >= threshold:
                self.streaks[category] = self.streaks.get(category, 0) + 1
            else:
                self.streaks[category] = 0

        alerts: Dict[str, Dict[str, Any]] = {}
        weapon = best["weapon"]
        if (
            self.enable_weapon_alerts
            and weapon is not None
            and weapon.confidence >= self.conf
            and self.streaks["weapon"] >= self.confirm_frames
        ):
            alerts["Weapon Threat"] = {
                "alert": True,
                "category": "weapon",
                "confidence": round(float(weapon.confidence), 3),
                "evidence": [weapon.to_dict()],
                "message": "Weapon candidate persisted across frames.",
            }
        for category, title in (("fire", "Fire Threat"), ("smoke", "Smoke Threat")):
            detection = best[category]
            if (
                self.enable_fire_alerts
                and detection is not None
                and self.streaks[category] >= self.confirm_frames
            ):
                alerts[title] = {
                    "alert": True,
                    "category": category,
                    "confidence": round(float(detection.confidence), 3),
                    "evidence": [detection.to_dict()],
                    "message": f"{category.title()} candidate persisted across frames.",
                }
        bag = best["bag"]
        if (
            self.enable_bag_alerts
            and bag is not None
            and self._bag_is_separated(bag, persons or [])
            and self.streaks["bag"] >= self.confirm_frames
        ):
            alerts["Unattended Bag Candidate"] = {
                "alert": True,
                "category": "bag",
                "confidence": round(float(bag.confidence), 3),
                "message": "Bag candidate remains separated from tracked people.",
            }
        return ThreatFrame(
            detections=detections, alerts=alerts, models=dict(self.status)
        )


# Backwards-compatible name for existing callers.
MultiModelThreatDetector = SurveillanceThreatDetector

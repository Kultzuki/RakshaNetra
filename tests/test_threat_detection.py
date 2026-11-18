import numpy as np
import pytest

from core.threat_detection import AUXILIARY_MODELS, MultiModelThreatDetector


class _Field:
    def __init__(self, values):
        self.values = values

    def int(self):
        return _Field(np.asarray(self.values, dtype=int))

    def cpu(self):
        return self

    def tolist(self):
        return np.asarray(self.values).tolist()


class _Boxes:
    def __init__(self, classes, confidences, boxes):
        self.cls = _Field(classes)
        self.conf = _Field(confidences)
        self.xyxy = _Field(boxes)

    def __len__(self):
        return len(self.conf.values)


class _Result:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _Model:
    def __init__(self, result):
        self.result = result

    def predict(self, frame, **kwargs):
        return [self.result]


def _model(label, confidence):
    return _Model(
        _Result(
            {0: label},
            _Boxes([0], [confidence], [[10.0, 10.0, 50.0, 50.0]]),
        )
    )


def _detector(**kwargs):
    detector = MultiModelThreatDetector(
        download_missing=False, confirm_frames=2, **kwargs
    )
    detector._load_attempted = True
    return detector


def test_fire_model_is_configured_only_when_fire_alerts_are_enabled():
    assert set(_detector().weights) == {"surveillance", "fire"}
    assert set(_detector(enable_fire_alerts=False).weights) == {"surveillance"}
    assert AUXILIARY_MODELS["fire"].repo == "SalahALHaismawi/yolov26-fire-detection"


def test_threat_weights_argument_is_ignored():
    detector = _detector(threat_weights="weights/best.pt")
    assert "threat" not in detector.weights


@pytest.mark.parametrize("label", ["short-gun", "long-gun", "knife"])
def test_visible_weapon_alert_needs_consecutive_detections(label):
    detector = _detector(enable_weapon_alerts=True)
    detector.models = {"surveillance": _model(label, 0.65)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    assert "Weapon Threat" not in detector.detect(frame).alerts
    alert = detector.detect(frame).alerts["Weapon Threat"]
    assert alert["evidence"][0]["label"] == label
    assert alert["confidence"] == 0.65


def test_weapon_alert_requires_enable_flag():
    detector = _detector()
    detector.models = {"surveillance": _model("short-gun", 0.8)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    detector.detect(frame)
    assert "Weapon Threat" not in detector.detect(frame).alerts


def test_low_confidence_weapon_never_alerts():
    detector = _detector(enable_weapon_alerts=True, conf=0.9)
    detector.models = {"surveillance": _model("short-gun", 0.8)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    for _ in range(5):
        assert "Weapon Threat" not in detector.detect(frame).alerts


def test_weapon_confirmation_resets_after_a_missing_detection():
    detector = _detector(enable_weapon_alerts=True)
    detector.models = {"surveillance": _model("knife", 0.65)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    assert not detector.detect(frame).alerts
    detector.models = {}
    assert not detector.detect(frame).alerts
    detector.models = {"surveillance": _model("knife", 0.65)}
    assert not detector.detect(frame).alerts
    assert "Weapon Threat" in detector.detect(frame).alerts


@pytest.mark.parametrize("label", ["fire", "smoke"])
def test_fire_model_alerts_need_high_confidence_streak(label):
    detector = _detector()
    detector.models = {"fire": _model(label, 0.9)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    title = f"{label.title()} Threat"
    assert title not in detector.detect(frame).alerts
    alert = detector.detect(frame).alerts[title]
    assert alert["evidence"][0]["source"] == "fire"
    assert alert["confidence"] == 0.9


def test_dedicated_fire_model_takes_priority_and_fallback_starts_fresh(monkeypatch):
    detector = _detector()
    fire_model = _model("fire", 0.9)
    detector.models = {
        "fire": fire_model,
        "surveillance": _model("fire", 0.99),
    }
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    first = detector.detect(frame)
    assert not first.alerts
    assert [item.source for item in first.detections] == ["fire"]

    def fail(*args, **kwargs):
        raise RuntimeError("dedicated model offline")

    monkeypatch.setattr(fire_model, "predict", fail)
    fallback = detector.detect(frame)
    assert not fallback.alerts
    assert not fallback.models["fire"]["ready"]
    alert = detector.detect(frame).alerts["Fire Threat"]
    assert alert["evidence"][0]["source"] == "surveillance"


def test_other_fire_model_labels_do_not_create_alerts():
    detector = _detector(enable_weapon_alerts=True)
    detector.models = {"fire": _model("other", 0.99)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    assert not detector.detect(frame).detections
    assert not detector.detect(frame).alerts


def test_reset_retries_unavailable_fire_model_without_reloading_surveillance(
    monkeypatch,
):
    detector = MultiModelThreatDetector(download_missing=False)
    loads = []
    paths = {"surveillance": "surveillance.pt", "fire": None}

    def load(path):
        loads.append(path)
        return _model("fire", 0.9)

    monkeypatch.setattr(detector, "_resolve_weight", paths.get)
    monkeypatch.setattr("core.threat_detection.load_yolo", load)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    assert not detector.detect(frame).models["fire"]["ready"]
    paths["fire"] = "fire.pt"
    detector.reset()
    assert detector.detect(frame).models["fire"]["ready"]
    assert loads == ["surveillance.pt", "fire.pt"]


def test_fire_alert_respects_enable_flag():
    detector = _detector(enable_fire_alerts=False)
    detector.models = {"surveillance": _model("fire", 0.95)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    for _ in range(5):
        assert "Fire Threat" not in detector.detect(frame).alerts
        assert "Smoke Threat" not in detector.detect(frame).alerts


def test_bag_alert_dark_by_default():
    detector = _detector()
    detector.models = {"surveillance": _model("bag", 0.8)}
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    for _ in range(5):
        assert "Unattended Bag Candidate" not in detector.detect(frame).alerts

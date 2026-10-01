"""Tests for the Grounding DINO head detector, landmark yaw and face visibility.

Only ``test_grounding_dino_real_frame`` needs the real video and cached weights
(skipped otherwise)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from horse_reid.config import PipelineConfig, QualityWeights
from horse_reid.face.grounded_head_detector import GroundingDinoHeadDetector, collect_parts, pick_head
from horse_reid.face.head_detector import BBoxTopHeadDetector, build_head_detector
from horse_reid.pipeline import auto_min_gap
from horse_reid.quality import FrameQualityScorer
from horse_reid.quality.metrics import face_visibility_score, yaw_from_landmarks
from horse_reid.types import Detection, FrameScore, HeadDetection

ROOT = Path(__file__).resolve().parents[1]
REAL_VIDEO = ROOT / "data/input/1000018050.mp4"
HF_CACHE = ROOT / "models/hf/hub"
GD_CACHED = (HF_CACHE / "models--IDEA-Research--grounding-dino-tiny").exists()

HEAD = (100, 100, 300, 400)  # w = 200, centre x = 200


# --------------------------------------------------------------------------- #
# landmark yaw
# --------------------------------------------------------------------------- #
def test_yaw_frontal_two_eyes() -> None:
    kps = {"eye": [[130.0, 200.0, 0.6], [270.0, 200.0, 0.5]], "nose": [[200.0, 360.0, 0.5]], "ear": []}
    yaw, conf = yaw_from_landmarks(HEAD, kps)
    assert abs(yaw) < 20
    assert conf == pytest.approx(0.8)


def test_yaw_profile_nose_extreme_one_eye() -> None:
    left = {"eye": [[180.0, 200.0, 0.5]], "nose": [[102.0, 350.0, 0.6]]}
    yaw, conf = yaw_from_landmarks(HEAD, left)
    assert yaw < -80 and conf == pytest.approx(0.6)
    right = {"eye": [[220.0, 200.0, 0.5]], "nose": [[298.0, 350.0, 0.6]]}
    yaw, _ = yaw_from_landmarks(HEAD, right)
    assert yaw > 80


def test_yaw_other_cases() -> None:
    # nose only, centred -> 30 deg
    yaw, conf = yaw_from_landmarks(HEAD, {"nose": [[200.0, 350.0, 0.5]]})
    assert abs(yaw) == pytest.approx(30.0) and conf == pytest.approx(0.6)
    # one eye only -> 60, sign from the eye
    yaw, conf = yaw_from_landmarks(HEAD, {"eye": [[150.0, 200.0, 0.4]]})
    assert yaw == pytest.approx(-60.0) and conf == pytest.approx(0.3)
    # weak keypoints are ignored
    assert yaw_from_landmarks(HEAD, {"eye": [[150.0, 200.0, 0.1]], "nose": [[200.0, 350.0, 0.2]]}) is None
    assert yaw_from_landmarks(HEAD, None) is None
    assert yaw_from_landmarks(HEAD, {"ear": [[150.0, 100.0, 0.9]]}) is None


def test_face_visibility_score() -> None:
    assert face_visibility_score(None) == pytest.approx(0.1)
    strong = {"eye": [[0, 0, 0.5]], "nose": [[0, 0, 0.45]], "ear": [[0, 0, 0.4]]}
    assert face_visibility_score(strong) == pytest.approx(1.0)
    weak = {"eye": [[0, 0, 0.1]], "nose": [[0, 0, 0.05]], "ear": []}
    assert face_visibility_score(weak) < 0.3
    assert face_visibility_score({"ear": [[0, 0, 0.5]], "eye": [], "nose": []}) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# detector helpers / registry (no weights)
# --------------------------------------------------------------------------- #
def test_pick_head_and_parts() -> None:
    items = [
        ("horse head", 0.30, (0.0, 0.0, 95.0, 95.0)),      # whole horse -> rejected (> 50% of 100x100)
        ("horse head", 0.45, (10.0, 10.0, 50.0, 60.0)),
        ("head", 0.25, (12.0, 12.0, 48.0, 58.0)),
        ("horse horse eye", 0.40, (20.0, 20.0, 26.0, 26.0)),
        ("eye", 0.30, (21.0, 21.0, 27.0, 27.0)),            # duplicate of the first eye
        ("horse eye", 0.35, (38.0, 20.0, 44.0, 26.0)),
        ("horse eye", 0.20, (80.0, 80.0, 86.0, 86.0)),      # outside the head box
        ("horse nose", 0.50, (25.0, 50.0, 35.0, 58.0)),
        ("horse nose", 0.22, (40.0, 50.0, 46.0, 58.0)),     # only 1 nose kept
        ("ear", 0.3, (12.0, 5.0, 18.0, 12.0)),
    ]
    best = pick_head(items, 100, 100)
    assert best is not None and best[0] == pytest.approx(0.45)
    parts = collect_parts(items, best[1])
    assert len(parts["eye"]) == 2 and parts["eye"][0][2] == pytest.approx(0.40)
    assert len(parts["nose"]) == 1 and len(parts["ear"]) == 1
    assert pick_head([("horse head", 0.3, (0.0, 0.0, 95.0, 95.0))], 100, 100) is None


def test_grounding_dino_registry_is_lazy() -> None:
    det = build_head_detector("grounding_dino", fallback_name="none")
    assert isinstance(det, GroundingDinoHeadDetector)
    assert not det.loaded and det.fallback is None
    assert det.cache_dir is not None and det.cache_dir.is_absolute()
    det = build_head_detector("grounding_dino", fallback_name="bbox_top")
    assert isinstance(det.fallback, BBoxTopHeadDetector) and not det.loaded
    det = build_head_detector("grounding_dino")  # default fallback: mask_top (or bbox_top if seg weights fail)
    assert det.fallback is not None and not det.loaded
    with pytest.raises(ValueError):
        build_head_detector("grounding_dino", fallback_name="nope")


class _NoHeadGD(GroundingDinoHeadDetector):
    def run(self, crop_bgr: np.ndarray):  # type: ignore[override]
        return []


def test_grounding_dino_fallback_suffix() -> None:
    frame = np.zeros((400, 300, 3), np.uint8)
    det = _NoHeadGD(fallback=BBoxTopHeadDetector()).detect(frame, (50, 100, 250, 300), 0.8)
    assert det is not None and det.method == "bbox_top_heuristic(gd_fallback)"
    assert _NoHeadGD().detect(frame, (50, 100, 250, 300), 0.8) is None


# --------------------------------------------------------------------------- #
# scorer / config / pipeline helpers
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one() -> None:
    assert QualityWeights().total() == pytest.approx(1.0)


def test_scorer_visibility_gate_and_heuristic_factor() -> None:
    rng = np.random.default_rng(0)
    frame = (rng.random((400, 400, 3)) * 255).astype(np.uint8)
    horse = Detection(0, 1, (0, 0, 400, 400), 0.9, "horse")
    good = {"eye": [[130.0, 150.0, 0.6], [270.0, 150.0, 0.5]], "nose": [[200.0, 300.0, 0.5]], "ear": []}
    rear = {"eye": [[150.0, 150.0, 0.1]], "nose": [], "ear": [[150.0, 110.0, 0.4]]}
    sc = FrameQualityScorer()
    fs_good = sc.score(frame, horse, HeadDetection((100, 100, 300, 340), 0.5, "grounding_dino", good))
    fs_rear = sc.score(frame, horse, HeadDetection((100, 100, 300, 340), 0.5, "grounding_dino", rear))
    assert fs_good.visibility == pytest.approx(1.0) and not fs_good.gated
    assert fs_good.extra["yaw_source"] == "landmarks" and abs(fs_good.yaw) < 20
    assert fs_rear.gated and fs_rear.score < 0.3 * fs_good.score
    heur = sc.score(frame, horse, HeadDetection((100, 100, 300, 340), 0.5, "mask_top_heuristic",
                                                {"poll_top": [[200.0, 100.0, 1.0]]}))
    assert heur.extra["yaw_source"] == "appearance" and heur.visibility == pytest.approx(0.1)
    assert not heur.gated and heur.score < fs_good.score
    fb = sc.score(frame, horse, HeadDetection((100, 100, 300, 340), 0.5, "mask_top_heuristic(gd_fallback)"))
    assert fb.gated
    d = fs_good.to_dict()
    assert "visibility" in d and d["extra"]["keypoints"]["eye"][0] == [130.0, 150.0, 0.6]


def test_head_stride_config_and_min_gap() -> None:
    assert PipelineConfig().head_stride == 3 and PipelineConfig().head_detector == "grounding_dino"
    with pytest.raises(ValueError):
        PipelineConfig(head_stride=0)
    with pytest.raises(ValueError):
        PipelineConfig(head_fallback="x")
    assert PipelineConfig().to_dict()["head_stride"] == 3
    # 1800 frames, every 3rd scored -> 600 scored; 30 picks -> 10 scored frames = 30 frame indices
    assert auto_min_gap(1800, 600, 30, 3) == 30
    assert auto_min_gap(45, 15, 5, 3) >= 3
    assert auto_min_gap(10, 4, 30, 5) >= 5


def test_frame_score_visibility_default() -> None:
    fs = FrameScore(frame=0, time_s=0, track_id=1, horse_bbox=(0, 0, 1, 1), head_bbox=(0, 0, 1, 1),
                    head_conf=1, head_method="t", size=1, blur_var=1, blur=1, exposure=1, occlusion=1,
                    yaw=0, yaw_conf=1, view=1, det_conf=1, score=1)
    assert fs.to_dict()["visibility"] == 0.0


# --------------------------------------------------------------------------- #
# real model (optional)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (REAL_VIDEO.exists() and GD_CACHED), reason="real video or Grounding DINO cache missing")
def test_grounding_dino_real_frame() -> None:
    pytest.importorskip("transformers")
    from horse_reid.detection import YoloHorseDetector
    from horse_reid.video import VideoReader

    with VideoReader(REAL_VIDEO) as r:
        idx = int(round(25.0 * r.fps))
        frame = r.read_frame(idx)
    assert frame is not None
    yolo = ROOT / "models/yolo11s.pt"
    if not yolo.exists():
        pytest.skip("YOLO weights missing")
    dets = YoloHorseDetector(yolo, device="cpu").track(frame, idx, 25.0)
    horses = [d for d in dets if d.cls_name == "horse"]
    assert horses, "no horse detected at t=25s"
    horse = max(horses, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    gd = GroundingDinoHeadDetector(device="cpu", cache_dir=HF_CACHE)
    head = gd.detect(frame, horse.bbox, horse.confidence)
    assert head is not None and head.method == "grounding_dino"
    hx1, hy1, hx2, hy2 = head.bbox
    x1, y1, x2, y2 = horse.bbox
    m = 0.15 * max(x2 - x1, y2 - y1)
    assert x1 - m <= hx1 < hx2 <= x2 + m and y1 - m <= hy1 < hy2 <= y2 + m
    assert head.keypoints is not None
    assert head.keypoints["eye"] and head.keypoints["nose"]


def test_gd_fallback_never_outranks_real_detection() -> None:
    from horse_reid.config import QualityParams, QualityWeights
    from horse_reid.quality.scorer import combine_score

    p, w = QualityParams(), QualityWeights()
    base = dict(size=1.0, blur=1.0, exposure=1.0, occlusion=1.0, view=1.0, det_conf=1.0, visibility=0.59)
    fb, fb_gated = combine_score({**base, "head_method": "mask_top_heuristic(gd_fallback)"}, p, w)
    mediocre = dict(size=0.5, blur=0.5, exposure=0.6, occlusion=0.8, view=0.5, det_conf=0.6, visibility=0.6)
    real, real_gated = combine_score({**mediocre, "head_method": "grounding_dino"}, p, w)
    assert fb_gated and not real_gated and fb < real
    # stand-alone heuristic: not gated, only the 0.85 factor
    h, h_gated = combine_score({**base, "head_method": "mask_top_heuristic"}, p, w)
    plain, _ = combine_score({**base, "head_method": "grounding_dino"}, p, w)
    assert not h_gated and h == pytest.approx(plain * p.heuristic_head_factor)


def test_head_conf_and_area_gates() -> None:
    from horse_reid.config import QualityParams, QualityWeights
    from horse_reid.quality.scorer import combine_score

    p, w = QualityParams(), QualityWeights()
    base = dict(size=1.0, blur=1.0, exposure=1.0, occlusion=1.0, view=1.0, det_conf=0.9, horse_conf=0.9,
                visibility=0.9, head_method="grounding_dino", head_conf=0.6, head_area_ratio=0.2)
    ok, g_ok = combine_score(base, p, w)
    low, g_low = combine_score({**base, "head_conf": 0.2}, p, w)
    assert not g_ok and g_low and low < 0.3 * ok
    heur, g_h = combine_score({**base, "head_conf": 0.2, "head_method": "mask_top_heuristic"}, p, w)
    assert not g_h
    big, g_big = combine_score({**base, "head_area_ratio": 0.5}, p, w)
    assert g_big and big == pytest.approx(ok * p.gate_factor)
    # det_conf component = horse_conf * clip(head_conf/0.5)
    half, _ = combine_score({**base, "head_conf": 0.4, "head_area_ratio": 0.2}, p, w)
    assert half < ok

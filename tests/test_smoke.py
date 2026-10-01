"""Fast unit tests; none require the real video or model weights."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from horse_reid.config import PipelineConfig
from horse_reid.device import select_device
from horse_reid.face.head_detector import BBoxTopHeadDetector, available_head_detectors, build_head_detector
from horse_reid.quality.metrics import (
    blur_score,
    exposure_score,
    face_size_score,
    occlusion_score,
    view_score,
    yaw_bin,
    yaw_estimate,
)
from horse_reid.selection import select_frames
from horse_reid.tracking import TrackStore
from horse_reid.types import Detection, FrameScore
from horse_reid.video import VideoReader, rotate_frame
from horse_reid.visualization import draw_detections, make_contact_sheet

ROOT = Path(__file__).resolve().parents[1]
REAL_VIDEO = ROOT / "data/input/1000018050.mp4"


def _textured(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
def test_select_device_cpu() -> None:
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in ("cuda", "mps", "cpu")
    with pytest.raises(ValueError):
        select_device("tpu")


def test_rotate_frame_shapes() -> None:
    f = np.zeros((720, 1280, 3), np.uint8)
    f[0, 0] = 255  # top-left marker
    r90 = rotate_frame(f, 90)
    assert r90.shape == (1280, 720, 3)
    assert r90[0, -1].max() == 255  # top-left -> top-right after 90 deg clockwise
    assert rotate_frame(f, 180).shape == (720, 1280, 3)
    assert rotate_frame(f, 270).shape == (1280, 720, 3)
    assert rotate_frame(f, 0) is f


def _write_video(path: Path, n: int = 12, w: int = 64, h: int = 48) -> bool:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not vw.isOpened():
        return False
    for i in range(n):
        f = np.full((h, w, 3), i * 20 % 255, np.uint8)
        vw.write(f)
    vw.release()
    return path.exists() and path.stat().st_size > 0


def test_video_reader_rotation_override(tmp_path: Path) -> None:
    p = tmp_path / "tiny.mp4"
    if not _write_video(p):
        pytest.skip("cv2.VideoWriter cannot write mp4 here")
    with VideoReader(p) as r:
        if r.frame_count <= 0:
            pytest.skip("cannot read back synthetic video")
        frames = list(r)
        assert frames[0][2].shape[:2] == (r.height, r.width)
        assert len(frames) == r.frame_count
    with VideoReader(p, rotation=90, stride=3) as r:
        frames = list(r)
        assert frames[0][2].shape[:2] == (64, 48)  # portrait after forced rotation
        assert [f[0] for f in frames] == list(range(0, r.frame_count, 3))
        assert r.read_frame(5).shape[:2] == (64, 48)
    with VideoReader(p, max_frames=4) as r:
        assert len(list(r)) == 4


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="real video not available")
def test_real_video_is_portrait() -> None:
    with VideoReader(REAL_VIDEO, max_frames=1) as r:
        idx, ts, f = next(iter(r))
        assert f.shape[:2] == (1280, 720)


# --------------------------------------------------------------------------- #
def test_blur_score_ordering() -> None:
    sharp = _textured()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    assert blur_score(sharp) > blur_score(blurred)
    assert 0.0 <= blur_score(blurred) <= 1.0


def test_exposure_score() -> None:
    mid = np.full((64, 64, 3), 128, np.uint8)
    dark = np.full((64, 64, 3), 5, np.uint8)
    bright = np.full((64, 64, 3), 252, np.uint8)
    assert exposure_score(mid) == pytest.approx(1.0)
    assert exposure_score(dark) < 0.2
    assert exposure_score(bright) < 0.2


def test_occlusion_score() -> None:
    head = (100, 100, 200, 200)
    assert occlusion_score(head, []) == 1.0
    assert occlusion_score(head, [(150, 100, 250, 200)]) == pytest.approx(0.5)
    assert occlusion_score(head, [(0, 0, 500, 500)]) == 0.0
    assert occlusion_score(head, [(300, 300, 400, 400)]) == 1.0
    # touching the border is penalised
    assert occlusion_score((0, 100, 100, 200), [], frame_shape=(720, 1280)) < 1.0


def test_size_view_yaw() -> None:
    assert face_size_score((0, 0, 300, 300), (720, 1280)) == 1.0
    assert face_size_score((0, 0, 150, 150), (720, 1280)) == pytest.approx(0.5)
    assert view_score(0) == view_score(45) == view_score(-60) == 1.0
    assert view_score(90) == pytest.approx(0.4)
    assert yaw_bin(10) == "frontal" and yaw_bin(-45) == "three_quarter" and yaw_bin(80) == "profile"
    sym = np.zeros((100, 60, 3), np.uint8)
    cv2.circle(sym, (30, 50), 20, (255, 255, 255), -1)
    yaw_sym, _ = yaw_estimate(sym)
    asym = np.zeros((60, 100, 3), np.uint8)
    cv2.rectangle(asym, (60, 10), (95, 50), (255, 255, 255), -1)
    yaw_asym, _ = yaw_estimate(asym)
    assert abs(yaw_sym) < abs(yaw_asym)


def test_bbox_top_head_detector() -> None:
    frame = np.zeros((400, 300, 3), np.uint8)
    det = BBoxTopHeadDetector().detect(frame, (50, 100, 250, 300), horse_conf=0.8)
    assert det is not None
    x1, y1, x2, y2 = det.bbox
    assert y1 == 100 and y2 == 100 + round(200 * 0.42)
    assert 50 <= x1 < x2 <= 250
    assert det.confidence == pytest.approx(0.4)
    assert det.method == "bbox_top_heuristic"
    # with a mask, the denser half wins
    mask = np.zeros((200, 200), bool)
    mask[:, 150:] = True
    det = BBoxTopHeadDetector().detect(frame, (50, 100, 250, 300), mask=mask)
    assert det.bbox[2] == 250 and det.bbox[0] > 50


def test_head_registry() -> None:
    assert isinstance(build_head_detector("bbox_top"), BBoxTopHeadDetector)
    assert "grounding_dino" in available_head_detectors()
    with pytest.raises(ValueError):
        build_head_detector("nope")


# --------------------------------------------------------------------------- #
def _fs(frame: int, score: float, yaw: float = 0.0) -> FrameScore:
    return FrameScore(frame=frame, time_s=frame / 30, track_id=1, horse_bbox=(0, 0, 10, 10),
                      head_bbox=(0, 0, 5, 5), head_conf=0.5, head_method="t", size=1, blur_var=1,
                      blur=1, exposure=1, occlusion=1, yaw=yaw, yaw_conf=0.5, view=1, det_conf=1,
                      score=score)


def test_select_frames_spacing() -> None:
    rng = np.random.default_rng(1)
    scores = [_fs(i, float(rng.random())) for i in range(300)]
    sel = select_frames(scores, num_frames=10, min_gap_frames=20)
    assert len(sel) == 10
    frames = [s.frame for s in sel]
    assert frames == sorted(frames)
    assert all(b - a >= 20 for a, b in zip(frames, frames[1:]))
    # default gap = max(3, 300 // 20) = 15
    sel = select_frames(scores, num_frames=10)
    frames = [s.frame for s in sel]
    assert all(b - a >= 15 for a, b in zip(frames, frames[1:]))


def test_select_frames_diversity() -> None:
    # frontal frames score highest; a few good profile frames exist.
    scores = [_fs(i, 0.9 - i * 1e-4, yaw=5.0) for i in range(0, 200)]
    scores += [_fs(i, 0.6, yaw=80.0) for i in range(200, 300, 10)]
    sel = select_frames(scores, num_frames=10, min_gap_frames=5)
    bins = [yaw_bin(s.yaw) for s in sel]
    assert len(sel) == 10
    assert bins.count("profile") >= 2  # ceil(10 * 0.15)
    frames = [s.frame for s in sel]
    assert all(b - a >= 5 for a, b in zip(frames, frames[1:]))


def test_select_frames_empty() -> None:
    assert select_frames([], 5) == []


# --------------------------------------------------------------------------- #
def test_contact_sheet(tmp_path: Path) -> None:
    items = [(_textured(100 + 10 * i, 80, seed=i), [f"frame: {i}", "score: 0.9"]) for i in range(5)]
    out = tmp_path / "sheet.jpg"
    img = make_contact_sheet(items, save_path=out)
    assert out.exists()
    assert img.shape[1] == 3 * (256 + 12)  # ceil(sqrt(5)) = 3 columns
    assert img.shape[0] == 2 * (256 + 6 + 2 * 18 + 6)
    assert make_contact_sheet([]).size > 0


def test_track_store_and_annotate() -> None:
    store = TrackStore()
    store.add([Detection(0, 1, (0, 0, 10, 10), 0.9, "horse"), Detection(0, 2, (0, 0, 5, 5), 0.9, "person")])
    store.add([Detection(1, 1, (0, 0, 10, 10), 0.9, "horse")])
    store.add([Detection(5, 3, (0, 0, 20, 20), 0.9, "horse")])
    assert store.primary_horse_track() == 1
    assert store.primary_track_group() == [1, 3]
    js = store.to_json()
    assert js["primary_track_id"] == 1 and len(js["detections"]) == 4
    img = draw_detections(np.zeros((50, 50, 3), np.uint8), store.frame_detections(0)[:1],
                          store.frame_detections(0)[1:], (1, 1, 5, 5), "hi")
    assert img.shape == (50, 50, 3)


def test_config_roundtrip() -> None:
    cfg = PipelineConfig(num_frames=5)
    d = cfg.to_dict()
    assert d["num_frames"] == 5 and isinstance(d["input"], str)
    with pytest.raises(ValueError):
        PipelineConfig(rotation=45)


# --------------------------------------------------------------------------- #
def test_track_class_is_majority_not_first() -> None:
    store = TrackStore()
    # track 69: first box labelled "person", then a long run of "horse" boxes
    store.add([Detection(10, 69, (0, 0, 400, 600), 0.9, "person")])
    for f in range(11, 60):
        store.add([Detection(f, 69, (0, 0, 400, 600), 0.9, "horse")])
    # track 1: shorter, disjoint horse track before it
    for f in range(0, 10):
        store.add([Detection(f, 1, (0, 0, 400, 600), 0.9, "horse")])
    # tiny false horse track, disjoint in time -> excluded from the group by area
    for f in range(60, 70):
        store.add([Detection(f, 186, (0, 0, 90, 90), 0.5, "horse")])
    # a real person track stays a person track
    store.add([Detection(f, 2, (0, 0, 50, 50), 0.9, "person") for f in range(0, 80)])
    assert store.track_class(69) == "horse"
    assert store.track_class(2) == "person"
    assert store.horse_track_ids() == [1, 69, 186]
    assert store.track_length(69) == 49  # the stray "person" frame does not count
    assert all(d.cls_name == "horse" for d in store.track(69))
    assert store.primary_horse_track() == 69
    assert store.primary_track_group() == [69, 1]
    js = store.to_json()
    assert js["primary_track_id"] == 69 and js["primary_track_length"] == 49
    assert len(js["detections"]) == 50 + 10 + 10 + 80  # export keeps everything
    # roundtrip via detection dicts
    store2 = TrackStore.from_detection_dicts(js["detections"])
    assert store2.primary_horse_track() == 69 and store2.primary_track_group() == [69, 1]


def test_track_class_tie_breaks_by_area() -> None:
    store = TrackStore()
    store.add([Detection(0, 5, (0, 0, 10, 10), 0.9, "person"),
               Detection(1, 5, (0, 0, 100, 100), 0.9, "horse")])
    assert store.track_class(5) == "horse"
    store.add([Detection(2, 5, (0, 0, 10, 10), 0.9, "person")])
    assert store.track_class(5) == "person"  # cache invalidated by add()


def test_select_frames_min_score() -> None:
    good = [_fs(i * 20, 0.8) for i in range(5)]
    junk = [_fs(200 + i * 20, 0.1) for i in range(20)]
    sel = select_frames(good + junk, num_frames=10, min_gap_frames=5)
    assert len(sel) == 5 and all(s.score >= 0.35 for s in sel)
    sel = select_frames(good + junk, num_frames=10, min_gap_frames=5, min_score=0.0)
    assert len(sel) == 10
    # no candidate passes -> low-quality fallback still returns frames, flagged
    sel = select_frames(junk, num_frames=3)
    assert len(sel) == 3 and sel.low_quality and "only 0 of 20" in sel.reason
    assert select_frames(junk, num_frames=3, low_quality_fallback=False) == []
    assert not select_frames(good + junk, num_frames=10, min_gap_frames=5).low_quality


def test_select_frames_low_quality_fallback() -> None:
    # 2 of 40 pass (< min(30, 5)) -> fallback ranks everything, keeps gap/diversity rules
    cands = [_fs(i * 10, 0.1 + (i % 7) * 0.01, yaw=(80.0 if i % 4 == 0 else 10.0)) for i in range(40)]
    cands[3].score = cands[17].score = 0.6
    sel = select_frames(cands, num_frames=30, min_gap_frames=20)
    assert sel.low_quality and sel.n_pass == 2 and sel.n_candidates == 40
    assert "only 2 of 40 candidates" in sel.reason
    frames = [s.frame for s in sel]
    assert len(frames) >= 15 and all(b - a >= 20 for a, b in zip(frames, frames[1:]))
    assert 30 in frames and 170 in frames  # the two passing frames are still picked
    # 5 passing (= min(30, 5)) -> no fallback, only the passing ones
    for i in (5, 9, 25):
        cands[i].score = 0.6
    sel = select_frames(cands, num_frames=30, min_gap_frames=20)
    assert not sel.low_quality and len(sel) == 5 and sel.reason is None


def _blur_fs(frame: int, blur_var: float) -> FrameScore:
    fs = _fs(frame, 0.0, yaw=10.0)
    fs.blur_var = blur_var
    fs.head_method, fs.head_conf, fs.visibility = "grounding_dino", 0.6, 0.9
    fs.extra = {"keypoints": {"eye": [[2.0, 2.0, 0.9]], "nose": [[3.0, 3.0, 0.9]]}}
    return fs


def test_adaptive_blur_ref_scale_invariant() -> None:
    from horse_reid.config import QualityParams, QualityWeights
    from horse_reid.quality.scorer import rescore_all, resolve_blur_ref

    rng = np.random.default_rng(3)
    base = rng.lognormal(mean=np.log(400.0), sigma=0.6, size=200)  # far median ~40 > blur_ref_min
    near = [_blur_fs(i, float(v)) for i, v in enumerate(base)]
    far = [_blur_fs(i, float(v) / 10.0) for i, v in enumerate(base)]  # far-away horse: 10x lower
    p, w = QualityParams(), QualityWeights()
    assert p.blur_ref_mode == "adaptive"
    ref_near, ref_far = rescore_all(near, p, w), rescore_all(far, p, w)
    assert ref_near == pytest.approx(float(np.median(base)))
    assert ref_far == pytest.approx(max(20.0, float(np.median(base)) / 10.0))
    bn = np.array([s.blur for s in near])
    bf = np.array([s.blur for s in far])
    assert np.median(bn) == pytest.approx(1 - np.exp(-1), abs=0.01)
    assert np.all(np.abs(np.sort(bn) - np.sort(bf)) < 0.05)  # similar blur-score distributions
    assert np.mean([s.blur < p.gate_min_blur for s in far]) < 0.1
    # absolute mode penalises the far run
    pa = QualityParams(blur_ref_mode="absolute")
    assert rescore_all(far, pa, w) == 150.0
    assert np.median([s.blur for s in far]) < 0.3  # vs ~0.63 adaptive: absolute penalises far runs
    # floor: a uniformly tiny blur_var is not normalised below blur_ref_min
    assert resolve_blur_ref([1.0, 2.0, 3.0], p) == 20.0
    assert resolve_blur_ref([], p) == p.blur_ref
    with pytest.raises(ValueError):
        QualityParams(blur_ref_mode="nope")


def test_absolute_blur_mode_unchanged() -> None:
    from horse_reid.config import QualityParams, QualityWeights
    from horse_reid.quality.metrics import blur_variance
    from horse_reid.quality.scorer import FrameQualityScorer, rescore

    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:300, 100:300] = _textured(200, 200, seed=4)
    horse = Detection(0, 1, (50, 50, 350, 350), 0.9, "horse")
    from horse_reid.types import HeadDetection
    head = HeadDetection((100, 100, 300, 300), 0.6, "grounding_dino",
                         {"eye": [[150.0, 150.0, 0.9]], "nose": [[200.0, 250.0, 0.9]]})
    pa = QualityParams(blur_ref_mode="absolute")
    fs = FrameQualityScorer(QualityWeights(), pa).score(frame, horse, head)
    var = blur_variance(frame[100:300, 100:300])
    assert fs.blur_var == pytest.approx(var)
    assert fs.blur == pytest.approx(1 - np.exp(-var / 150.0))
    score0, blur0 = fs.score, fs.blur
    rescore(fs, pa, QualityWeights())
    assert fs.blur == pytest.approx(blur0) and fs.score == pytest.approx(score0)


def test_contact_sheet_title() -> None:
    items = [(_textured(100, 80, seed=i), ["LOW QUALITY frame: 1", "score: 0.1"]) for i in range(2)]
    plain = make_contact_sheet(items)
    titled = make_contact_sheet(items, title="LOW QUALITY selection")
    assert titled.shape[0] > plain.shape[0] and titled.shape[1] == plain.shape[1]


def test_reselect_rewrites_outputs(tmp_path: Path) -> None:
    import json

    from horse_reid.pipeline import run_reselect

    vid = tmp_path / "v.mp4"
    if not _write_video(vid, n=40, w=64, h=48):
        pytest.skip("cv2.VideoWriter cannot write mp4 here")
    with VideoReader(vid) as r:
        if r.frame_count <= 0:
            pytest.skip("cannot read back synthetic video")
    out = tmp_path / "out"
    out.mkdir()
    store = TrackStore()
    store.add([Detection(0, 7, (0, 0, 40, 40), 0.9, "person", 0.0)])
    store.add([Detection(f, 7, (0, 0, 40, 40), 0.9, "horse", f / 10) for f in range(1, 30)])
    store.add([Detection(f, 1, (0, 0, 40, 40), 0.9, "horse", f / 10) for f in range(30, 36)])
    store.export(out / "detections.json")

    def fsd(frame: int, tid: int, score: float) -> dict:
        d = _fs(frame, score, yaw=20.0).to_dict()
        q = score  # component quality drives the rescored score
        d.update(size=q, blur=q, exposure=q, occlusion=q, view=q, det_conf=q, visibility=q,
                 blur_var=float(-150.0 * np.log(1.0 - q)))  # raw blur_var consistent with blur=q @ ref 150
        d.update(track_id=tid, time_s=frame / 10, horse_bbox=[0, 0, 40, 40], head_bbox=[5, 5, 20, 20],
                 extra={"yaw_source": "landmarks", "keypoints": {"eye": [[10.0, 10.0, 0.9]]}})
        return d

    scores = [fsd(f, 7, 0.8) for f in range(3, 30, 3)] + [fsd(f, 1, 0.1) for f in (30, 33)]
    (out / "selected_frames").mkdir()
    (out / "selected_frames" / "frame_99999.jpg").write_bytes(b"stale")
    video = {"path": str(vid), "fps": 10.0, "frame_count": 40, "raw_size": [64, 48], "size": [64, 48],
             "rotation_meta": 0, "rotation_applied": 0, "duration_s": 4.0}
    summary = {"frames_processed": 40, "frames_with_horse": 36, "head_stride": 3, "frames_head_stage": 14,
               "device": "cpu", "wall_time_s": 123.0, "processed_fps": 0.3, "video_duration_s": 4.0,
               "selected": 2}
    (out / "frame_scores.json").write_text(json.dumps({
        "video": video, "config": PipelineConfig().to_dict(), "device": "cpu", "primary_track_id": 1,
        "primary_track_group": [1], "min_gap_frames": 3, "head_stride": 3, "selected": [30, 33],
        "timing_s": {"total_wall": 123.0}, "summary": summary, "scores": scores}))

    from horse_reid.config import QualityParams
    absq = QualityParams(blur_ref_mode="absolute")
    cfg = PipelineConfig(input=vid, output=out, num_frames=5, min_frame_gap=3, quality=absq)
    res = run_reselect(cfg)
    assert res["primary_track_id"] == 7 and res["primary_track_group"] == [1, 7]
    assert res["selected"] == 5 and res["num_frames_requested"] == 5
    doc = json.loads((out / "frame_scores.json").read_text())
    assert doc["primary_track_id"] == 7 and len(doc["selected"]) == 5
    assert all(f < 30 for f in doc["selected"])  # the 0.1-score frames are below min_score
    assert "reselected_utc" in doc and len(doc["scores"]) == len(scores)
    assert doc["summary"]["original_run"]["wall_time_s"] == 123.0
    assert json.loads((out / "detections.json").read_text())["primary_track_id"] == 7
    for sub in ("selected_frames", "selected_frames_raw", "face_crops"):
        assert sorted(p.name for p in (out / sub).glob("frame_*.jpg")) == \
            [f"frame_{f:05d}.jpg" for f in doc["selected"]]
    assert (out / "contact_sheet.jpg").exists()
    txt = (out / "summary.txt").read_text()
    assert "mode:                 reselect" in txt and "original run:" in txt and "123.0s" in txt
    assert "5 of 5 requested" in txt
    assert "rescored with current weights" in txt and doc["reselect"]["rescored"] is True
    assert doc["low_quality_selection"] is False and res["low_quality_selection"] is False
    assert doc["config"]["blur_ref_used"] == 150.0 and "blur_ref:             150.0 (absolute)" in txt
    assert res["median_head_px"] == "15x15" and "median_head_px:       15x15" in txt
    assert "quality warning: median head width 15 px < 96 px" in txt
    s0 = next(s for s in doc["scores"] if s["frame"] == 3)
    assert s0["extra"]["score_original"] == 0.8 and s0["score"] == pytest.approx(0.8)

    # changing weights changes the rescored score; score_original is not overwritten
    from horse_reid.config import QualityWeights
    cfg2 = PipelineConfig(input=vid, output=out, num_frames=5, min_frame_gap=3, quality=absq,
                          weights=QualityWeights(size=0.0, blur=0.0, exposure=0.0, occlusion=0.0,
                                                 view=0.0, det_conf=0.0, visibility=1.0))
    run_reselect(cfg2)
    doc2 = json.loads((out / "frame_scores.json").read_text())
    s1 = next(s for s in doc2["scores"] if s["frame"] == 3)
    assert s1["extra"]["score_original"] == 0.8 and s1["score"] == pytest.approx(0.8)
    # all components equal -> any weights give the same score; use unequal component to prove change
    sc = [s for s in doc2["scores"] if s["frame"] == 3][0]
    from horse_reid.quality.scorer import rescore
    fs = FrameScore.from_dict(sc)
    fs.size = 0.0
    before = rescore(fs, PipelineConfig().quality, PipelineConfig().weights).score
    after = rescore(fs, PipelineConfig().quality, cfg2.weights).score
    assert before != pytest.approx(after)

    # adaptive mode (default): blur is re-normalised by the run median; with a raised
    # min_score nothing passes -> low-quality fallback, flagged everywhere
    cfg3 = PipelineConfig(input=vid, output=out, num_frames=5, min_frame_gap=3, min_score=0.99)
    res3 = run_reselect(cfg3)
    doc3 = json.loads((out / "frame_scores.json").read_text())
    txt3 = (out / "summary.txt").read_text()
    assert res3["selected"] == 5 and res3["low_quality_selection"] is True
    assert "only 0 of 11 candidates" in res3["low_quality_reason"]
    assert doc3["low_quality_selection"] is True and doc3["summary"]["low_quality_selection"] is True
    assert doc3["config"]["blur_ref_used"] == pytest.approx(241.4, abs=0.1)  # median blur_var (q=0.8)
    assert "quality warning: low-quality selection: only 0 of 11" in txt3
    assert "(adaptive)" in txt3

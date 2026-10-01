"""Phase 2 (white-marking) unit tests. No weights needed except the skipped SAM test."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from horse_reid.marking import (
    AdaptiveColorSegmenter,
    FaceRegionExtractor,
    MarkingResult,
    available_marking_segmenters,
    build_marking_segmenter,
)
from horse_reid.marking.adaptive_color import AdaptiveColorParams
from horse_reid.marking.aihub import convert_to_yolo_seg, discover_categories

ROOT = Path(__file__).resolve().parents[1]
SAM_WEIGHTS = ROOT / "models/mobile_sam.pt"

BLOB_C, BLOB_R = (60, 60), 7  # 14-px white blob


def _synthetic_face(seed: int = 0) -> np.ndarray:
    """200x200 dark-brown coat + noise, white blob, thin bright strap, bright green square."""
    rng = np.random.default_rng(seed)
    img = np.empty((200, 200, 3), np.float32)
    img[:] = (25, 40, 70)  # BGR dark brown
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.circle(img, BLOB_C, BLOB_R, (240, 240, 240), -1)
    img[150:154, 50:140] = (235, 235, 240)  # 4 px wide, 90 px long strap
    img[30:50, 140:160] = (40, 220, 40)      # green tag
    return img


def _blob_found(res: MarkingResult) -> bool:
    return bool(res.mask[BLOB_C[1], BLOB_C[0]]) and any(
        c["bbox"][0] <= BLOB_C[0] < c["bbox"][2] and c["bbox"][1] <= BLOB_C[1] < c["bbox"][3]
        for c in res.components)


def test_adaptive_color_blob_strap_tag() -> None:
    img = _synthetic_face()
    res = AdaptiveColorSegmenter().predict(img)
    assert res.mask.shape == img.shape[:2] and res.mask.dtype == np.uint8
    assert set(np.unique(res.mask)) <= {0, 1}
    assert res.prob.dtype == np.float32
    assert float(res.prob.min()) >= 0.0 and float(res.prob.max()) <= 1.0
    assert len(res.components) == 1 and _blob_found(res)
    reasons = {e["reason"]: e for e in res.excluded}
    assert "strap_shape" in reasons and reasons["strap_shape"]["bbox"][1] >= 145
    assert "high_chroma" in reasons and reasons["high_chroma"]["bbox"][0] >= 135
    assert res.mask[152, 95] == 0 and res.mask[40, 150] == 0
    st = res.stats
    assert st["n_components"] == 1 and 0 < st["marking_area_frac"] < 0.01
    assert {"coat_L_median", "coat_L_mad"} <= set(st)


def test_local_illumination_gradient_blob() -> None:
    rng = np.random.default_rng(1)
    base = np.full((200, 200, 3), (25, 40, 70), np.float32) + rng.normal(0, 4, (200, 200, 3))
    lab = cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] += np.linspace(0, 60 * 255 / 100, 200, dtype=np.float32)[None, :]  # +60 L left->right
    img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    cv2.circle(img, (40, 100), 5, (240, 240, 240), -1)  # 10-px blob in the dark half
    res = AdaptiveColorSegmenter(AdaptiveColorParams(local_illumination=True)).predict(img)
    assert res.stats["illumination"] == "local" and res.debug["illumination"].shape == img.shape[:2]
    assert any(c["bbox"][0] <= 40 < c["bbox"][2] and c["bbox"][1] <= 100 < c["bbox"][3] for c in res.components)
    assert res.mask[100, 40]
    res_g = AdaptiveColorSegmenter(AdaptiveColorParams(local_illumination=False)).predict(img)
    assert res_g.stats["illumination"] == "global"


def test_adaptive_color_respects_face_mask() -> None:
    img = _synthetic_face()
    fm = np.zeros(img.shape[:2], bool)
    fm[100:, :] = True  # blob (y=60) outside the face
    res = AdaptiveColorSegmenter().predict(img, fm)
    assert res.mask[~fm].sum() == 0 and float(res.prob[~fm].max()) == 0.0
    assert not res.components


def test_registry() -> None:
    names = available_marking_segmenters()
    assert names[0] == "sam_refined" and {"adaptive_color", "yolo_seg"} <= set(names)
    assert build_marking_segmenter("adaptive_color").name == "adaptive_color"
    assert build_marking_segmenter("sam_refined").name == "sam_refined"  # SAM is lazy-loaded
    with pytest.raises(FileNotFoundError, match="aihub_dataset.md"):
        build_marking_segmenter("yolo_seg", weights_path="models/does_not_exist.pt")
    with pytest.raises(ValueError):
        build_marking_segmenter("nope")


def test_face_region_ellipse_fallback() -> None:
    frame = np.full((400, 600, 3), 90, np.uint8)
    ext = FaceRegionExtractor(seg_model_path=None, upscale=2.0)
    head = (200, 100, 300, 220)  # 100 x 120
    region = ext.extract(frame, (150, 80, 450, 380), head, margin=0.15)
    crop_img, fm, cb = region
    assert region.face_mask_source == "ellipse_fallback"
    assert cb == (185, 82, 315, 238)
    assert region.upscale == 2.0
    assert crop_img.shape[:2] == fm.shape == (2 * (cb[3] - cb[1]), 2 * (cb[2] - cb[0]))
    assert fm.dtype == bool
    h, w = fm.shape
    assert fm[h // 2, w // 2] and not fm[0, 0] and not fm[-1, -1]
    # inscribed ellipse of the head box: area ~ pi/4 of the (upscaled) head box
    assert abs(fm.sum() / (np.pi / 4 * 200 * 240) - 1) < 0.05


def test_aihub_converter(tmp_path: Path) -> None:
    # Synthetic fixture for converter logic only, not real AI-Hub data.
    jdir, idir, odir = tmp_path / "labels", tmp_path / "images", tmp_path / "yolo"
    jdir.mkdir(); idir.mkdir()
    cv2.imwrite(str(idir / "horse_0001.jpg"), np.zeros((100, 200, 3), np.uint8))
    fixture = {
        "horse_id": "fixture-0001", "horse_breed": "fixture", "horse_birth": "", "horse_sex": "",
        "horse_purpose": "",
        "image": {"file_name": "horse_0001.jpg", "file_path": "x/y",
                  "image_resolution": {"width": 200, "height": 100}},
        "horse_object": {
            "polygon": [
                {"category": "fixture_whitespot", "coor": [[20, 10], [60, 10], [60, 50], [20, 50]]},
                {"category": "fixture_whitespot", "coor": [100, 20, 140, 20, 120, 80]},  # flat list
                {"category": "fixture_other", "coor": [[0, 0], [10, 0], [10, 10]]},
                {"category": "fixture_broken", "coor": [[1, 2]]},
            ],
            "head_whitespot_shape": "", "leg_whitespot_shape": "", "horse_color": "",
        },
    }
    (jdir / "horse_0001.json").write_text(json.dumps(fixture))
    cats = discover_categories(jdir)
    assert cats == {"fixture_whitespot": 2, "fixture_other": 1}
    stats = convert_to_yolo_seg(jdir, idir, odir, {"fixture_whitespot": 0})
    labels = list((odir / "labels").rglob("*.txt"))
    assert len(labels) == 1 and stats["polygons"] == 2
    rows = labels[0].read_text().strip().splitlines()
    assert len(rows) == 2
    for row in rows:
        vals = row.split()
        assert vals[0] == "0" and len(vals[1:]) % 2 == 0
        coords = np.array(vals[1:], float)
        assert ((coords >= 0) & (coords <= 1)).all()
    assert np.allclose(np.array(rows[0].split()[1:], float), [0.1, 0.1, 0.3, 0.1, 0.3, 0.5, 0.1, 0.5])
    assert (odir / "data.yaml").exists() and (odir / "images" / "train" / "horse_0001.jpg").exists()


@pytest.mark.skipif(not SAM_WEIGHTS.exists(), reason="models/mobile_sam.pt not available")
def test_sam_refined_synthetic() -> None:
    from horse_reid.marking import SamRefinedSegmenter

    img = _synthetic_face()
    res = SamRefinedSegmenter(sam_weights=SAM_WEIGHTS).predict(img)
    assert "sam_error" not in res.stats
    assert _blob_found(res)
    assert res.mask.sum() < 0.20 * res.face_mask.sum()
    assert res.mask[152, 95] == 0 and res.mask[40, 150] == 0
    assert float(res.prob.min()) >= 0.0 and float(res.prob.max()) <= 1.0


def test_component_gates_solidity_edge_eye() -> None:
    rng = np.random.default_rng(3)
    img = np.clip(np.full((200, 200, 3), (25, 40, 70), np.float32) + rng.normal(0, 5, (200, 200, 3)), 0, 255).astype(np.uint8)
    white = (240, 240, 240)
    cv2.circle(img, (100, 100), 5, white, -1)                 # 10-px solid blob: kept
    img[110:113, 40:70] = white; img[96:127, 53:56] = white   # thin cross (low solidity)
    img[20:26, 100:106] = white                                # fragment on the mask edge
    cv2.circle(img, (140, 60), 4, white, -1)                  # eye glint
    fm = np.zeros((200, 200), bool)
    fm[20:180, 20:180] = True
    kps = {"eye": [[140.0, 60.0, 0.9]], "ear": [[30.0, 30.0, 0.9]]}  # far from every component
    res = AdaptiveColorSegmenter().predict(img, fm, kps)
    assert len(res.components) == 1
    c = res.components[0]
    assert c["bbox"][0] <= 100 < c["bbox"][2] and c["bbox"][1] <= 100 < c["bbox"][3]
    by_reason = res.stats["n_excluded_by_reason"]
    assert by_reason.get("low_solidity") == 1, by_reason
    assert by_reason.get("edge_fragment") == 1, by_reason
    assert by_reason.get("eye_glint") == 1, by_reason
    assert res.mask[100, 100] and not res.mask[60, 140] and not res.mask[22, 103]
    # without eye keypoints the glint is accepted
    res2 = AdaptiveColorSegmenter().predict(img, fm)
    assert len(res2.components) == 2 and "eye_glint" not in res2.stats["n_excluded_by_reason"]


def _flat_face(blob_gray: int = 240, halo: bool = False, centers=((100, 100),)) -> np.ndarray:
    rng = np.random.default_rng(5)
    img = np.clip(np.full((200, 200, 3), (25, 40, 70), np.float32) + rng.normal(0, 5, (200, 200, 3)), 0, 255).astype(np.uint8)
    for cx, cy in centers:
        if halo:
            cv2.circle(img, (cx, cy), 10, (36, 64, 104), -1)  # lit brown, below the brightness threshold
        cv2.circle(img, (cx, cy), 6, (blob_gray,) * 3, -1)
    return img


_FM = np.zeros((200, 200), bool)
_FM[20:180, 20:180] = True


def test_above_ears_and_ear_zone() -> None:
    img = _flat_face(centers=((60, 60), (140, 140)))
    kps = {"ear": [[100.0, 100.0, 0.9], [100.0, 90.0, 0.9]]}  # lowest ear y = 100 -> line 100 + 3.2
    res = AdaptiveColorSegmenter().predict(img, _FM, kps)
    assert res.stats["n_excluded_by_reason"].get("above_ears") == 1, res.stats
    assert not res.mask[60, 60] and res.mask[140, 140]
    # no ear keypoints: both kept
    assert len(AdaptiveColorSegmenter().predict(img, _FM).components) == 2
    # ear zone: blob below the ear line but within 0.10 x 160 = 16 px of an ear keypoint
    img2 = _flat_face(centers=((100, 120), (140, 150)))
    res2 = AdaptiveColorSegmenter().predict(img2, _FM, {"ear": [[100.0, 105.0, 0.9]]})
    assert res2.stats["n_excluded_by_reason"].get("above_ears") == 1, res2.stats
    assert not res2.mask[120, 100] and res2.mask[150, 140]


def test_blown_highlight() -> None:
    res = AdaptiveColorSegmenter().predict(_flat_face(255), _FM)
    assert res.stats["n_excluded_by_reason"].get("blown_highlight") == 1 and not res.components
    res = AdaptiveColorSegmenter().predict(_flat_face(215), _FM)
    assert len(res.components) == 1 and "blown_highlight" not in res.stats["n_excluded_by_reason"]
    # bright core in a strongly coloured lit halo -> specular on coat
    res = AdaptiveColorSegmenter().predict(_flat_face(215, halo=True), _FM)
    assert res.stats["n_excluded_by_reason"].get("blown_highlight") == 1, (res.stats, res.excluded)


def test_outside_landmark_hull() -> None:
    img = _flat_face(centers=((100, 100), (30, 160)))
    # hull: eye/eye/nose triangle around (100,100); face width 160 -> margin ~19 px
    kps = {"eye": [[80.0, 80.0, 0.9], [120.0, 80.0, 0.9]], "nose": [[100.0, 130.0, 0.9]]}
    res = AdaptiveColorSegmenter().predict(img, _FM, kps)
    assert res.stats["n_excluded_by_reason"].get("outside_landmark_hull") == 1, res.stats
    assert res.mask[100, 100] and not res.mask[160, 30]
    # <3 keypoints: no hull gate
    res2 = AdaptiveColorSegmenter().predict(img, _FM, {"eye": [[80.0, 80.0, 0.9]], "nose": [[100.0, 130.0, 0.9]]})
    assert "outside_landmark_hull" not in res2.stats["n_excluded_by_reason"] and len(res2.components) == 2

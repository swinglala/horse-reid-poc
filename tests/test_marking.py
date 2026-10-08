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


def _coat_img(bgr: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((200, 200, 3), np.float32)
    img[:] = bgr
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_coat_class_and_applicability() -> None:
    from horse_reid.marking_pipeline import assess_applicability

    light = AdaptiveColorSegmenter().predict(_coat_img((215, 215, 215)))
    dark = AdaptiveColorSegmenter().predict(_synthetic_face())
    assert light.stats["coat_L_median"] >= 65 and light.stats["coat_class"] == "light"
    assert dark.stats["coat_L_median"] < 45 and dark.stats["coat_class"] == "dark"
    a = assess_applicability([{"stats": light.stats}] * 3 + [{"stats": dark.stats}])
    assert a["white_marking_applicable"] is False and "3/4" in a["reason"]
    assert a["coat_class_counts"] == {"light": 3, "dark": 1}
    b = assess_applicability([{"stats": dark.stats}] * 3 + [{"stats": light.stats}])
    assert b["white_marking_applicable"] is True and b["reason"] is None


# ---------------------------------------------------------------- stripe / strong gates


def _coat(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.full((200, 200, 3), (25, 40, 70), np.float32) + rng.normal(0, 5, (200, 200, 3)),
                   0, 255).astype(np.uint8)


_FM = np.zeros((200, 200), bool)
_FM[10:190, 20:180] = True
_KPS_UPRIGHT = {"left_eye": [60.0, 40.0], "right_eye": [140.0, 40.0], "nose": [100.0, 180.0],
                "left_ear": [50.0, 15.0], "right_ear": [150.0, 15.0]}


def test_face_axis_and_region_axis_angle() -> None:
    from horse_reid.marking.adaptive_color import face_axis, region_axis_angle

    assert face_axis(_KPS_UPRIGHT) == pytest.approx((0.0, 1.0))
    assert face_axis(None) == (0.0, 1.0)                       # fallback: upright crop
    assert face_axis({"nose": [100.0, 10.0], "left_eye": [100.0, 110.0]}) == pytest.approx((0.0, -1.0))
    vert = np.zeros((200, 200), bool); vert[60:150, 98:103] = True
    horz = np.zeros((200, 200), bool); horz[160:164, 40:140] = True
    assert region_axis_angle(vert, (0.0, 1.0)) < 5.0
    assert region_axis_angle(horz, (0.0, 1.0)) > 85.0
    assert region_axis_angle(horz, (1.0, 0.0)) < 5.0


def test_stripe_along_face_axis_is_not_a_strap() -> None:
    """A thin, strongly white strip along the ear/eye -> nose axis is a blaze and is kept;
    the same strip across the face is a halter strap."""
    img = _coat()
    img[60:150, 98:103] = (240, 240, 240)   # along the axis (blaze)
    img[160:164, 40:140] = (240, 240, 240)  # across the axis (noseband)
    res = AdaptiveColorSegmenter().predict(img, _FM, _KPS_UPRIGHT)
    assert len(res.components) == 1
    c = res.components[0]
    assert c["stripe"] is True and c["axis_deg"] < 5.0 and c["bbox"] == [98, 60, 103, 150]
    straps = [e for e in res.excluded if e["reason"] == "strap_shape"]
    assert len(straps) == 1 and straps[0]["bbox"] == [40, 160, 140, 164] and straps[0]["axis_deg"] > 85.0
    assert res.mask[100, 100] and not res.mask[162, 90]

    # Rotate the face axis (eyes left, nose right): the roles swap.
    kps = {"left_eye": [40.0, 60.0], "right_eye": [40.0, 140.0], "nose": [180.0, 100.0]}
    img = _coat()
    img[98:103, 60:150] = (240, 240, 240)
    img[40:140, 160:164] = (240, 240, 240)
    res = AdaptiveColorSegmenter().predict(img, _FM, kps)
    assert [c["bbox"] for c in res.components] == [[60, 98, 150, 103]]
    assert [e["bbox"] for e in res.excluded if e["reason"] == "strap_shape"] == [[160, 40, 164, 140]]

    # Without keypoints the axis falls back to vertical: horizontal strap still rejected.
    res = AdaptiveColorSegmenter().predict(img, _FM)
    assert "strap_shape" in res.stats["n_excluded_by_reason"]

    # A mid-brightness strip along the axis (sunlit sheen on the nasal bone) is not strong -> still a strap.
    img = _coat()
    img[60:150, 98:103] = (120, 120, 120)
    res = AdaptiveColorSegmenter().predict(img, _FM, _KPS_UPRIGHT)
    [e] = res.excluded
    assert e["reason"] == "strap_shape" and e["stripe"] is False and e["mean_z"] < 6.5


def test_strong_white_region_skips_solidity_and_exposure_gates() -> None:
    """A large, strongly white region (a sunlit blaze) may be concave and sensor-clipped;
    the same shape at mid brightness is still rejected as low_solidity."""
    def l_shape(col: tuple[int, int, int]) -> np.ndarray:
        img = _coat()
        img[60:150, 85:93] = col
        img[142:150, 85:145] = col
        return img

    res = AdaptiveColorSegmenter().predict(l_shape((255, 255, 255)), _FM, _KPS_UPRIGHT)
    assert len(res.components) == 1, res.stats["n_excluded_by_reason"]
    c = res.components[0]
    assert c["strong"] is True and c["solidity"] < 0.55 and c["clip_frac"] > 0.9
    assert c["area_frac_of_face"] >= 0.01 and c["mean_z"] >= 6.5
    assert "blown_highlight" not in res.stats["n_excluded_by_reason"]

    res = AdaptiveColorSegmenter().predict(l_shape((120, 120, 120)), _FM, _KPS_UPRIGHT)
    assert res.components == []
    [e] = [e for e in res.excluded if e["area_px"] > 500]
    assert e["reason"] == "low_solidity" and e["strong"] is False and e["mean_z"] < 6.5


def test_face_mask_not_clipped_by_narrow_horse_box() -> None:
    """The tracker's horse box can be narrower than the head (handler next to the face);
    the seg mask must still cover the whole head crop."""
    class FakeSeg:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int]] = []

        def horse_mask(self, frame: np.ndarray, hb: tuple[int, int, int, int]) -> np.ndarray:
            self.calls.append(hb)
            return np.ones((hb[3] - hb[1], hb[2] - hb[0]), bool)

    frame = np.zeros((300, 400, 3), np.uint8)
    ext = FaceRegionExtractor(seg_model_path="unused.pt", upscale=1.0)
    ext._seg = FakeSeg()
    horse_bbox = (50, 50, 220, 250)       # right edge x=220 ...
    head_bbox = (180, 60, 260, 200)       # ... but the head extends to x=260
    region = ext.extract(frame, horse_bbox, head_bbox, margin=0.0)
    assert region.face_mask_source == "seg"
    assert region.crop_bbox == (180, 60, 260, 200)
    (hb,) = ext._seg.calls
    assert hb[0] <= 50 and hb[2] >= 260 and hb[1] <= 50 and hb[3] >= 250
    # The whole crop is horse pixels (fake mask is all-ones) - nothing is cut off at x=220.
    assert region.face_mask.shape == (140, 80) and region.face_mask[:, 45:].mean() > 0.9


def test_face_mask_extended_to_landmark_hull() -> None:
    """A seg mask that stops at the noseband is extended down to the nose landmark."""
    class TopHalfSeg:
        def horse_mask(self, frame: np.ndarray, hb: tuple[int, int, int, int]) -> np.ndarray:
            m = np.zeros((hb[3] - hb[1], hb[2] - hb[0]), bool)
            m[: m.shape[0] // 2] = True            # upper half of the box only
            return m

    frame = np.zeros((300, 400, 3), np.uint8)
    ext = FaceRegionExtractor(seg_model_path="unused.pt", upscale=1.0)
    ext._seg = TopHalfSeg()
    head = (100, 50, 200, 250)
    kps = {"left_ear": [[120.0, 60.0, 0.9]], "right_ear": [[180.0, 60.0, 0.9]],
           "left_eye": [[125.0, 110.0, 0.9]], "right_eye": [[175.0, 110.0, 0.9]],
           "nose": [[150.0, 240.0, 0.9]]}
    base = ext.extract(frame, head, head, margin=0.0)
    assert base.face_mask_source == "seg" and not base.face_mask[190, 50]
    region = ext.extract(frame, head, head, margin=0.0, keypoints=kps)
    assert region.face_mask_source == "seg+landmark_hull"
    assert 0.0 < region.hull_added_frac < 1.0
    assert region.coat_mask is not None and region.coat_mask[50, 50] and not region.coat_mask[190, 50]
    assert base.coat_mask is None
    assert region.face_mask[190, 50]          # muzzle (nose landmark y=240 -> crop y=190) now inside
    assert region.face_mask[:95].all()        # seg part kept (bottom 2 px eroded)
    assert not region.face_mask[190, 2]       # hull does not spill to the crop corner
    # no seg at all: the hull alone becomes the face mask
    ext._seg_failed = True
    region = ext.extract(frame, head, head, margin=0.0, keypoints=kps)
    assert region.face_mask_source == "landmark_hull" and region.hull_added_frac == 1.0


def test_coat_mask_keeps_reference_statistics() -> None:
    """A large white area inside the hull extension must not lift the coat reference."""
    img = _coat()
    fm = _FM.copy()
    img[150:185, 40:160] = (250, 250, 250)        # white muzzle-like block (~23% of the face mask)
    coat = fm.copy(); coat[150:] = False          # seg part = everything above it
    plain = AdaptiveColorSegmenter().predict(_coat(), fm, _KPS_UPRIGHT)
    with_cm = AdaptiveColorSegmenter().predict(img, fm, _KPS_UPRIGHT, coat_mask=coat)
    no_cm = AdaptiveColorSegmenter().predict(img, fm, _KPS_UPRIGHT)
    assert with_cm.stats["coat_ref_frac"] < 0.9 and no_cm.stats["coat_ref_frac"] == 1.0
    assert abs(with_cm.stats["coat_L_median"] - plain.stats["coat_L_median"]) < 1.0
    big = lambda r: [c for c in r.components + r.excluded if c["area_px"] > 1000][0]
    # Without coat_mask the white block lifts the local illumination field and its own z drops.
    assert big(no_cm)["mean_z"] < big(with_cm)["mean_z"] - 1.0
    assert big(with_cm)["mean_z"] >= 6.5 and big(with_cm)["strong"] is True

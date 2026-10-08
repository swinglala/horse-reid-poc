"""Phase 3 canonical mapping tests (synthetic data only; no weights, no video)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from horse_reid.canonical.base import CanonicalObservation, aggregate_support, find_peaks
from horse_reid.canonical.fourdequine import (
    FourDEquineMapper,
    FourDEquineNotAvailable,
    aggregate_vertices,
    project_vertices,
    sample_mask_at_vertices,
    vertex_visibility,
    vertices_to_uv_image,
)
from horse_reid.canonical.io import load_joined
from horse_reid.canonical.planar import (
    CANONICAL_H,
    CANONICAL_LANDMARKS,
    CANONICAL_W,
    PlanarCanonicalMapper,
    estimate_similarity,
    visibility_profile,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _canon_to_img(pts, s=0.5, theta_deg=10.0, t=(40.0, 20.0)):
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (np.asarray(pts, float) @ R.T) * s + np.asarray(t)


def _kps_from_canon(names, **kw):
    pts = _canon_to_img([CANONICAL_LANDMARKS[n] for n in names], **kw)
    kps: dict[str, list] = {"eye": [], "nose": [], "ear": []}
    for n, (x, y) in zip(names, pts):
        part = "eye" if "eye" in n else "nose" if n == "nose" else "ear"
        kps[part].append([float(x), float(y), 0.9])
    kps["eye"] = kps["eye"][::-1]  # order must not matter
    return kps


def _blob_mask(center, radius, shape=(200, 200)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (int(round(center[0])), int(round(center[1]))), radius, 255, -1)
    return m


def _centroid(p):
    ys, xs = np.nonzero(p > 0.5)
    return float(xs.mean()), float(ys.mean())


ALL = ["left_eye", "right_eye", "nose", "left_ear_base", "right_ear_base"]


# --------------------------------------------------------------------------- #
# planar
# --------------------------------------------------------------------------- #
def test_similarity_recovers_known_transform():
    src = np.array([[0, 0], [10, 0], [0, 20], [7, 3]], float)
    th = np.radians(25)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = src @ R.T * 1.7 + [5, -3]
    M, res = estimate_similarity(src, dst)
    assert res < 1e-9
    np.testing.assert_allclose(M[:, :2], 1.7 * R, atol=1e-9)


def test_planar_blob_lands_at_expected_canonical_location():
    target = (128.0, 200.0)
    blob_img = _canon_to_img([target])[0]
    mask = _blob_mask(blob_img, 6)
    m = PlanarCanonicalMapper()
    obs = m.map_frame(mask, None, _kps_from_canon(ALL), yaw=0.0, quality=1.0, frame=7)
    assert obs is not None and obs.prob.shape == (CANONICAL_H, CANONICAL_W)
    cx, cy = _centroid(obs.prob)
    assert abs(cx - target[0]) < 6 and abs(cy - target[1]) < 6
    assert obs.meta["residual"] < 1e-6
    assert set(obs.meta["landmarks"]) == set(ALL)
    assert obs.frame == 7 and obs.view == "frontal"


def test_planar_eye_nose_only_and_uint8_prob():
    target = (100.0, 160.0)
    mask = _blob_mask(_canon_to_img([target])[0], 6)
    obs = PlanarCanonicalMapper().map_frame(mask, mask.copy(), _kps_from_canon(["left_eye", "right_eye", "nose"]),
                                            yaw=10.0, quality=0.8)
    cx, cy = _centroid(obs.prob)
    assert abs(cx - target[0]) < 6 and abs(cy - target[1]) < 6
    assert obs.weight.max() == pytest.approx(0.8)


def test_single_eye_profile_assignment_follows_yaw_sign():
    m = PlanarCanonicalMapper()
    # horse faces image-left (yaw<0): its anatomical LEFT eye = canonical right_eye
    kps = _kps_from_canon(["right_eye", "nose"])
    tr = m.estimate_transform(kps, yaw=-80.0)
    assert "right_eye" in tr["landmarks"] and "left_eye" not in tr["landmarks"]
    tr = m.estimate_transform(_kps_from_canon(["left_eye", "nose"]), yaw=80.0)
    assert "left_eye" in tr["landmarks"]


def test_single_eye_ambiguous_uses_residual():
    # 3/4 view, one eye + nose + both ears -> residual decides; the true eye is left_eye
    # even though the yaw prior (yaw<0 -> right) and eye-vs-nose prior disagree with nothing here.
    kps = _kps_from_canon(["left_eye", "nose", "left_ear_base", "right_ear_base"])
    tr = PlanarCanonicalMapper().estimate_transform(kps, yaw=-45.0)
    assert "left_eye" in tr["landmarks"] and tr["residual"] < 1e-6
    assert tr["n_candidates"] == 2


def test_duplicate_eye_detections_are_merged():
    kps = _kps_from_canon(["right_eye", "nose", "left_ear_base", "right_ear_base"])
    e = kps["eye"][0]
    kps["eye"].append([e[0] + 4.0, e[1] + 2.0, 0.3])  # second box on the same eye
    tr = PlanarCanonicalMapper().estimate_transform(kps, yaw=-70.0)
    assert tr["landmarks"].count("right_eye") == 1 and "left_eye" not in tr["landmarks"]
    assert tr["residual"] < 1e-6


def test_leave_one_out_drops_spurious_landmark():
    kps = _kps_from_canon(ALL)
    kps["nose"][0][0] += 80.0  # wildly wrong nose
    tr = PlanarCanonicalMapper().estimate_transform(kps, yaw=0.0)
    assert "nose" not in tr["landmarks"] and tr["dropped"] == 1 and tr["residual"] < 1e-6


def test_near_side_from_single_eye_overrides_yaw_sign():
    # yaw sign says "faces image-left" but the only eye fits left_eye -> near half = canonical left
    kps = _kps_from_canon(["left_eye", "nose", "left_ear_base", "right_ear_base"])
    obs = PlanarCanonicalMapper().map_frame(np.full((200, 200), 255, np.uint8), None, kps, yaw=-50.0,
                                            quality=1.0)
    assert obs.meta["near_side_source"] == "single_eye" and obs.meta["visibility_yaw"] == 50.0
    valid = obs.weight > 0
    assert np.all(obs.weight[:, :128][valid[:, :128]] == pytest.approx(1.0))
    assert np.all(obs.weight[:, 128:][valid[:, 128:]] == pytest.approx(0.4))


def test_single_landmark_fallback_uses_bbox_scale():
    kps = {"nose": [[50.0, 90.0, 0.9]]}
    obs = PlanarCanonicalMapper().map_frame(_blob_mask((50, 80), 4, (100, 100)), None, kps, yaw=0.0,
                                            quality=1.0, head_bbox_px=[0, 0, 100, 100])
    assert obs is not None and obs.meta["fallback"]
    assert obs.weight.max() == pytest.approx(0.3)
    assert obs.meta["scale"] == pytest.approx(CANONICAL_H / 100)
    assert PlanarCanonicalMapper().map_frame(np.zeros((50, 50), np.uint8), None, {}, 0.0, 1.0) is None


def test_visibility_profile_bins():
    np.testing.assert_array_equal(visibility_profile(20.0), 1.0)
    p = visibility_profile(-45.0)  # faces image-left -> far half = canonical x < 128
    assert np.all(p[:, :128] == 0.4) and np.all(p[:, 128:] == 1.0)
    p = visibility_profile(80.0)   # faces image-right -> far half = canonical x >= 128
    assert np.all(p[:, 141:] == 0.0) and np.all(p[:, :115] == 1.0)
    assert np.all(p[:, 117:139] == 0.5)


def test_profile_weight_map_zeroes_far_half():
    kps = _kps_from_canon(["right_eye", "nose"])
    mask = np.full((200, 200), 255, np.uint8)
    obs = PlanarCanonicalMapper().map_frame(mask, None, kps, yaw=-80.0, quality=1.0)
    assert obs.view == "profile"
    assert np.all(obs.weight[:, :115] == 0)
    assert obs.weight[:, 141:].max() > 0
    # warped-outside pixels have weight 0
    assert obs.weight[0, -1] == 0 or obs.prob[0, -1] > 0


def _obs(p, w, frame, yaw=0.0):
    return CanonicalObservation(frame=frame, yaw=yaw, view="frontal", prob=p.astype(np.float32),
                                weight=w.astype(np.float32), quality=1.0)


def test_aggregate_disjoint_weights():
    H, W = CANONICAL_H, CANONICAL_W
    w1 = np.zeros((H, W)); w1[:, :100] = 1
    w2 = np.zeros((H, W)); w2[:, 150:200] = 1
    ref = PlanarCanonicalMapper().aggregate([_obs(np.full((H, W), 0.8), w1, 1),
                                             _obs(np.full((H, W), 0.2), w2, 2, yaw=-70.0)])
    np.testing.assert_allclose(ref.prob[:, :100], 0.8, atol=1e-6)
    np.testing.assert_allclose(ref.prob[:, 150:200], 0.2, atol=1e-6)
    assert np.all(ref.coverage[:, 100:150] == 0) and np.all(ref.coverage[:, 200:] == 0)
    assert np.all(ref.coverage[:, :100] > 0) and np.all(ref.coverage[:, 150:200] > 0)
    assert np.all(ref.mask[:, :100] == 1) and ref.mask[:, 100:].sum() == 0
    np.testing.assert_allclose(ref.consistency[:, :100], 1.0, atol=1e-5)
    assert ref.view_count == 2 and ref.frames == [1, 2]
    assert ref.views == {"frontal": 2}


def test_aggregate_mask_threshold_and_consistency():
    H, W = CANONICAL_H, CANONICAL_W
    p = np.full((H, W), 0.9); p[:, 200:] = 0.5
    w = np.ones((H, W)); w[:, :50] = 0.1
    ref = PlanarCanonicalMapper().aggregate([_obs(p, w, 0)])
    assert ref.mask[:, :50].sum() == 0          # coverage 0.1 <= 0.15
    assert np.all(ref.mask[:, 50:200] == 1)
    assert ref.mask[:, 200:].sum() == 0         # prob 0.5 is not > 0.5
    a = np.zeros((H, W)); b = np.ones((H, W))
    ref2 = PlanarCanonicalMapper().aggregate([_obs(a, np.ones((H, W)), 0), _obs(b, np.ones((H, W)), 1)])
    np.testing.assert_allclose(ref2.prob, 0.5, atol=1e-6)
    np.testing.assert_allclose(ref2.consistency, 0.5, atol=1e-5)  # 1 - std(0, 1)
    empty = PlanarCanonicalMapper().aggregate([])
    assert empty.view_count == 0 and empty.mask.sum() == 0


def _blob_obs(frame, with_blob, center=(150, 90), r=4, p_blob=0.9, p_bg=0.05):
    H, W = CANONICAL_H, CANONICAL_W
    p = np.full((H, W), p_bg, np.float32)
    m = np.zeros((H, W), np.float32)
    if with_blob:
        cv2.circle(p, center, r, p_blob, -1)
        cv2.circle(m, center, r, 1.0, -1)
    return CanonicalObservation(frame=frame, yaw=0.0, view="frontal", prob=p, weight=np.ones((H, W), np.float32),
                                quality=1.0, mask=m)


def _two_of_ten():
    return [_blob_obs(100 + i, i in (3, 7)) for i in range(10)]


def test_support_and_peaks_two_of_ten():
    ref = PlanarCanonicalMapper().aggregate(_two_of_ten())
    x, y = 150, 90
    assert ref.support[y, x] == pytest.approx(0.2, abs=1e-6)
    assert ref.n_support[y, x] == 2 and ref.n_support.dtype == np.int32
    assert ref.support[10, 10] == 0 and ref.n_support[10, 10] == 0
    assert ref.mask.sum() == 0                      # weighted prob 0.22 < 0.5: no invented mask
    assert ref.support_stack.shape == (10, CANONICAL_H, CANONICAL_W)
    peaks = find_peaks(ref.prob, ref.coverage, ref.support, min_support=0.02, min_distance_px=12, max_peaks=8,
                       peak_sigma=4.0, n_support=ref.n_support, support_stack=ref.support_stack,
                       frames=ref.frames)
    assert peaks, "the blob must be reported as a peak"
    top = peaks[0]
    assert abs(top["x"] - x) <= 4 and abs(top["y"] - y) <= 4
    assert top["n_support"] == 2 and top["frames"] == [103, 107]
    assert top["support"] == pytest.approx(0.2, abs=1e-3) and top["id"] == "p0"
    assert top["prob"] == pytest.approx(0.1 * 0.9 * 2 + 0.8 * 0.05, abs=1e-3)
    assert len(peaks) == 1 and 0.02 <= top["support_s"] <= 0.2
    assert find_peaks(ref.prob, ref.coverage, ref.support, min_support=0.5) == []


def test_peaks_require_accepted_masks_and_merge_misaligned_frames():
    H, W = CANONICAL_H, CANONICAL_W
    # strong prob everywhere-ish texture blob with NO accepted mask -> never a peak
    tex = _blob_obs(1, True, center=(60, 250), r=6, p_blob=0.45)
    tex.mask[:] = 0
    # two frames with small masks ~10 px apart (landmark misalignment) -> one peak, n=2
    a = _blob_obs(2, True, center=(120, 90), r=3)
    b = _blob_obs(3, True, center=(130, 90), r=3)
    others = [_blob_obs(10 + i, False) for i in range(7)]
    ref = PlanarCanonicalMapper().aggregate([tex, a, b] + others)
    assert ref.n_support.max() == 1  # the raw masks do not overlap
    peaks = find_peaks(ref.prob, ref.coverage, ref.support, peak_sigma=4.0, support_stack=ref.support_stack,
                       frames=ref.frames)
    assert len(peaks) == 1
    top = peaks[0]
    assert 118 <= top["x"] <= 132 and abs(top["y"] - 90) <= 3
    assert top["frames"] == [2, 3] and top["n_support"] == 2
    assert not any(abs(d["x"] - 60) < 10 and abs(d["y"] - 250) < 10 for d in peaks)
    assert find_peaks(ref.prob, ref.coverage, np.zeros((H, W), np.float32)) == []


def test_aggregate_support_weights_and_255_masks():
    H, W = 4, 4
    m1 = np.full((H, W), 255, np.uint8)
    sup, ns, stack = aggregate_support([m1, None], [np.full((H, W), 3.0), np.full((H, W), 1.0)])
    np.testing.assert_allclose(sup, 0.75)
    assert np.all(ns == 1) and stack.shape == (2, H, W)
    sup, ns, _ = aggregate_support([m1], [np.zeros((H, W))])  # unobserved -> no support counted
    assert sup.max() == 0 and ns.max() == 0


def test_planar_observation_mask_is_binary_float():
    target = (128.0, 200.0)
    mask = _blob_mask(_canon_to_img([target])[0], 6)
    obs = PlanarCanonicalMapper().map_frame(mask, None, _kps_from_canon(ALL), yaw=0.0, quality=1.0)
    assert obs.mask.dtype == np.float32 and set(np.unique(obs.mask)) <= {0.0, 1.0} and obs.mask.max() == 1.0


def test_ascii_max_pooling_and_peak_marker():
    from horse_reid.canonical.reference import ascii_marking

    ref = PlanarCanonicalMapper().aggregate(_two_of_ten())
    peaks = find_peaks(ref.prob, ref.coverage, ref.support, n_support=ref.n_support,
                       support_stack=ref.support_stack, frames=ref.frames)
    txt = ascii_marking(ref.prob, ref.coverage, peaks=peaks)
    rows = txt.splitlines()[2:2 + 40]
    r, c = int(peaks[0]["y"] * 40 / CANONICAL_H), int(peaks[0]["x"] * 32 / CANONICAL_W)
    assert rows[r][1 + c] in "*░▒▓█"
    assert "p0(" in txt and "n=2" in txt and "s=" in txt
    # mean pooling would erase the small peak; max pooling keeps it
    assert sum(ch in "░▒▓█*" for ch in "".join(rows)) >= 1


# --------------------------------------------------------------------------- #
# 4DEquine (model-free parts)
# --------------------------------------------------------------------------- #
def _cube():
    from scipy.spatial import ConvexHull

    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    F = ConvexHull(V).simplices.copy()
    for i, f in enumerate(F):  # orient outward
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if np.dot(n, V[f].mean(0)) < 0:
            F[i] = f[[0, 2, 1]]
    return V, F


def test_pinhole_projection_cube():
    V, _ = _cube()
    cam = {"K": [[100, 0, 50], [0, 100, 50], [0, 0, 1]], "R": np.eye(3), "t": [0, 0, 5]}
    uv = project_vertices(V, cam)
    for v, (u, w) in zip(V, uv):
        z = v[2] + 5
        assert u == pytest.approx(50 + 100 * v[0] / z) and w == pytest.approx(50 + 100 * v[1] / z)
    i = np.where((V == [1, 1, -1]).all(1))[0][0]
    np.testing.assert_allclose(uv[i], [75.0, 75.0])


def test_weak_perspective_projection():
    V, _ = _cube()
    uv = project_vertices(V, {"scale": 10.0, "tx": 50.0, "ty": 60.0})
    i = np.where((V == [1, 1, -1]).all(1))[0][0]
    np.testing.assert_allclose(uv[i], [60.0, 70.0])
    with pytest.raises(ValueError):
        project_vertices(V, {"foo": 1})


def test_visibility_by_normals():
    V, F = _cube()
    cam = {"K": np.eye(3), "R": np.eye(3), "t": [0, 0, 5]}  # camera at z=-5 looking +z
    vis = vertex_visibility(V, F, cam)
    np.testing.assert_array_equal(vis, V[:, 2] < 0)
    vis_w = vertex_visibility(V, F, {"scale": 1.0})
    np.testing.assert_array_equal(vis_w, V[:, 2] < 0)


def test_sample_mask_at_vertices():
    P = np.tile(np.arange(100, dtype=np.float32) / 100.0, (100, 1))
    v2d = np.array([[10.5, 20.0], [50.0, 50.0], [150.0, 5.0], [30.0, 30.0]])
    out = sample_mask_at_vertices(P, v2d, visible=np.array([True, True, True, False]))
    assert out[0] == pytest.approx(0.105, abs=1e-6) and out[1] == pytest.approx(0.5)
    assert np.isnan(out[2]) and np.isnan(out[3])
    out8 = sample_mask_at_vertices((P * 255).astype(np.uint8), v2d[:2])
    assert out8[1] == pytest.approx(127 / 255, abs=1e-3)


def test_aggregate_vertices_nan_is_unobserved():
    r = aggregate_vertices([(np.array([1.0, np.nan, 0.0]), np.array([1.0, 1.0, 1.0])),
                            (np.array([0.0, 0.8, np.nan]), np.array([1.0, 1.0, 1.0]))])
    np.testing.assert_allclose(r["prob"], [0.5, 0.8, 0.0], atol=1e-6)
    np.testing.assert_allclose(r["weight_sum"], [2.0, 1.0, 1.0])


def test_vertices_to_uv_image_fills():
    vals = np.array([0.0, 0.5, 1.0, np.nan])
    uv = np.array([[0, 0], [1, 1], [0, 1], [0.5, 0.5]])
    img = vertices_to_uv_image(vals, uv, 8)
    assert img.shape == (8, 8) and not np.isnan(img).any()
    assert img[7, 0] == 0.0 and img[0, 7] == 0.5 and img[0, 0] == 1.0  # flip_v: v=0 bottom
    lim = vertices_to_uv_image(vals, uv, 8, max_fill_dist=1.0)
    assert np.isnan(lim[4, 4])


def test_fourdequine_unavailable(tmp_path):
    m = FourDEquineMapper()
    assert m.name == "4dequine"
    with pytest.raises(FourDEquineNotAvailable):
        m.map_frame(np.zeros((10, 10), np.uint8), None, {}, 0.0, 1.0)
    with pytest.raises(FourDEquineNotAvailable, match="varen"):
        m.load_results(tmp_path / "refined_results.pt")
    with pytest.raises(FourDEquineNotAvailable):
        m.load_template(tmp_path / "varen.pkl")
    with pytest.raises(FourDEquineNotAvailable):
        m.head_vertex_indices()
    assert issubclass(FourDEquineNotAvailable, RuntimeError)


def test_fourdequine_observe_synthetic_cube():
    V, F = _cube()
    cam = {"scale": 20.0, "tx": 50.0, "ty": 50.0}
    prob = np.zeros((100, 100), np.float32); prob[:, 50:] = 1.0  # right half marked
    m = FourDEquineMapper()
    obs = m.observe(prob, V, F, cam, yaw=0.0, quality=1.0, frame=3)
    assert obs.prob.shape == (1, 8)
    front = V[:, 2] < 0
    assert np.all(obs.weight[0, ~front] == 0) and np.all(obs.weight[0, front] == 1)
    np.testing.assert_allclose(obs.prob[0, front], (V[front, 0] > 0).astype(float))
    ref = m.aggregate([obs])
    assert ref.mapper == "4dequine" and ref.mask[0, front].sum() == 2


# --------------------------------------------------------------------------- #
# io + reference
# --------------------------------------------------------------------------- #
def _write_fixture(root: Path, frames=((10, 0.0, ALL), (20, -45.0, ALL), (30, -80.0, ["right_eye", "nose"])),
                   crop=(100, 100), upscale=2.0, sim=dict(s=0.6, theta_deg=0.0, t=(23.0, 29.0))):
    (root / "marking" / "masks").mkdir(parents=True)
    (root / "marking" / "prob").mkdir(parents=True)
    scores, entries = [], []
    blob = _canon_to_img([(128.0, 200.0)], **sim)[0]
    cw, ch = 100, 125
    for fr, yaw, names in frames:
        pts_px = _canon_to_img([CANONICAL_LANDMARKS[n] for n in names], **sim)
        kps: dict[str, list] = {"eye": [], "nose": [], "ear": []}
        for n, (x, y) in zip(names, pts_px):
            part = "eye" if "eye" in n else "nose" if n == "nose" else "ear"
            kps[part].append([x / upscale + crop[0], y / upscale + crop[1], 0.8])
        scores.append({"frame": fr, "time_s": fr / 30, "yaw": yaw, "view": 1.0, "score": 0.9,
                       "head_bbox": [crop[0], crop[1], crop[0] + cw, crop[1] + ch],
                       "extra": {"keypoints": kps}})
        mask = _blob_mask(blob, 9, (int(ch * upscale), int(cw * upscale)))
        cv2.imwrite(str(root / "marking" / "masks" / f"frame_{fr:05d}.png"), mask)
        cv2.imwrite(str(root / "marking" / "prob" / f"frame_{fr:05d}.png"), (mask * 0.9).astype(np.uint8))
        entries.append({"frame": fr, "crop_bbox": [crop[0], crop[1], crop[0] + cw, crop[1] + ch],
                        "upscale": upscale, "stats": {}, "method": "synthetic"})
    # broken entries: no score / no mask / no crop_bbox
    scores.append({"frame": 40, "yaw": 0.0, "score": 0.5, "head_bbox": [0, 0, 10, 10], "extra": {}})
    entries += [{"frame": 99, "crop_bbox": [0, 0, 10, 10], "upscale": 1.0},
                {"frame": 40, "crop_bbox": [0, 0, 10, 10], "upscale": 1.0},
                {"frame": 10}]
    (root / "frame_scores.json").write_text(json.dumps({"selected": [f[0] for f in frames], "scores": scores}))
    (root / "marking" / "marking_results.json").write_text(json.dumps(entries))
    cv2.imwrite(str(root / "contact_sheet.jpg"), np.full((300, 600, 3), 200, np.uint8))


def test_io_joins_and_converts_keypoints(tmp_path):
    _write_fixture(tmp_path)
    joined = load_joined(tmp_path)
    assert [j.frame for j in joined] == [10, 20, 30]
    j = joined[0]
    assert j.mask.shape == (250, 200) and j.prob is not None and j.upscale == 2.0
    nose_px = _canon_to_img([CANONICAL_LANDMARKS["nose"]], s=0.6, theta_deg=0.0, t=(23.0, 29.0))[0]
    np.testing.assert_allclose(j.keypoints_px["nose"][0][:2], nose_px, atol=1e-6)
    np.testing.assert_allclose(j.head_bbox_px, [0, 0, 200, 250])
    assert j.selected


def test_io_dict_shaped_marking_results(tmp_path):
    _write_fixture(tmp_path)
    entries = json.loads((tmp_path / "marking" / "marking_results.json").read_text())
    (tmp_path / "marking" / "marking_results.json").write_text(json.dumps({"frames": entries}))
    assert len(load_joined(tmp_path)) == 3


def test_build_reference_end_to_end(tmp_path):
    from horse_reid.canonical.reference import build_reference

    _write_fixture(tmp_path)
    res = build_reference(tmp_path, None, mapper_name="planar", compute_embedding=False)
    assert set(res) == {"horse_id", "reference"} and res["horse_id"] == "unknown"
    r = res["reference"]
    for k in ("canonical_marking_mask", "canonical_marking_mask_b64", "face_embedding", "face_embedding_method",
              "view_count", "views", "frames", "mapper", "created_utc", "notes"):
        assert k in r, k
    assert r["view_count"] == 3 and r["frames"] == [10, 20, 30] and r["mapper"] == "planar"
    assert r["views"] == {"frontal": 1, "three_quarter": 1, "profile": 1}
    assert any("real observations" in n.lower() for n in r["notes"])
    ref_dir = tmp_path / "reference"
    on_disk = json.loads((ref_dir / "horse_reference.json").read_text())
    assert on_disk["reference"]["canonical_marking_mask"] == "reference/canonical_mask.png"
    mask = cv2.imread(str(tmp_path / on_disk["reference"]["canonical_marking_mask"]), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (CANONICAL_H, CANONICAL_W)
    cx, cy = _centroid(mask / 255.0)
    assert abs(cx - 128) < 6 and abs(cy - 200) < 6
    top = r["canonical_peaks"][0]
    assert mask[top["y"], top["x"]] == 255  # flat synthetic blob: the peak lies inside the canonical mask
    import base64
    dec = cv2.imdecode(np.frombuffer(base64.b64decode(r["canonical_marking_mask_b64"]), np.uint8),
                       cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(dec, mask)
    assert "HORSE MARKING" in (ref_dir / "canonical_marking.txt").read_text(encoding="utf-8")
    assert isinstance(on_disk["reference"]["canonical_peaks"], list)
    st = on_disk["reference"]["stats"]
    assert st["n_peaks"] == len(on_disk["reference"]["canonical_peaks"]) >= 1
    assert st["n_peaks_multi"] == sum(d["n_support"] >= 2 for d in on_disk["reference"]["canonical_peaks"])
    assert st["peak_sigma"] == 4.0 and st["min_support"] == 0.02
    assert on_disk["reference"]["params"]["peak_sigma"] == 4.0
    top = on_disk["reference"]["canonical_peaks"][0]
    assert top["n_support"] >= 1
    assert set(top["frames"]) <= {10, 20, 30} and top["frames"]
    sup = cv2.imread(str(ref_dir / "canonical_support.png"), cv2.IMREAD_GRAYSCALE)
    ns = cv2.imread(str(ref_dir / "canonical_nsupport.png"), cv2.IMREAD_GRAYSCALE)
    assert sup.shape == ns.shape == (CANONICAL_H, CANONICAL_W) and sup.max() > 0 and 1 <= ns.max() <= 3
    for name in ("canonical_prob.png", "canonical_coverage.png", "canonical_consistency.png",
                 "canonical_marking.png"):
        assert (ref_dir / name).exists(), name
    assert len(list((ref_dir / "per_frame_canonical").glob("frame_*.png"))) == 3
    rep = cv2.imread(str(tmp_path / "final_report.jpg"))
    assert rep is not None and rep.shape[1] == 1400


def test_build_reference_4dequine_unavailable(tmp_path):
    from horse_reid.canonical.reference import build_reference

    _write_fixture(tmp_path)
    with pytest.raises(FourDEquineNotAvailable):
        build_reference(tmp_path, None, mapper_name="4dequine", compute_embedding=False)


def test_draw_on_diagram_aligns_landmarks(tmp_path) -> None:
    """The affine fitted on the five landmarks maps the template landmarks onto the
    diagram landmarks and warps a star at the template forehead to the diagram forehead."""
    import json

    import cv2
    import numpy as np

    from horse_reid.canonical.diagram import draw_on_diagram, estimate_affine, load_diagram, load_reference_maps
    from horse_reid.canonical.planar import CANONICAL_H, CANONICAL_LANDMARKS, CANONICAL_W

    img = np.full((245, 107, 4), (255, 255, 255, 0), np.uint8)       # transparent like the real sheet
    cv2.imwrite(str(tmp_path / "front.png"), img)
    lm = {"left_ear_base": [35, 62], "right_ear_base": [72, 62], "left_eye": [24, 110],
          "right_eye": [82, 110], "nose": [53, 213]}
    (tmp_path / "front.json").write_text(json.dumps({"image": "front.png", "landmarks": lm}))
    bgr, lmf = load_diagram(tmp_path / "front.json")
    assert bgr.shape == (245, 107, 3) and bgr.min() == 255             # flattened onto white
    M, rms = estimate_affine(CANONICAL_LANDMARKS, lmf)
    assert rms < 8.0, rms                                                # drawing is not to scale (eyes sit wider)
    # reference maps: a star between the eyes, slightly above (template (128, 95))
    ref = tmp_path / "reference"; ref.mkdir()
    sup = np.zeros((CANONICAL_H, CANONICAL_W), np.uint8); cv2.circle(sup, (128, 95), 8, 255, -1)
    cv2.imwrite(str(ref / "canonical_support.png"), sup)
    cv2.imwrite(str(ref / "canonical_mask.png"), sup)
    cv2.imwrite(str(ref / "canonical_nsupport.png"), (sup > 0).astype(np.uint8) * 5)
    cv2.imwrite(str(ref / "canonical_prob.png"), sup)
    (ref / "horse_reference.json").write_text(json.dumps({"horse_id": "t", "reference": {
        "canonical_peaks": [{"id": "p0", "x": 128, "y": 95, "n_support": 5, "frames": [1, 2, 3, 4, 5]}],
        "frames": [1, 2, 3, 4, 5], "views": {"frontal": 5}}}))
    maps = load_reference_maps(ref)
    out, info = draw_on_diagram(bgr, lmf, maps, scale=4.0)
    assert out.shape == (980, 428, 3) and info["mask_px"] > 0
    px, py = info["peaks"][0]["diagram_x"], info["peaks"][0]["diagram_y"]
    # between the eyes (x ~ 53*4) and above the eye line (y < 110*4), below the ear bases (y > 62*4)
    assert abs(px - 53 * 4) < 12 and 62 * 4 < py < 110 * 4
    assert tuple(out[int(py), int(px)]) == (255, 255, 255)              # mask drawn white at the peak

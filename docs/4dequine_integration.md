# 4DEquine integration plan (Phase 3, `--mapper 4dequine`)

Status: **adapter implemented and unit-tested on synthetic meshes; the external model cannot run on this
machine.** The working mapper today is `planar` (`src/horse_reid/canonical/planar.py`).

## What 4DEquine is

- Repo: <https://github.com/luoxue-star/4DEquine>. Paper: CVPR 2026, arXiv 2603.10125.
- It reconstructs a 4D horse from a monocular video: it fits per-frame pose and shape parameters of the
  **VAREN** parametric horse model (<https://varen.is.tue.mpg.de>), then optimizes them over time.
- Inference has two stages. Stage 1 is a feed-forward prediction. Stage 2 is post-optimization. The refined
  per-frame parameters are written to `refined_results.pt`. A separate avatar demo produces a textured or
  animated avatar.

### Inference commands (from the 4DEquine README)

```bash
# 1) motion / shape reconstruction from video (stage 1 regression + stage 2 refinement)
python post_optimization_from_video.py \
    --video_path  <video.mp4> \
    --checkpoint  <path/to/checkpoint> \
    --cfg         <path/to/config.yaml> \
    --output_dir  <out_dir> \
    --stage1 --stage2
#    -> <out_dir>/.../refined_results.pt   (per-frame refined VAREN pose/shape (+ camera))

# 2) avatar (optional; not needed for marking transfer)
python demo_avatar.py ...      # see the 4DEquine README for its arguments
```

Check the exact flags against the README at the commit you clone. The commands above are the documented
two-stage entry points.

## Required artifacts

| Artifact | Source | Notes |
|---|---|---|
| 4DEquine code | GitHub repo | **BSL-1.1** license. Check it before any commercial or production use. |
| 4DEquine checkpoints | Google Drive links in the README | Large downloads |
| VAREN model `.pkl` | <https://varen.is.tue.mpg.de> | **Requires registration**. Has its own research license. |
| Python 3.10, torch 2.5.1, CUDA 12.1, pytorch3d | conda env from the README | GPU-only build |

## Why it cannot run here

This project's machine is an Intel Mac (CPU only, torch 2.2.2, Python 3.11). 4DEquine needs CUDA 12.1,
torch 2.5.1, and pytorch3d (rasterizer with CUDA kernels). CPU and macOS are not supported. The VAREN model
also needs a registration that we do not have. So we never invoke the model here.
`FourDEquineMapper.load_results / load_template / head_vertex_indices / map_frame` raise
`FourDEquineNotAvailable` with these instructions. `scripts/build_reference.py --mapper 4dequine` exits with
code 2 and prints the message.

## What is implemented (model-free, tested in `tests/test_canonical.py`)

`src/horse_reid/canonical/fourdequine.py`:

- `project_vertices(V, camera)`. Supports a pinhole camera `{"K","R","t"}` (OpenCV: x right, y down,
  looking along +z) and a weak-perspective camera `{"scale","tx","ty"[, "R"]}`.
- `vertex_visibility(V, F, camera)`. Back-face test with area-weighted vertex normals.
  **TODO:** add a z-buffer (pytorch3d rasterizer or a depth-map test) so that self-occlusion is handled.
- `sample_mask_at_vertices(prob, verts2d, visible)`. Bilinear sampling. Returns NaN for vertices that are
  not visible or fall outside the image.
- `aggregate_vertices([(prob_v, weight_v), ...])`. Uses the same weighted fusion as the planar mapper:
  prob, coverage, consistency, and a mask with `prob > 0.5 & coverage > 0.15`.
- `vertices_to_uv_image(values, uv, size)`. Scatter followed by nearest-neighbour fill. The `max_fill_dist`
  parameter limits how far the fill reaches. UV uses the OpenGL convention (v = 0 at the bottom).
- `FourDEquineMapper.observe(prob, V, F, camera, yaw, quality)`. This is the core of `map_frame` and works
  on any posed mesh.

## Integration plan (on a CUDA Linux box)

1. **Video → 4DEquine.** Run `post_optimization_from_video.py ... --stage1 --stage2` on the *same* video
   with the same rotation that Phase 1 applies (Phase 1 rotates frames by the container metadata, which is
   90° for `1000018050.mp4`). Keep `refined_results.pt`.
2. **Refined parameters → VAREN vertices per frame.** Do a VAREN forward pass (pose, shape, and global
   orientation/translation per frame) to get `V_t` (N×3) and the shared faces `F`. Implement this in
   `FourDEquineMapper.frame_geometry(frame)`.
3. **Camera.** Take the per-frame camera from the results (pinhole K/R/t or weak perspective). Convert it
   to **mask-pixel** coordinates of the Phase 2 crop using `u_mask = (u_frame - crop_x1) * upscale`, and
   the same for v. For a pinhole camera this is equivalent to `K' = S·T·K`. Align the 4DEquine frame
   indices with ours, accounting for any frame subsampling.
4. **Project and sample.** Run `vertex_visibility`, then `project_vertices`, then
   `sample_mask_at_vertices(prob_t)`, restricted to `head_vertex_indices()`. Get these indices from the
   VAREN part segmentation.
5. **Aggregate over frames.** `aggregate_vertices` uses weight = quality × view_score(yaw). With a z-buffer,
   the visibility term replaces the planar near/far-half heuristic. The result is the canonical marking
   for each vertex: prob, coverage, and consistency.
6. **Optional UV unwrap.** If VAREN provides UVs, use `vertices_to_uv_image` to produce a 2D canonical
   marking image. This image is comparable across horses and can be dropped into
   `horse_reference.json` like the planar mask.
7. **Sanity checks.** Reproject the per-vertex reference into each frame and compute IoU with the per-frame
   masks (a hold-out frame should match). Then compare the result against the planar reference for the
   frontal frames.

## Open questions (to resolve on first real run)

- **Vertex count / topology** of VAREN, and which part id or key in the pkl marks the **head** vertices.
- **UV availability.** Does the VAREN release ship UV coordinates? If it does not, we would need our own
  head UV chart, for example a cylindrical projection of the head vertices.
- **Output keys** of `refined_results.pt`: the names and shapes of pose, betas, translation, and camera,
  whether the camera is in full-frame or crop coordinates, and whether all frames are present or only
  sampled frames.
- Whether pickling the VAREN pkl needs `chumpy`. `load_template` uses `pickle(..., encoding="latin1")`.
- Licensing: BSL-1.1 (4DEquine) and VAREN terms for any non-research deployment.

## Principle

As with the planar mapper, only real observations are aggregated. Do **not** use 4DEquine's avatar or
texture output as marking evidence, because it is generated content. Use the mesh only as a geometric
carrier for sampling the real per-frame marking masks.

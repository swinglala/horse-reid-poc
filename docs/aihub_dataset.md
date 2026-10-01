# AI-Hub horse body-part dataset (white-marking training data)

**Status: NOT downloaded in this project, so no model was trained.** The
converter (`src/horse_reid/marking/aihub/convert.py`, CLI
`scripts/convert_aihub.py`) is prepared for when access is granted; the
`yolo_seg` marking segmenter (`src/horse_reid/marking/yolo_seg.py`) is a stub
that loads the resulting fine-tuned weights.

## Dataset

| | |
|---|---|
| Name | 말(馬) 부위 식별 및 이상상태 진단 이미지 데이터 (horse body-part identification and abnormal-condition diagnosis images) |
| URL | https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71707 |
| dataSetSn | 71707 |
| Organisations | Lime Solution (라임솔루션), Korea Racing Authority (KRA, 한국마사회), Jeju National University (제주대학교) |
| Built / updated | 2023 / updated 2024-10 |
| Size | 650,497 images in total |
| Body-part identification | 36,000 images / 2,000 horses |
| Gait | 600,000 images / 1,000 horses |
| Hoof | 14,497 images / 1,391 horses |
| Views per horse (body-part identification) | whole face, forehead, nose bridge, lips, left / right eyes, front / rear / left / right sides |
| Access | AI-Hub login + application approval; **Korean nationals only** |

## Annotation schema (as documented on the dataset page)

One JSON per image. Documented fields:

- horse info: `horse_id`, `horse_breed`, `horse_birth`, `horse_sex`, `horse_purpose`
- image info: `file_name`, `file_path`, `image_resolution` (`width`, `height`)
- `horse_object`:
  - `polygon`: list of `{"category": str, "coor": [[x, y], ...]}`
  - `head_whitespot_shape`, `leg_whitespot_shape` (white-marking shape attributes)
  - `vortex_bbox` (hair whorls), `eye_bbox`, `eye_shape`
  - `horse_color` (coat colour)

The exact polygon `category` strings are **not** published. Run
`python scripts/convert_aihub.py --json-dir <labels> --dry-run` first, then
pass the white-marking categories with `--category-map '{"<category>": 0}'`.
The converter is tolerant to nesting depth, flat vs. `[[x, y]]` polygons and
`{"x","y"}` point dicts; files with missing fields / images are logged and
skipped. Field names must be re-verified on real files
(`TODO(phase2-upgrade)` in `convert.py`).

## Baseline models listed on the page

- SegFormer: head segmentation and body segmentation
- Mask R-CNN: legs
- MMPose: gait (pose)
- YOLO: hoof cracks
- EfficientNet: hoof damage classification

No published metrics were found for these baselines.

## Intended use here

1. Obtain access, download the body-part identification subset (face /
   forehead / nose-bridge views carry the head white markings).
2. `scripts/convert_aihub.py --dry-run` -> pick the white-marking categories.
3. Convert to YOLO-seg and fine-tune e.g. `yolo11s-seg.pt`
   (`yolo segment train data=data/aihub_yolo/data.yaml model=yolo11s-seg.pt`).
4. Run Phase 2 with `--segmenter yolo_seg --marking-weights <best.pt>`; the
   heuristic `sam_refined` segmenter stays available for comparison.

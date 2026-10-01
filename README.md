# horse_reid

말 영상에서 **얼굴의 흰 무늬(white marking)** 를 기반으로 개체를 식별하기 위한 reference를 만드는 PoC.

## 1. 프로젝트 목표

영상 속 말의 머리를 검출해 품질이 좋은 20~50개 프레임을 고르고, 각 프레임의 흰 무늬를 segmentation한 뒤 공통 canonical 좌표계로 모아 개체 식별용 reference(`horse_reference.json`)를 만든다.
Phase 1(프레임 선택), Phase 2(흰 무늬 segmentation), Phase 3(canonical marking reference)가 순서대로 구현되어 있다.

```
말 영상 → 말 검출 → 추적 → 머리 검출 → 품질 평가 → 20~50 프레임 선택
        → 얼굴 crop → 흰무늬 segmentation → canonical mapping → reference
```

**핵심 원칙:** 생성된(generated) 얼굴이나 "예쁜" 얼굴을 identity reference로 쓰지 않는다. 실제로 관측된 픽셀만 canonical 공간에 aggregation하며, 관측되지 않은 영역은 coverage 0으로 남긴다(in-painting, mirroring 없음).

## 2. 현재 상태

| Phase | 내용 | 실제 모델 | heuristic / placeholder |
|---|---|---|---|
| 1 | 검출/추적/머리 검출/품질 평가/선택 | YOLO11s (말/사람 검출), ByteTrack, Grounding DINO tiny (머리 검출) | 품질 점수 가중합, yaw 추정(landmark + 외형 heuristic), `mask_top` fallback 머리 검출, track 병합 규칙 |
| 2 | 흰 무늬 segmentation | MobileSAM (`sam_refined`), YOLO11s-seg (얼굴 영역 mask) | adaptive color 후보 확률, 형태/위치 gate (strap, landmark hull 등). `yolo_seg`는 stub (fine-tuned 가중치 없음) |
| 3 | canonical marking 지도 + reference | CLIP ViT-B/32 (보조 face embedding) | `planar` mapper (2D similarity warp, 256x320 template). `4dequine` mapper는 adapter만 구현 (실행 불가, 6절 참고) |

## 3. 환경 / 설치

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .            # src layout의 horse_reid 패키지 설치
```

- Intel (x86_64) macOS는 PyTorch wheel이 `torch==2.2.2` / `torchvision==0.17.2`까지만 존재하고, 이는 `numpy<2`를 요구한다. `requirements.txt`는 이에 맞춰 pin되어 있으므로 올리지 말 것. Python은 3.11.
- device는 자동 선택: CUDA > MPS > CPU (`--device {auto,cpu,cuda,mps}`로 강제 가능). GPU(CUDA)가 있으면 자동으로 사용한다.
- Intel Mac + AMD GPU에서는 MPS가 보고되지만 `torchvision::nms`가 구현되어 있지 않다. `horse_reid/__init__.py`가 torch import 전에 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하고, 시작 시 nms probe가 실패하면 MPS를 무시하고 CPU를 쓴다. (torch를 먼저 import한 경우 fallback이 적용되지 않을 수 있다.)

### 모델 가중치

모두 `.gitignore` 대상이며 저장소에 포함되지 않는다.

| 가중치 | 크기 | 위치 | 획득 |
|---|---|---|---|
| YOLO11s (`yolo11s.pt`) | 18 MB | `models/` | ultralytics auto-download |
| YOLO11s-seg (`yolo11s-seg.pt`) | 20 MB | `models/` | ultralytics auto-download |
| Grounding DINO tiny | ~700 MB (cache) | `models/hf/hub` | HuggingFace에서 auto-download |
| MobileSAM (`mobile_sam.pt`) | 40 MB | `models/` | ultralytics auto-download |
| CLIP ViT-B/32 | ~340 MB | `models/weights/clip` | auto-download |

입력 영상은 `data/input/`에 직접 넣는다. 테스트에 쓴 `1000018050.mp4`는 저장소에 없다.

## 4. 실행 (CLI)

```bash
# Phase 1: 프레임 선택 (30프레임, 주석 영상 포함)
python scripts/run_pipeline.py --input data/input/1000018050.mp4 --output data/output \
    --num-frames 30 --head-detector grounding_dino --head-stride 3 --visualize

# Phase 2: 흰 무늬 segmentation (Phase 1 output 필요)
python scripts/run_marking.py --output data/output --segmenter sam_refined

# Phase 3: canonical reference (Phase 1+2 output 필요)
python scripts/build_reference.py --output data/output --input data/input/1000018050.mp4 --mapper planar

# Phase 1+2+3 한 번에
python scripts/run_pipeline.py --input data/input/1000018050.mp4 --output data/output --phase all

# 검출을 다시 돌리지 않고 저장된 detections/scores로 track 병합 + 점수 + 선택만 재실행 (수 초)
python scripts/run_pipeline.py --reselect --output data/output --num-frames 30 --min-score 0.35

# 테스트 (영상/가중치 불필요)
python -m pytest tests -q
```

`scripts/select_frames.py`는 `run_pipeline.py`와 같은 CLI의 alias이다.

| Flag | 설명 |
|---|---|
| `--input` | 입력 영상 |
| `--output` | output 디렉터리 (Phase 2/3은 여기서 Phase 1 결과를 읽음) |
| `--num-frames` | 선택할 프레임 수 |
| `--device` | `auto` / `cpu` / `cuda` / `mps` |
| `--save-all` | 점수를 계산한 모든 head crop 저장 |
| `--visualize` | `annotated.mp4` 저장 |
| `--head-stride N` | `frame_idx % N == 0`인 프레임에서만 머리 검출/채점 (tracking은 매 프레임) |
| `--head-detector` | `bbox_top` / `mask_top` / `grounding_dino` |
| `--head-fallback` | grounding_dino가 머리를 못 찾을 때 쓸 detector (`mask_top`/`bbox_top`/`none`) |
| `--min-score` | 선택 하한 (기본 0.35). 미달 프레임으로 채우지 않고 개수를 줄임 |
| `--min-gap` | 선택된 프레임 사이 최소 간격 (기본 auto) |
| `--tracker` | `bytetrack.yaml` / `botsort.yaml` |
| `--marking-segmenter` | `sam_refined` (기본) / `adaptive_color` / `yolo_seg` (`run_marking.py`에서는 `--segmenter`) |
| `--mapper` | Phase 3 mapper: `planar` / `4dequine` |
| `--horse-id` | reference의 `horse_id` |

그 외: `--phase {1,2,3,all}`, `--frame-stride`, `--max-frames` (debug), `--rotation`, `--conf`, `--hf-cache-dir`, `--sam-weights`.

### Output layout

```
data/output/
  detections.json        tracked horse/person boxes + primary track group
  frame_scores.json      모든 FrameScore, selected 인덱스, config, stage timing
  selected_frames/       annotated full frames        selected_frames_raw/  clean full frames
  face_crops/            head crops (+15% margin)     all_face_crops/       (--save-all)
  contact_sheet.jpg      선택된 crop + 점수 grid
  summary.txt            Phase 1 요약                 annotated.mp4         (--visualize)
  marking/               Phase 2: face mask, 후보 확률, 최종 mask, overlay, marking_sheet.jpg, summary.txt
  reference/             Phase 3: canonical_{prob,coverage,mask,consistency,support,nsupport,marking}.png,
                         canonical_marking.txt, horse_reference.json
  final_report.jpg       Phase 3 종합 report
```

## 5. 결과

실제 영상 `1000018050.mp4` (33.5 s, 1005 frames, 720x1280 세로, CPU 실행). 원본 결과물은 `docs/results/`에 있다.

### Phase 1: 프레임 선택

![Phase 1 contact sheet](docs/results/phase1_contact_sheet.jpg)

| 항목 | 값 |
|---|---|
| 말이 검출된 프레임 | 965 / 1005 |
| head stage 실행 프레임 | 335 (stride 3) |
| primary track group | 5개 track 병합 (ByteTrack이 한 마리를 12개 ID로 분할; 한 track은 "person"으로 시작해 majority-class track typing으로 해결) |
| scored candidates | 304 (min_score 0.35 통과 168) |
| 선택 | 30 / 30, 0.0–33.1 s 전 구간 |
| yaw bin (선택) | frontal 5 / three-quarter 11 / profile 14 |
| mean visibility (선택) | 0.957 |
| wall time | 1219 s (CPU). head stage 1065 s, Grounding DINO 약 3.2 s/frame |

25–28 s 구간에서는 frame 750, 765, 783, 798, 813, 828, 843 (25.0–28.1 s)이 선택되었다.

**검증 질문 7개**

1. **얼굴 검출이 되는가:** 예. 304개 후보 중 289개가 Grounding DINO, 15개는 heuristic fallback이며 gating 덕분에 fallback 프레임은 선택되지 않았다.
2. **tracking이 되는가:** 예. majority-class 수정 후 단일 말이 하나의 track group으로 연속 추적된다.
3. **큰 얼굴을 선호하는가:** 예. head size가 점수의 20%를 차지한다.
4. **blur 프레임이 제외되는가:** 예. Laplacian variance gate로 제외된다.
5. **사람이 가린 프레임이 제외되는가:** 예. person box 기반 occlusion score를 쓰며, 촬영자가 시야를 가린 6–8 s 부근 프레임은 0.2 미만으로 채점되었다.
6. **3/4 측면이 포함되는가:** 예. three-quarter 11 + profile 14 + frontal 5.
7. **사람이 봤을 때 진짜 머리인가:** 30장 모두 실제 머리 crop이다. 초기 실행에서 엉덩이 false positive (frame 240, head_conf 0.20)가 있어 head-confidence / head-area gate를 추가했다.

### Phase 2: 흰 무늬 segmentation

![Phase 2 marking sheet](docs/results/phase2_marking_sheet.jpg)

Frame 975 overlay (crop | face mask | candidate prob | final):

![Phase 2 overlay frame 975](docs/results/phase2_overlay_frame_00975.jpg)

| 항목 | 값 |
|---|---|
| segmenter | `sam_refined` (30 frames) |
| marking이 accept된 프레임 | 9 / 30 (frontal 3/5, three_quarter 4/11, profile 2/14) |
| 평균 marking 면적 | 얼굴의 0.23% (전체), 0.77% (accept된 프레임) |
| 제외된 영역 | too_small 324, edge_fragment 37, strap_shape 36, outside_landmark_hull 31, low_solidity 31, sam_flood 11, above_ears 7 |
| wall time | 34.6 s (CPU) |

**솔직한 평가:** 이 말은 dark bay이고 이마에 작은 star만 있다 (2x upscale에서 약 80–180 px). star는 정면 프레임 975, 993에서 accept된다. 나머지 accept 성분(frame 363/558/576/618/843)은 ear-base / poll highlight로 false positive이다. 갈기, 목, halter strap, 초록색 tag는 gate (`strap_shape`, `outside_landmark_hull`, high_chroma 등)로 제외된다. 3/4 view에서는 star가 보이지만 z-threshold 아래라 recall이 낮다. segmenter는 교체 가능한 baseline이다 (`HorseMarkingSegmenter` registry: `adaptive_color`, `sam_refined` 기본, `yolo_seg` stub).

### Phase 3: canonical marking reference

![Phase 3 canonical marking](docs/results/phase3_canonical_marking.png)

```
          HORSE MARKING           
+--------------------------------+
|                                |
|                                |
|                                |
|                                |
|                  *░            |
|              ░  ░ ░            |
|           ░░░░    ░            |
|            *░░                 |
|                                |
|                   *            |
|               *                |
|        *                       |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|         *                      |
|                                |
|                                |
|                                |
|           ░                    |
|                                |
|                                |
|             ░                  |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
+--------------------------------+
value = max(prob x coverage) per cell;  ' ' <.2  ░ <.4  ▒ <.6  ▓ <.8  █ >=.8;  * = peak below ░
```

![Final report](docs/results/final_report.jpg)

- 30프레임 중 29개 mapping (frame 648은 landmark가 없어 skip). view: frontal 5 / 3-4 10 / profile 14.
- canonical mask px = 0: star가 29프레임 중 2프레임에서만 accept되어 prob이 0.5를 넘지 못한다.
- candidate peak 7개, 그중 multi-frame support 2개:
  - p2 (96,58) n=3, frames [363, 558, 576]: ear-base highlight, false positive로 추정.
  - p5 (125,85) n=2, frames [975, 993]: 이마 star. 서로 다른 두 프레임이 같은 canonical 위치에 정렬되었다 (정렬 오차 약 10 px, sigma 4 px smoothing으로 흡수).
- `face_embedding`: CLIP ViT-B/32, 29개 crop의 mean (보조 용도). 주 identity 증거는 canonical marking mask이다.

`reference/horse_reference.json` 구조 (일부 생략):

```json
{
  "horse_id": "unknown",
  "reference": {
    "canonical_marking_mask": "reference/canonical_mask.png",
    "face_embedding": [ "... 512-d CLIP vector ..." ],
    "face_embedding_method": "clip_ViT-B-32_mean_of_29",
    "view_count": 29,
    "views": {"three_quarter": 10, "profile": 14, "frontal": 5},
    "frames": [0, 24, 42, "..."],
    "mapper": "planar",
    "canonical_size": [256, 320],
    "files": {"prob": "...", "coverage": "...", "mask": "...", "marking": "...", "ascii": "..."},
    "canonical_peaks": [{"id": "p5", "x": 125, "y": 85, "n_support": 2, "frames": [975, 993], "...": "..."}],
    "params": {"peak_sigma": 4.0, "prob_threshold": 0.5, "...": "..."},
    "stats": {"mask_px": 0, "n_peaks": 7, "n_peaks_multi": 2, "skipped_frames": [648]},
    "notes": ["..."]
  }
}
```

## 6. 한계와 다음 단계

- **head-stride 비용:** CPU에서 Grounding DINO가 약 3.2 s/frame이라 stride 3에도 head stage가 1065 s. GPU 또는 더 가벼운 detector가 필요하다.
- **yaw heuristic:** profile을 과다 판정하는 경향이 있다 (profile 140 / three_quarter 139 / frontal 25).
- **segmenter:** 햇빛이 강한 dark coat에서 recall/precision이 낮다 (ear-base highlight false positive, 3/4 view 누락). AI-Hub 등으로 fine-tune한 `yolo_seg`가 다음 단계.
- **2D planar canonical:** 3D 머리 표면이 아닌 평면 template이라 3/4, profile의 far half는 down-weight만 한다. 정렬 오차 약 10 px.
- **4DEquine:** CUDA 전용 + BSL-1.1 + VAREN 모델 등록이 필요해 이 환경에서 실행할 수 없다. adapter와 model-free 수학은 구현/테스트되어 있다. 참고: [docs/4dequine_integration.md](docs/4dequine_integration.md).
- **AI-Hub 데이터셋:** 한국 국적 로그인/승인이 필요해 이 환경에서 받을 수 없다. 가짜 데이터는 만들지 않았고 converter(`scripts/convert_aihub.py`)만 준비되어 있다. 참고: [docs/aihub_dataset.md](docs/aihub_dataset.md).
- **다중 말 영상:** 현재 track 병합은 시간 기반 규칙이다. 여러 마리가 나오는 영상에는 appearance 기반 track 병합이 필요하다.
- 모델 선택 근거는 [docs/model_selection.md](docs/model_selection.md).

## 7. 저장소 구조

```
src/horse_reid/
  config.py device.py types.py pipeline.py marking_pipeline.py
  video/reader.py                 rotation-aware VideoReader
  detection/horse_detector.py     YOLO11 + ByteTrack
  tracking/tracks.py              TrackStore, majority-class typing, primary track group
  face/                           head_detector.py (registry, heuristics), grounded_head_detector.py
  quality/                        metrics.py, scorer.py
  selection/selector.py           temporal NMS + yaw-diversity
  visualization/                  annotate.py, contact_sheet.py
  marking/                        base.py, registry.py, face_region.py, adaptive_color.py,
                                  sam_refined.py, yolo_seg.py (stub), visualize.py, aihub/convert.py
  canonical/                      base.py, planar.py, fourdequine.py, embedding.py, reference.py, io.py
scripts/                          run_pipeline.py, select_frames.py, run_marking.py,
                                  build_reference.py, convert_aihub.py
tests/                            75 tests (test_smoke, test_grounded_head, test_marking, test_canonical)
docs/                             model_selection.md, aihub_dataset.md, 4dequine_integration.md, results/
```

## 8. 라이선스 / 출처

| 구성요소 | 라이선스 | 비고 |
|---|---|---|
| YOLO11 (ultralytics) | AGPL-3.0 | |
| Grounding DINO | Apache-2.0 | HuggingFace transformers 경유 |
| MobileSAM | Apache-2.0 | |
| CLIP | MIT | |
| 4DEquine | BSL-1.1 | 저장소에 포함하지 않음 |
| VAREN | 별도 등록 필요 | 포함하지 않음 |

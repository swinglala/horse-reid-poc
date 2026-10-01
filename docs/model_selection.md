# 모델 선택 및 구성 가이드

## 환경 요약

| 항목 | 상세 |
|------|------|
| **하드웨어** | Intel Mac (i5-10600, AMD Radeon Pro 5300 4GB) |
| **OS** | macOS (CUDA 불가) |
| **PyTorch** | 2.2.2 (Intel-macOS 공식 지원 마지막 버전) |
| **Python** | 3.11 venv |
| **의존성** | torch 2.2.2, torchvision 0.17.2, numpy<2 |
| **GPU** | MPS 사용 가능 (torchvision::nms 미지원 → fallback 필수) |
| **테스트 비디오** | 1000018050.mp4: 1280×720, 90° rotation metadata, 30 fps, 1005 프레임 (33.5초) |

---

## 1. 말 탐지 (Horse Detection)

### 선택
**Ultralytics YOLO11s (COCO 사전학습 모델, class 17 = horse)**
- 추적(tracking) API 포함 (ByteTrack, BoT-SORT)
- COCO 동물 감지 신뢰도 높음
- Class 0 (person) 포함 → 폐색(occlusion) 채점에 활용

### 이유
- **편의성**: pip 설치만 필요, 가중치 자동 다운로드 (`models/yolo11s.pt`, 18 MB)
- **성능**: 이 CPU에서 ≈0.12초/프레임 (720×1280), 이 비디오에서 말 신뢰도 0.93–0.96
- **시간 제약**: 전체 파이프라인 내 동작 필요 (배치 처리 불가)
- **추적 통합**: 기본 제공 API로 빠른 구현

### 검토한 대안

| 대안 | 이유 (불선택) |
|------|-------------|
| YOLO-World | Open-vocabulary로 "말 머리"를 개별 감지로 기대했으나, 모든 프롬프트("horse head", "horse face" 등)에서 전체 말 바운딩박스만 반환 → 머리 감지기로 부적절 |
| YOLOE | MobileCLIP 다운로드 필요, torch 2.2.2와 호환성 미검증 |
| MMPose/mmdet | Intel Mac 상 torch 2.2.2와 빌드 불가 |

### 한계/TODO
- 폐색 후 추적 ID 변경: 이 비디오에서 단일 말이 12개 추적 ID로 분산
- TrackStore: 다수 클래스 감지 시 majority class로 분류하고, 시간적으로 분리된 말 트랙을 주 트랙 그룹으로 병합
- **TODO**: 다중 말 비디오 시 외형 기반 검증 필요

---

## 2. 추적 (Tracking)

### 선택
**Ultralytics 기본 ByteTrack** (`model.track(..., tracker="bytetrack.yaml", persist=True)`)
- BoT-SORT는 `--tracker` 플래그로 선택 가능

### 이유
- YOLO11s 스택에 통합됨 (별도 설치 불필요)
- CPU에서 안정적 성능
- 동영상 전체에 persistent state 유지

### 검토한 대안
BoT-SORT: 가용하나, 이 비디오 폐색 시나리오에서 특별한 개선 없음

### 한계/TODO
- **추적 분산**: 한 말이 12개 ID로 분산 (폐색 중 핸들러/카메라맨 감지)
- **클래스 점프**: 한 트랙(ID 69, 프레임 225–882)이 "person"으로 시작 후 "horse"로 변경
- **다중 말**: 외형 기반 재식별(Re-ID) 모듈 필요 (TODO)

---

## 3. 말 머리 감지 (Horse Head Detection)

### 선택
**Grounding DINO tiny** (IDEA-Research/grounding-dino-tiny, Apache-2.0)
- HuggingFace transformers 4.49 경유
- 프롬프트: "horse head. horse ear. horse eye. horse nose."
- 머리 박스 + 귀/눈/코 부분 박스 (landmark로 사용)

### 이유
- 공식 사전학습 말-머리 감지기 부재 (커뮤니티 Roboflow 데이터셋만 존재)
- 오픈 어휘 제로샷(zero-shot) → 즉시 배포 가능
- 말 자르기(crop) 영역(bbox +10%)에서 실행 → 계산 비용 낮음
- 부분 landmark 제공 (후속 단계에서 활용)

### 감지 및 필터링 절차
1. 프로세서 설정: `shortest_edge=480` (≈3초/프레임 CPU; 기본값 800은 ≈9초)
2. 머리 박스 + 부분 박스 반환
3. 전체 말 박스가 낮은 신뢰도로 "horse head" 레이블될 때 필터: ≥50% crop 영역 차지 시 거부
4. 후향(rear view) 처리: 눈/코 신뢰도로부터 얼굴 가시성 점수 계산

### 검증 및 성능
- 표본 12 프레임: 머리 감지 11/12 (1회 미스 = 프레임 밖)
- 계산 비용 때문에 `--head-stride 3`으로 실행 (335/1005 프레임; 전체 1219초 중 1065초)

### 검토한 대안

| 대안 | 상태 | 이유 |
|------|------|------|
| DeepLabCut SuperAnimal-Quadruped | 문서화됨 | 39 keypoint (HRNet+FasterRCNN), CPU 속도 느림 |
| MMPose AP-10K | 문서화됨 | 3 head keypoint, mmcv 빌드 필수 |
| OWLv2 | 테스트됨 | 작동하나 ≈10–15초/프레임, Grounding DINO보다 노이지 |
| Mask-Top 휴리스틱 | TODO | YOLO11s-seg 말 mask, head = 상단 mask 픽셀 영역 |
| BBox-Top 휴리스틱 | TODO | 말 박스 상단 42% |

### 한계/TODO
- **각도 편향**: 4분각(3/4) 뷰에서 "정면(profile)" 과판정 경향 (두 번째 눈이 0.25 점수 임계값 이하)
- **속도**: 3프레임마다만 실행 (전체 비용 제한)
- **TODO**: 학습된 자세(pose) 모델로 개선

---

## 4. 말 얼굴 방향각 (Yaw Estimation)

### 선택
**Landmark 기반 휴리스틱 + 외형 fallback**

### 이유
- 공식 사전학습 말-얼굴 yaw 모델 부재
- 즉시 배포 가능, CPU 오버헤드 최소

### 휴리스틱 로직
1. **두 눈 감지**: 눈 spread 비율로 yaw 판정
2. **한 눈 + 코**: 코 중심 상대 위치로 방향 결정
3. **코 측면**: 코 위치의 부호로 좌/우 판별
4. **외형 fallback**: 종횡비(aspect ratio) + 좌우 대칭성(mirror symmetry)

### 검토한 대안
- 학습된 자세 모델 (성능 더 우수하나 CPU 속도 문제)

### 한계/TODO
- **과판정**: 3/4 뷰에서 "정면(profile)"으로 과판정 (두 번째 눈 임계값 0.25 이하)
- **TODO**: 학습된 자세 모델 도입 필수

---

## 5. 흰색 표시 분할 (White-Marking Segmentation, Phase 2)

### 선택
**Lab 색상 + MobileSAM 정제 파이프라인**

#### 파이프라인 단계
1. **얼굴-영역 마스크**: `YOLO11s-seg 말 mask ∩ 머리 박스`, 침식(erosion)
2. **색상 적응 분할**: Lab L* z-score (국소 조명장 = 큰 커널 masked blur) + 저-색도(low-chroma) 게이트
3. **컴포넌트 필터**:
   - 최소 면적, 솔리드성(solidity)
   - 스트랩(strap) 형태: min-area-rect + SAM-마스크 둘레²
   - 고-색도(high chroma) 거부
   - 엣지 조각(edge fragments) 제거
   - 눈 반짝임(eye glint) 제거
4. **MobileSAM 점-프롬프트 정제**:
   - ultralytics `mobile_sam.pt` (40 MB, ≈1.7초/호출 CPU)
   - 양(positive): 후보 무게중심
   - 음(negative): 2개 코트 포인트
   - 수용 조건: 면적 < 얼굴 20%, 평균 z-score > 1.5, dilated 후보와 IoU > 0.3

### 이유
- RGB 임계값 기반 접근은 설계상 거부됨 (조명 민감도)
- Lab L* z-score: 국소 조명 정규화
- MobileSAM: 후보 정제로 오탐 제거
- **레지스트리 확장성**: `HorseMarkingSegmenter` 레지스트리로 구현 교체 가능

### 대체 구현 (레지스트리)
| 구현 | 상태 | 용도 |
|------|------|------|
| `adaptive_color` | 사용 가능 | RGB 임계값 대체 |
| `sam_refined` | 기본값 | MobileSAM 정제 |
| `yolo_seg` | 스텁(stub) | AI-Hub 미세조정 모델 (미사용) |

### 한계/TODO
- **일루미네이션 편향**: 역광/측광 환경에서 성능 저하 예상
- **다중 표시**: 현재 단일 마스크 (복수 표시 미지원)
- **TODO**: CLIP embedding 기반 재식별(Re-ID) 보강

---

## 6. 정규화 매핑 (Canonical Mapping, Phase 3)

### 선택
**모델-프리 구성요소 + 2D Planar Canonical Mapper**

### 현황
- **4DEquine (CVPR 2026, BSL-1.1)**: VAREN 모델은 등록 필수, CUDA 12.1/pytorch3d 필수 → 이 머신 불가
- **어댑터 모듈**:
  - 모델-프리 부분: 꼭짓점 투영, 가시성, 마스크 샘플링, 집계 (구현 완료 및 테스트)
  - **PlanarCanonicalMapper**: 2D 구현 (프로덕션 준비)

#### PlanarCanonicalMapper 알고리즘
1. **Landmark 입력**: 눈/코/귀 랜드마크 (좌표 + 신뢰도)
2. **정규화 템플릿**: 256×320 정면 템플릿
3. **변환**: 유사성 워프(similarity warp), yaw 의존 가시성 가중치
4. **집계**: 가중 평균, coverage/consistency 맵
5. **보조 임베딩**: CLIP ViT-B/32 (ultralytics 캐시)
6. **ID 증거**: 흰색 표시 마스크 (주요), CLIP 임베딩 (보조)

### 이유
- 4DEquine VAREN: 이 머신 불가능
- 2D planar: CPU 실시간 처리 가능
- CLIP embedding: 이미 캐시 (추가 다운로드 불필요)

### 검토한 대안
- 완전 3D (pytorch3d): CUDA 필수 (불가)
- Heatmap-기반 정규화: 속도 느림

### 한계/TODO
- **2D 제약**: 3D 재구성 미지원
- **CLIP 보조성**: 표시 마스크가 주요 ID 증거
- **TODO**: 다중 말 구분 개선, 표시 변화 추적

---

## 다운로드되는 가중치

| 모델 | 크기 | 저장 위치 | 설명 |
|------|------|---------|------|
| yolo11s.pt | 18 MB | models/yolo11s.pt | YOLO11s 가중치 (말 탐지) |
| yolo11s-seg.pt | 20 MB | models/yolo11s-seg.pt | YOLO11s-seg 가중치 (분할) |
| grounding-dino-tiny | ~700 MB | models/hf/hub | Grounding DINO tiny (HF cache) |
| mobile_sam.pt | 40 MB | models/mobile_sam.pt | MobileSAM (흰색 표시 정제) |
| CLIP ViT-B/32 | ~340 MB | models/weights/clip | CLIP 임베딩 (정규화 보조) |

### 자동 다운로드
- 모든 가중치는 **첫 실행 시 자동 다운로드**
- `.gitignore`에 등록되어 리포 제외
- 환경 변수 `TORCH_HOME` (PyTorch), `HF_HOME` (HuggingFace) 지원


"""Configuration dataclasses for the Phase 1 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(p: str | Path) -> Path:
    """Absolute paths are returned unchanged; relative ones are resolved
    against the project root (the directory containing ``src/``)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class QualityWeights:
    """Weights of the linear frame-quality combination (see ``FrameQualityScorer``)."""

    size: float = 0.20
    blur: float = 0.20
    exposure: float = 0.10
    occlusion: float = 0.15
    view: float = 0.10
    det_conf: float = 0.10
    visibility: float = 0.15

    def total(self) -> float:
        return (self.size + self.blur + self.exposure + self.occlusion + self.view + self.det_conf
                + self.visibility)


@dataclass
class QualityParams:
    """Tunable constants for the individual quality metrics."""

    size_ref_px: float = 300.0          # sqrt(head area) that maps to size score 1.0
    blur_ref: float = 150.0             # absolute Laplacian variance reference ("absolute" mode)
    # "adaptive": blur_ref = max(blur_ref_min, median blur_var over the run's candidates), so a
    # far-away / soft video is judged relative to itself; "absolute": always ``blur_ref``.
    blur_ref_mode: str = "adaptive"
    blur_ref_min: float = 20.0
    blur_resize_width: int = 256
    # Hard gates: frames failing any gate are kept but demoted (score *= gate_factor).
    gate_min_occlusion: float = 0.5
    gate_min_blur: float = 0.15
    gate_min_exposure: float = 0.2
    gate_min_visibility: float = 0.25   # eye+nose evidence (landmark detectors only)
    gate_min_head_conf: float = 0.30    # head detector confidence (skipped for heuristic heads)
    gate_max_head_area_ratio: float = 0.45  # head box area / horse box area above this is implausible
    head_conf_ref: float = 0.5          # head_conf at which the det_conf component is not reduced
    gate_factor: float = 0.2
    heuristic_head_factor: float = 0.85  # score multiplier for heuristic head boxes
    fallback_visibility: float = 0.1     # visibility assigned to gd_fallback heads (detector found no head)
    min_head_width_px: float = 96.0      # median head width below this -> resolution warning

    def __post_init__(self) -> None:
        if self.blur_ref_mode not in ("adaptive", "absolute"):
            raise ValueError("blur_ref_mode must be 'adaptive' or 'absolute'")


@dataclass
class PipelineConfig:
    """All settings for ``run_phase1``."""

    input: Path = Path("data/input/1000018050.mp4")
    output: Path = Path("data/output")
    num_frames: int = 30
    device: str = "auto"
    save_all: bool = False
    visualize: bool = False
    frame_stride: int = 1
    min_frame_gap: Optional[int] = None  # None -> max(3, total_frames // (num_frames * 2))
    max_frames: Optional[int] = None     # debug: stop after this many processed frames
    min_score: float = 0.35              # selection quality floor (fewer frames rather than junk)

    # Video
    rotation: Optional[int] = None       # None -> use container metadata; else 0/90/180/270

    # Detection / tracking
    model: Path = Path("models/yolo11s.pt")
    tracker: str = "bytetrack.yaml"
    conf: float = 0.25
    imgsz: int = 640

    # Head detection
    head_detector: str = "grounding_dino"
    head_fallback: str = "mask_top"      # fallback of grounding_dino: mask_top | bbox_top | none
    head_stride: int = 3                 # run head detection + scoring every N-th frame index
    hf_cache_dir: Path = Path("models/hf/hub")  # relative -> project root
    seg_model: Path = Path("models/yolo11s-seg.pt")
    seg_imgsz: int = 416
    head_crop_margin: float = 0.15

    weights: QualityWeights = field(default_factory=QualityWeights)
    quality: QualityParams = field(default_factory=QualityParams)

    def __post_init__(self) -> None:
        self.input = Path(self.input)
        self.output = Path(self.output)
        self.model = Path(self.model)
        self.seg_model = Path(self.seg_model)
        self.hf_cache_dir = Path(self.hf_cache_dir)
        if self.head_stride < 1:
            raise ValueError("head_stride must be >= 1")
        if self.head_fallback not in ("mask_top", "bbox_top", "none"):
            raise ValueError("head_fallback must be one of mask_top, bbox_top, none")
        if self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if self.rotation is not None and self.rotation % 360 not in (0, 90, 180, 270):
            raise ValueError("rotation must be one of 0, 90, 180, 270")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

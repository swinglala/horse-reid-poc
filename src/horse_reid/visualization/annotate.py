"""Draw detections on frames."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..types import BBox, Detection

HORSE_COLOR = (0, 200, 0)
PRIMARY_COLOR = (0, 255, 255)
PERSON_COLOR = (255, 128, 0)
HEAD_COLOR = (0, 0, 255)
# keypoint part -> (BGR colour, letter)
KEYPOINT_STYLE: dict[str, tuple[tuple[int, int, int], str]] = {
    "ear": ((255, 0, 255), "E"),      # magenta
    "eye": ((255, 255, 0), "Y"),      # cyan
    "nose": ((0, 165, 255), "N"),     # orange
    "poll_top": ((255, 255, 255), "P"),
}


def draw_keypoints(img: np.ndarray, keypoints: Optional[Mapping[str, Any]], radius: int = 6) -> np.ndarray:
    """Draw part keypoints (``{part: [[x, y, score], ...]}``) in place as small
    filled circles coloured per part, with the part letter next to them."""
    if not keypoints:
        return img
    for part, pts in keypoints.items():
        color, letter = KEYPOINT_STYLE.get(part, ((200, 200, 200), part[:1].upper() or "?"))
        for pt in pts or []:
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            cv2.circle(img, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
            org = (x + radius + 2, y + radius)
            cv2.putText(img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int],
           scale: float = 0.6) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    horse_dets: Iterable[Detection],
    person_dets: Iterable[Detection],
    head_bbox: Optional[BBox] = None,
    label_text: Optional[str | Sequence[str]] = None,
    primary_track_id: Optional[int] = None,
    copy: bool = True,
    keypoints: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Draw horse (green; primary track yellow), person (orange) and head (red)
    boxes, optional head keypoints (see :func:`draw_keypoints`) plus optional
    multi-line label text in the top-left corner."""
    img = frame.copy() if copy else frame
    for d in person_dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        _label(img, f"person#{d.track_id} {d.confidence:.2f}", (x1, y1), PERSON_COLOR)
    for d in horse_dets:
        color = PRIMARY_COLOR if (primary_track_id is not None and d.track_id == primary_track_id) else HORSE_COLOR
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        _label(img, f"horse#{d.track_id} {d.confidence:.2f}", (x1, y1), color)
    if head_bbox is not None:
        x1, y1, x2, y2 = head_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), HEAD_COLOR, 3)
    draw_keypoints(img, keypoints)
    if label_text:
        lines = [label_text] if isinstance(label_text, str) else list(label_text)
        y = 30
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30
    return img

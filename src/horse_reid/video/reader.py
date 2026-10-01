"""Rotation-aware sequential / random-access video reader built on OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate a frame clockwise by ``rotation`` degrees (0/90/180/270)."""
    rotation %= 360
    if rotation == 0:
        return frame
    if rotation not in _ROTATE_CODES:
        raise ValueError(f"Unsupported rotation {rotation}")
    return cv2.rotate(frame, _ROTATE_CODES[rotation])


class VideoReader:
    """Iterate a video as ``(frame_idx, timestamp_s, frame_bgr)`` tuples.

    OpenCV's automatic orientation handling is explicitly disabled and the
    container rotation metadata (``CAP_PROP_ORIENTATION_META``) is applied
    manually, so behaviour is identical across OpenCV builds. Pass
    ``rotation`` to override the metadata (e.g. ``0`` to disable rotation).

    Args:
        path: Video file.
        stride: Yield every ``stride``-th frame (frames in between are
            grabbed but not decoded).
        rotation: ``None`` to use metadata, otherwise 0/90/180/270 clockwise.
        max_frames: Stop after yielding this many frames.
    """

    def __init__(
        self,
        path: str | Path,
        stride: int = 1,
        rotation: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None
        self._rotation_override = rotation
        self._open()

    # ------------------------------------------------------------------ #
    def _open(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video {self.path}")
        # Make orientation handling deterministic: we rotate ourselves.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        self._cap = cap

        self.fps: float = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.raw_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
        self.metadata_rotation: int = meta
        self.rotation: int = (
            meta if self._rotation_override is None else int(self._rotation_override) % 360
        )
        if self.rotation not in (0, 90, 180, 270):
            logger.warning("Unsupported rotation %s; ignoring", self.rotation)
            self.rotation = 0
        if self.rotation in (90, 270):
            self.width, self.height = self.raw_height, self.raw_width
        else:
            self.width, self.height = self.raw_width, self.raw_height
        logger.info(
            "Opened %s: raw %dx%d, rotation meta=%d applied=%d -> %dx%d, %.3f fps, %d frames",
            self.path.name, self.raw_width, self.raw_height, meta, self.rotation,
            self.width, self.height, self.fps, self.frame_count,
        )

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def timestamp(self, frame_idx: int) -> float:
        return frame_idx / self.fps if self.fps > 0 else 0.0

    def _post(self, frame: np.ndarray) -> np.ndarray:
        return rotate_frame(frame, self.rotation)

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Sequential pass from frame 0 (re-seeks to the start)."""
        assert self._cap is not None, "reader is closed"
        cap = self._cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        yielded = 0
        while True:
            if self.max_frames is not None and yielded >= self.max_frames:
                break
            if idx % self.stride == 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield idx, self.timestamp(idx), self._post(frame)
                yielded += 1
            else:
                if not cap.grab():
                    break
            idx += 1

    def __len__(self) -> int:
        """Number of frames the iterator is expected to yield."""
        n = (self.frame_count + self.stride - 1) // self.stride
        return min(n, self.max_frames) if self.max_frames is not None else n

    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        """Random access to frame ``idx`` (rotated). Uses a separate capture so it
        does not disturb an ongoing iteration. Returns ``None`` on failure."""
        cap = cv2.VideoCapture(str(self.path))
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("read_frame(%d) failed", idx)
                return None
            return self._post(frame)
        finally:
            cap.release()

    def read_frames(self, indices: list[int]) -> dict[int, np.ndarray]:
        """Read several frames with one sequential pass (exact, avoids seek
        inaccuracies of some codecs). Only the requested frames are kept."""
        wanted = set(int(i) for i in indices)
        out: dict[int, np.ndarray] = {}
        if not wanted:
            return out
        last = max(wanted)
        cap = cv2.VideoCapture(str(self.path))
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            idx = 0
            while idx <= last:
                if idx in wanted:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    out[idx] = self._post(frame)
                elif not cap.grab():
                    break
                idx += 1
        finally:
            cap.release()
        return out

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

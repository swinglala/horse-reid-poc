"""Accumulate per-frame detections into tracks and choose the primary horse."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from ..types import Detection, bbox_area

logger = logging.getLogger(__name__)


class TrackStore:
    """Holds every detection of a video, grouped by frame and by track id.

    Only small tuples are stored, so memory stays tiny even for long videos.
    """

    def __init__(self, horse_cls: str = "horse") -> None:
        self.horse_cls = horse_cls
        self.detections: list[Detection] = []
        self._by_track: dict[int, list[Detection]] = defaultdict(list)
        self._by_frame: dict[int, list[Detection]] = defaultdict(list)
        self._class_cache: dict[int, str] = {}

    def add(self, dets: Iterable[Detection]) -> None:
        for d in dets:
            self.detections.append(d)
            self._by_frame[d.frame].append(d)
            if d.track_id is not None:
                self._by_track[d.track_id].append(d)
                self._class_cache.pop(d.track_id, None)

    @classmethod
    def from_detection_dicts(cls, rows: Iterable[dict], horse_cls: str = "horse") -> "TrackStore":
        """Rebuild a store from :meth:`Detection.to_dict` rows (e.g. ``detections.json``)."""
        store = cls(horse_cls=horse_cls)
        store.add(
            Detection(
                frame=int(r["frame"]),
                track_id=None if r.get("track_id") is None else int(r["track_id"]),
                bbox=tuple(int(v) for v in r["bbox"]),  # type: ignore[arg-type]
                confidence=float(r["confidence"]),
                cls_name=str(r["cls"]),
                time_s=float(r.get("time_s", 0.0)),
            )
            for r in rows
        )
        return store

    # ------------------------------------------------------------------ #
    def frame_detections(self, frame: int) -> list[Detection]:
        return list(self._by_frame.get(frame, []))

    def track_class(self, track_id: int) -> Optional[str]:
        """Majority class over the track's detections.

        The tracker may keep one id while the detector's class flips (e.g. the
        first box of a horse track labelled ``person``). Ties go to the class
        with the larger total box area.
        """
        if track_id in self._class_cache:
            return self._class_cache[track_id]
        ds = self._by_track.get(track_id)
        if not ds:
            return None
        counts: Counter[str] = Counter()
        areas: dict[str, int] = defaultdict(int)
        for d in ds:
            counts[d.cls_name] += 1
            areas[d.cls_name] += bbox_area(d.bbox)
        cls = max(counts, key=lambda c: (counts[c], areas[c]))
        self._class_cache[track_id] = cls
        return cls

    def track(self, track_id: int) -> list[Detection]:
        """Detections of the track restricted to its majority class
        (stray other-class frames are dropped; :attr:`detections` keeps all)."""
        cls = self.track_class(track_id)
        return [d for d in self._by_track.get(track_id, []) if d.cls_name == cls]

    def horse_track_ids(self) -> list[int]:
        return sorted(tid for tid in self._by_track if self.track_class(tid) == self.horse_cls)

    def track_length(self, track_id: int) -> int:
        return len(self.track(track_id))

    def track_mean_area(self, track_id: int) -> float:
        ds = self.track(track_id)
        return sum(bbox_area(d.bbox) for d in ds) / len(ds) if ds else 0.0

    def primary_horse_track(self) -> Optional[int]:
        """Primary horse = longest horse track; ties broken by mean box area.

        # TODO(phase1-upgrade): merge fragmented tracks of the same horse
        # (ID switches after occlusion) using appearance / box continuity.
        """
        best: Optional[int] = None
        best_key = (-1, -1.0)
        for tid in self.horse_track_ids():
            key = (self.track_length(tid), self.track_mean_area(tid))
            if key > best_key:
                best, best_key = tid, key
        return best

    def primary_track_group(self, min_area_ratio: float = 0.25) -> list[int]:
        """Primary horse track plus horse tracks that never co-occur with it
        (or with each other) in any frame and whose mean box area is at least
        ``min_area_ratio`` x the primary track's mean area.

        With a single horse in view, a tracker ID switch splits the horse into
        several temporally disjoint tracks; treating them as one animal keeps
        the whole video available for frame selection. The area floor drops
        tiny false-positive "horse" boxes (background objects, distant animals).

        # TODO(phase1-upgrade): verify the merge with appearance embeddings;
        # disjoint tracks can be different horses entering one after another.
        """
        primary = self.primary_horse_track()
        if primary is None:
            return []
        group = [primary]
        occupied = {d.frame for d in self.track(primary)}
        min_area = min_area_ratio * self.track_mean_area(primary)
        others = sorted(
            (t for t in self.horse_track_ids() if t != primary),
            key=lambda t: (-self.track_length(t), t),
        )
        for t in others:
            if self.track_mean_area(t) < min_area:
                continue
            frames = {d.frame for d in self.track(t)}
            if frames.isdisjoint(occupied):
                group.append(t)
                occupied |= frames
        return group

    @staticmethod
    def primary_from_frame(dets: list[Detection], horse_cls: str = "horse",
                           prefer_track: Optional[int] = None) -> Optional[Detection]:
        """Online choice of the horse to analyse in a single frame: the
        ``prefer_track`` one if present, else the largest horse box."""
        horses = [d for d in dets if d.cls_name == horse_cls]
        if not horses:
            return None
        if prefer_track is not None:
            for d in horses:
                if d.track_id == prefer_track:
                    return d
        return max(horses, key=lambda d: (bbox_area(d.bbox), d.confidence))

    # ------------------------------------------------------------------ #
    def to_json(self) -> dict:
        primary = self.primary_horse_track()
        return {
            "primary_track_id": primary,
            "primary_track_length": self.track_length(primary) if primary is not None else 0,
            "primary_track_group": self.primary_track_group(),
            "horse_track_ids": self.horse_track_ids(),
            # length = detections of the track's majority class (see track_class)
            "track_lengths": {str(t): self.track_length(t) for t in sorted(self._by_track)},
            "track_classes": {str(t): self.track_class(t) for t in sorted(self._by_track)},
            "detections": [d.to_dict() for d in self.detections],
        }

    def export(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)
        logger.info("Wrote %d detections to %s", len(self.detections), path)
        return path

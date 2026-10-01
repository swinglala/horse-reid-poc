"""Select N high-quality, temporally spread, view-diverse frames.

Algorithm (:func:`select_frames`):

0. **Quality floor.** Candidates scoring below ``min_score`` are dropped; if
   fewer than ``num_frames`` remain, fewer frames are returned (no junk fill).
   **Low-quality fallback:** if fewer than ``min(num_frames, 5)`` candidates
   pass, a WARNING is logged and *all* candidates are ranked by score with the
   same temporal / diversity rules; the result is flagged ``low_quality=True``.
1. **Greedy temporal NMS.** Sort candidates by score (desc). Accept a
   candidate if it is at least ``min_gap`` frames away from every already
   accepted frame; stop at ``num_frames``.
2. **Yaw-diversity pass.** Bin candidates by |yaw| into frontal (< 30 deg),
   three-quarter (30-60 deg) and profile (> 60 deg). A bin is *eligible* if it
   has candidates with ``score >= 0.5 * best_score``. Each eligible bin should
   own at least ``ceil(num_frames * 0.15)`` slots. While a bin is under quota,
   its best eligible unselected candidate (that still respects ``min_gap``)
   replaces the lowest-scoring pick of the most over-represented bin (a bin
   holding more than its quota). If fewer than ``num_frames`` frames were
   picked, the candidate is simply added instead.
3. Return the picks sorted by frame index.

# TODO(phase1-upgrade): the yaw bins rely on the heuristic yaw estimate;
# replace with a landmark/learned yaw once available. Consider appearance
# diversity (embedding distance) in addition to time and yaw.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Optional, Sequence

from ..quality.metrics import yaw_bin
from ..types import FrameScore

logger = logging.getLogger(__name__)


def default_min_gap(total_frames: int, num_frames: int) -> int:
    """``max(3, total_frames // (num_frames * 2))``."""
    return max(3, int(total_frames) // max(1, num_frames * 2))


def _gap_ok(frame: int, selected: Sequence[FrameScore], min_gap: int,
            exclude: Optional[FrameScore] = None) -> bool:
    return all(abs(frame - s.frame) >= min_gap for s in selected if s is not exclude)


class Selection(list):
    """``list[FrameScore]`` returned by :func:`select_frames`, plus metadata.

    Attributes:
        low_quality: True if the low-quality fallback fired.
        reason: Human-readable reason for the fallback (``None`` otherwise).
        n_pass: Candidates scoring ``>= min_score``.
        n_candidates: All candidates.
    """

    def __init__(self, frames: Sequence[FrameScore] = (), low_quality: bool = False,
                 reason: Optional[str] = None, n_pass: int = 0, n_candidates: int = 0) -> None:
        super().__init__(frames)
        self.low_quality = low_quality
        self.reason = reason
        self.n_pass = n_pass
        self.n_candidates = n_candidates


def select_frames(
    scores: Sequence[FrameScore],
    num_frames: int,
    min_gap_frames: Optional[int] = None,
    yaw_bins: tuple[float, float] = (30.0, 60.0),
    total_frames: Optional[int] = None,
    min_bin_fraction: float = 0.15,
    eligible_fraction: float = 0.5,
    min_score: float = 0.35,
    low_quality_fallback: bool = True,
) -> Selection:
    """Select up to ``num_frames`` frames (see module docstring).

    Args:
        scores: Candidate frame scores (one per frame is expected).
        num_frames: Number of frames to return (fewer if not enough candidates).
        min_gap_frames: Minimum frame-index distance between picks; ``None`` ->
            :func:`default_min_gap` over ``total_frames``.
        yaw_bins: |yaw| thresholds separating frontal / three-quarter / profile.
        total_frames: Length of the analysed span (defaults to the span of the
            candidate frame indices).
        min_bin_fraction: Fraction of ``num_frames`` guaranteed per eligible bin.
        eligible_fraction: A bin is eligible if it has a candidate scoring at
            least this fraction of the best score.
        min_score: Candidates scoring below this are never selected, unless
            the low-quality fallback fires.
        low_quality_fallback: If fewer than ``min(num_frames, 5)`` candidates
            pass ``min_score``, rank all candidates instead and flag the result.
    """
    if not scores or num_frames <= 0:
        return Selection(n_candidates=len(scores or []))
    if total_frames is None:  # span of all candidates, before the quality floor
        total_frames = max(s.frame for s in scores) - min(s.frame for s in scores) + 1
    n_all = len(scores)
    passing = [s for s in scores if s.score >= min_score]
    n_pass = len(passing)
    low_quality, reason = False, None
    if low_quality_fallback and n_pass < min(num_frames, 5):
        low_quality = True
        reason = f"only {n_pass} of {n_all} candidates \u2265 min_score {min_score:.2f}"
        logger.warning("LOW QUALITY selection: %s; falling back to ranking all %d candidates by score",
                       reason, n_all)
        scores = list(scores)
    else:
        scores = passing
        if n_pass < num_frames:
            logger.warning("only %d of %d requested frames met min_score=%.2f (%d candidates in total)",
                           n_pass, num_frames, min_score, n_all)
    if not scores:
        return Selection(n_pass=n_pass, n_candidates=n_all)
    min_gap = default_min_gap(total_frames, num_frames) if min_gap_frames is None else max(0, int(min_gap_frames))

    ranked = sorted(scores, key=lambda s: (-s.score, s.frame))
    selected: list[FrameScore] = []
    for s in ranked:
        if len(selected) >= num_frames:
            break
        if _gap_ok(s.frame, selected, min_gap):
            selected.append(s)
    if len(selected) < num_frames:
        logger.info("Only %d/%d frames satisfy min_gap=%d", len(selected), num_frames, min_gap)

    # ---- yaw diversity pass ------------------------------------------------ #
    def bin_of(s: FrameScore) -> str:
        return yaw_bin(s.yaw, yaw_bins)

    best = ranked[0].score
    quota = math.ceil(num_frames * min_bin_fraction)
    eligible_bins = {bin_of(s) for s in ranked if s.score >= eligible_fraction * best}
    for b in ("frontal", "three_quarter", "profile"):
        if b not in eligible_bins:
            continue
        while True:
            counts = Counter(bin_of(s) for s in selected)
            if counts[b] >= quota:
                break
            chosen = set(id(s) for s in selected)
            pool = [s for s in ranked if id(s) not in chosen and bin_of(s) == b
                    and s.score >= eligible_fraction * best]
            if not pool:
                break
            if len(selected) < num_frames:
                cand = next((c for c in pool if _gap_ok(c.frame, selected, min_gap)), None)
                if cand is None:
                    break
                selected.append(cand)
                continue
            # Over-represented donor bins (above their quota), most crowded first.
            donors = sorted((ob for ob, n in counts.items() if ob != b and n > quota),
                            key=lambda ob: -counts[ob])
            swapped = False
            for ob in donors:
                victim = min((s for s in selected if bin_of(s) == ob), key=lambda s: s.score)
                cand = next((c for c in pool if _gap_ok(c.frame, selected, min_gap, exclude=victim)), None)
                if cand is not None:
                    selected.remove(victim)
                    selected.append(cand)
                    swapped = True
                    break
            if not swapped:
                break

    return Selection(sorted(selected, key=lambda s: s.frame), low_quality=low_quality, reason=reason,
                     n_pass=n_pass, n_candidates=n_all)

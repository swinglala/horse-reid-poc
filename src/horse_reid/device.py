"""Torch device selection."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VALID = ("auto", "cuda", "mps", "cpu")


_MPS_OK: bool | None = None


def _mps_available() -> bool:
    """MPS is reported available AND can run the ops the detectors need.

    On Intel Macs with an AMD GPU, MPS exists but ``torchvision::nms`` is not
    implemented; it only works with ``PYTORCH_ENABLE_MPS_FALLBACK=1`` set
    before torch is imported (``horse_reid/__init__.py`` does this). If torch
    was imported earlier without it, the probe fails and we use the CPU.
    """
    global _MPS_OK
    if _MPS_OK is not None:
        return _MPS_OK
    import torch

    backend = getattr(torch.backends, "mps", None)
    ok = bool(backend is not None and backend.is_available())
    if ok:
        try:
            import torchvision

            b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device="mps")
            torchvision.ops.nms(b, torch.tensor([0.9, 0.8], device="mps"), 0.5)
        except Exception as e:  # pragma: no cover - hardware dependent
            logger.warning("MPS available but unusable for detection (%s); ignoring MPS",
                           str(e).splitlines()[0])
            ok = False
    _MPS_OK = ok
    return ok


def select_device(preference: str = "auto") -> str:
    """Return the torch device string to use.

    Args:
        preference: ``"auto"`` (cuda > mps > cpu) or an explicit device. An
            explicit device that is unavailable falls back to ``"auto"`` with
            a warning. ``"cuda:N"`` style strings are accepted.

    Returns:
        One of ``"cuda"`` (or ``"cuda:N"``), ``"mps"`` or ``"cpu"``.
    """
    import torch

    pref = (preference or "auto").lower().strip()
    base = pref.split(":")[0]
    if base not in _VALID:
        raise ValueError(f"Unknown device preference {preference!r}; expected one of {_VALID}")

    if base == "cuda":
        if torch.cuda.is_available():
            return pref
        logger.warning("CUDA requested but not available; falling back to auto selection")
    elif base == "mps":
        if _mps_available():
            return "mps"
        logger.warning("MPS requested but not available; falling back to auto selection")
    elif base == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"

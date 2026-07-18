"""Facade-contract adaptation helpers for the GPUStack built-in backend.

Translates ACE-Step's native in-memory job store into the async task protocol
that the GPUStack facade (``routes/videos.py``) and its sweeper speak, and
materializes generated audio to the facade-injected NFS ``save_result_path`` with
an atomic write. These are pure, side-effect-explicit helpers; the FastAPI router
that wires them to the existing queue/store lives in ``tasks_facade_routes.py``.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

# ACE-Step store status -> GPUStack facade status. Must match the facade's
# _ENGINE_STATE_MAP exactly (gpustack/routes/videos.py): pending/processing/
# completed/failed, plus cancelled (double-L) for a DELETE'd task.
_FACADE_STATUS_MAP: Dict[str, str] = {
    "queued": "pending",
    "running": "processing",
    "succeeded": "completed",
    "failed": "failed",
}

FACADE_CANCELLED = "cancelled"

# Facade-only keys carried in the submit body that are not GenerateMusicRequest
# fields; stripped before constructing the engine request.
_FACADE_ONLY_KEYS = ("save_result_path", "task_id")


def map_status(store_status: str) -> str:
    """Map an ACE-Step store status to the GPUStack facade status string.

    Args:
        store_status: One of ``queued|running|succeeded|failed``.

    Returns:
        The facade status; unknown inputs fall back to ``processing`` (in-flight)
        rather than a terminal state, so a transient/unexpected value never looks
        done to the facade.
    """

    return _FACADE_STATUS_MAP.get(store_status, "processing")


def _part_path(dst: str) -> str:
    """Return a unique same-directory temp path for ``dst``.

    The destination extension is preserved on the temp file (so a reader that
    sniffs encoding by suffix is never handed a half-written file) and a random
    token makes concurrent writers to the same destination use distinct temp
    files — belt-and-suspenders alongside the router's materialization lock.
    """

    root, ext = os.path.splitext(dst)
    return f"{root}.{uuid4().hex}.part{ext}"


def atomic_place(src: str, dst: str) -> None:
    """Copy ``src`` onto ``dst`` atomically via a same-dir temp + ``os.replace``.

    Args:
        src: Existing local source file (the engine's produced audio).
        dst: Destination path (the facade-injected absolute NFS path).

    Raises:
        OSError: If the copy or rename fails (temp file is cleaned up first).
    """

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = _part_path(dst)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _is_within(path: str, root: str) -> bool:
    """Return whether ``path`` resolves to ``root`` or a descendant of it."""

    real_path = os.path.realpath(path)
    real_root = os.path.realpath(root)
    return real_path == real_root or real_path.startswith(real_root + os.sep)


def materialize_output(
    record: Any, save_result_path: str, output_root: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Place a succeeded job's first audio at ``save_result_path``.

    ACE-Step writes generated audio to an internal temp dir and exposes the raw
    local paths on ``record.result['raw_audio_paths']``; this copies the first
    one to the facade's NFS path atomically.

    Args:
        record: A succeeded ``_JobRecord`` (has ``.result``).
        save_result_path: Destination absolute path injected by the facade.
        output_root: Optional defense-in-depth allow-root. When set, a
            ``save_result_path`` that resolves outside it is refused (guards
            against arbitrary-write via a caller-controlled path; symlink/``..``
            escapes are caught by ``realpath``). ``None`` disables the check.

    Returns:
        ``(True, None)`` on success, else ``(False, error_message)``.
    """

    if not save_result_path:
        return False, "save_result_path missing"
    if output_root and not _is_within(save_result_path, output_root):
        return False, "save_result_path escapes output root"
    result = getattr(record, "result", None) or {}
    raw = result.get("raw_audio_paths") or []
    if not raw:
        return False, "no audio produced"
    src = raw[0]
    if not os.path.exists(src):
        return False, f"engine output missing: {src}"
    try:
        atomic_place(src, save_result_path)
    except OSError as exc:
        return False, f"materialize failed: {exc}"
    return True, None


def build_request(body: Dict[str, Any], request_model_cls: Any) -> Any:
    """Build a ``GenerateMusicRequest`` from a facade submit body.

    Strips facade-only keys and derives ``audio_format`` from the
    ``save_result_path`` extension so the engine produces the exact format the
    facade will serve; engine-specific params (bpm/lyrics/repaint_*/etc.) pass
    through unchanged and unknown keys are ignored by the pydantic model.

    Args:
        body: Parsed JSON submit body.
        request_model_cls: The ``GenerateMusicRequest`` class.

    Returns:
        A validated request instance ready to enqueue for the worker.
    """

    params = {k: v for k, v in body.items() if k not in _FACADE_ONLY_KEYS}
    ext = os.path.splitext(body.get("save_result_path", ""))[1].lstrip(".").lower()
    if ext:
        params.setdefault("audio_format", ext)
    return request_model_cls(**params)

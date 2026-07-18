"""GPUStack built-in-backend async task protocol over ACE-Step's job store.

Exposes the endpoints the GPUStack facade + sweeper poll — ``/v1/tasks/audio/``,
``/v1/tasks/{id}/status``, ``/v1/tasks/queue/status``, ``DELETE /v1/tasks/{id}``,
and ``/ready`` — reusing the existing ``job_queue`` / ``_JobStore`` / worker
unchanged; only the wire protocol and the NFS output placement are new.

Deploy with startup model init (``ACESTEP_INIT_SERVICE=true`` / ``ACESTEP_NO_INIT=
false``) so ``/ready`` reflects true readiness: lazy-load leaves ``/health`` at 200
before models exist, which would let the facade route traffic too early.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from acestep.api.http.tasks_facade_service import (
    FACADE_CANCELLED,
    build_request,
    map_status,
    materialize_output,
)


def register_tasks_facade_routes(
    app: FastAPI,
    *,
    store: Any,
    request_model_cls: Any,
) -> None:
    """Register the GPUStack async-task facade routes on ``app``.

    Args:
        app: FastAPI app (must expose ``state.job_queue``/``pending_ids``/
            ``pending_lock`` and the ``_initialized`` load flag, set in lifespan).
        store: The shared ``_JobStore`` instance.
        request_model_cls: ``GenerateMusicRequest`` class for building engine reqs.
    """

    # Per-task facade state (save path / materialized / cancelled), created once
    # here (register runs during create_app). A single lock serializes the
    # poll-driven output placement so concurrent status polls of the same
    # succeeded task cannot double-copy / race on the temp file.
    app.state.facade_tasks = {}
    app.state.facade_materialize_lock = asyncio.Lock()

    # Optional defense-in-depth: confine materialized output under this root.
    # Off by default (engine trusts the facade-injected path on the internal
    # network); set ACESTEP_OUTPUT_ROOT to reject a path that escapes it.
    output_root = os.environ.get("ACESTEP_OUTPUT_ROOT") or None

    @app.get("/ready")
    async def ready():
        """503 until models are loaded (GPUStack ``health_check_path``)."""
        if bool(getattr(app.state, "_initialized", False)):
            return {"ready": True}
        return JSONResponse({"ready": False}, status_code=503)

    @app.post("/v1/tasks/audio/")
    async def submit_audio_task(request: Request):
        """Create a music-generation task; returns task_id + save_result_path."""
        body = await request.json()
        save_result_path = body.get("save_result_path")
        if not save_result_path:
            raise HTTPException(status_code=400, detail="save_result_path is required")
        try:
            req = build_request(body, request_model_cls)
        except Exception as exc:  # noqa: BLE001 - surface any validation error as 400
            raise HTTPException(status_code=400, detail=f"invalid request: {exc}")

        queue_ref: asyncio.Queue = app.state.job_queue
        if queue_ref.full():
            raise HTTPException(status_code=503, detail="queue is full")

        record = store.create()
        app.state.facade_tasks[record.job_id] = {
            "save_result_path": save_result_path,
            "materialized": False,
            "cancelled": False,
        }
        async with app.state.pending_lock:
            app.state.pending_ids.append(record.job_id)
        await queue_ref.put((record.job_id, req))
        return {
            "task_id": record.job_id,
            "task_status": "pending",
            "save_result_path": save_result_path,
        }

    # NOTE: register the static /v1/tasks/queue/status BEFORE the dynamic
    # /v1/tasks/{task_id}/status — Starlette matches in registration order, so
    # the dynamic route would otherwise shadow it (task_id="queue") and 404.
    @app.get("/v1/tasks/queue/status")
    async def get_queue_status():
        """Queue snapshot for facade backpressure / least-pending selection."""
        stats = store.get_stats() if hasattr(store, "get_stats") else {}
        queue_ref: asyncio.Queue = app.state.job_queue
        pending = int(stats.get("queued", 0))
        active = int(stats.get("running", 0))
        return {
            "is_processing": active > 0,
            "current_task": None,
            "pending_count": pending,
            "active_count": active,
            "queue_size": queue_ref.qsize(),
            "queue_available": not queue_ref.full(),
        }

    @app.get("/v1/tasks/{task_id}/status")
    async def get_task_status(task_id: str):
        """Report facade status; on first success, place output at the NFS path.

        A 404 (unknown/evicted task_id) is intentional — it is how the facade
        sweeper detects a lost engine task and re-dispatches it.
        """
        record = store.get(task_id)
        if record is None:
            # Store has evicted the task (24h cleanup) — drop the facade meta too
            # so it doesn't leak, then 404 (the facade sweeper's re-dispatch cue).
            app.state.facade_tasks.pop(task_id, None)
            raise HTTPException(status_code=404, detail="task not found")

        meta = app.state.facade_tasks.get(task_id, {})
        save_result_path = meta.get("save_result_path")
        error = record.error
        error_type = "engine_error" if record.error else None

        if meta.get("cancelled"):
            status_str = FACADE_CANCELLED
        else:
            status_str = map_status(record.status)

        # Place the engine's output at the injected NFS path exactly once, on the
        # first poll that observes success. The lock + in-lock re-check make
        # concurrent status polls safe; a cancelled task is never materialized.
        if (
            record.status == "succeeded"
            and meta
            and not meta.get("cancelled")
            and not meta.get("materialized")
        ):
            async with app.state.facade_materialize_lock:
                if not meta.get("materialized"):
                    ok, err = await asyncio.to_thread(
                        materialize_output, record, save_result_path, output_root
                    )
                    if ok:
                        meta["materialized"] = True
                    else:
                        status_str, error, error_type = "failed", err, "materialize_error"

        return {
            "task_id": task_id,
            "status": status_str,
            "save_result_path": save_result_path,
            "error": error,
            "error_type": error_type,
            "created_at": getattr(record, "created_at", None),
            "completed_at": getattr(record, "finished_at", None),
        }

    @app.delete("/v1/tasks/{task_id}")
    async def cancel_task(task_id: str):
        """Best-effort cancel: mark cancelled so status reports it.

        ACE-Step's worker has no interrupt hook, so an in-flight generation still
        runs to completion; the result is simply reported as cancelled and its
        output is not materialized.
        """
        record = store.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        # Already terminal — report the real state instead of a misleading cancel.
        if record.status in ("succeeded", "failed"):
            return {"task_id": task_id, "status": map_status(record.status)}
        meta = app.state.facade_tasks.get(task_id)
        if meta is not None:
            meta["cancelled"] = True
        return {"task_id": task_id, "status": FACADE_CANCELLED}

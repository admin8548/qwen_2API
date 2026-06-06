from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.adapter.standard_request import StandardRequest
from backend.core.config import settings
from backend.core.request_logging import new_request_id, request_context, update_request_context
from backend.runtime.execution import build_tool_directive, build_usage_delta_factory, request_max_attempts
from backend.services.attachment_preprocessor import preprocess_attachments
from backend.services.auth_quota import resolve_auth_context
from backend.services.completion_bridge import EmptyUpstreamResponseError, run_retryable_completion_bridge
from backend.services.context_attachment_manager import derive_session_key, prepare_context_attachments
from backend.services.prompt_builder import OPENCLAW_OPENAI_PROFILE
from backend.services.qwen_client import QwenClient
from backend.services.responses_adapter import adapt_responses_request_to_chat
from backend.services.responses_formatters import ResponsesStreamTranslator, build_responses_payload, new_response_id, sse_event
from backend.services.mock_upstream import MockUpstream
from backend.services.mock_stream_patch import simulate_mock_stream
from backend.services.standard_request_builder import build_chat_standard_request
from backend.services.task_session import (
    build_openai_assistant_history_message,
    clear_invalidated_session_chat,
    log_session_plan_reuse_cancelled,
    persist_session_turn,
    plan_persistent_session_turn,
)

log = logging.getLogger("qwen2api.responses")
router = APIRouter()

_KEEPALIVE_INTERVAL = 0.5


def _store_response(app, response_id: str, payload: dict, original_request: dict, *, owner_token: str = "") -> None:
    """Persist a completed response for later retrieval (compact / get)."""
    store = getattr(app.state, "response_store", None)
    if store is not None:
        store.put(response_id, payload, original_request, owner_token=owner_token)


async def _authorize_stored_response(request: Request, entry, auth_token: str | None = None) -> str:
    """Require a valid API key and enforce response ownership for sub-endpoints."""
    if auth_token is None:
        auth = await resolve_auth_context(request, request.app.state.users_db)
        auth_token = auth.token
    owner_token = getattr(entry, "owner_token", "") or ""
    if owner_token and auth_token != owner_token and auth_token != settings.ADMIN_KEY:
        raise HTTPException(status_code=403, detail={"error": {"message": "Forbidden", "type": "forbidden"}})
    return auth_token


def _build_standard_request(req_data: dict[str, Any]) -> StandardRequest:
    standard_request = build_chat_standard_request(
        req_data,
        default_model="gpt-3.5-turbo",
        surface="responses",
        client_profile=OPENCLAW_OPENAI_PROFILE,
    )
    log.info("[Responses] normalized tools=%s", standard_request.tool_names)
    return standard_request


@router.post("/responses")
@router.post("/v1/responses")
async def responses_create(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token

    try:
        original_req_data = await request.json()
    except Exception:
        raise HTTPException(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})
    if not isinstance(original_req_data, dict):
        raise HTTPException(400, {"error": {"message": "JSON body must be an object", "type": "invalid_request_error"}})

    req_data = adapt_responses_request_to_chat(original_req_data)
    session_key = derive_session_key("responses", token, req_data)
    history_messages = req_data.get("messages", [])

    file_store = getattr(app.state, "file_store", None)
    preprocessed = None
    if file_store is not None:
        preprocessed = await preprocess_attachments(req_data, file_store, owner_token=token)
        req_data = preprocessed.payload

    context_prepared = await prepare_context_attachments(
        app=app,
        payload=req_data,
        surface="responses",
        auth_token=token,
        client_profile=OPENCLAW_OPENAI_PROFILE,
        existing_attachments=(preprocessed.attachments if preprocessed is not None else None),
    )
    req_data = context_prepared["payload"]
    standard_request = _build_standard_request(req_data)
    if preprocessed is not None:
        standard_request.attachments = preprocessed.attachments
        standard_request.uploaded_file_ids = preprocessed.uploaded_file_ids
    standard_request.upstream_files = context_prepared["upstream_files"]
    standard_request.session_key = context_prepared["session_key"]
    standard_request.context_mode = context_prepared["context_mode"]
    standard_request.bound_account_email = context_prepared["bound_account_email"]
    standard_request.bound_account = context_prepared["bound_account"]

    session_plan = await plan_persistent_session_turn(app=app, request=standard_request, payload=req_data, surface="responses")
    if session_plan.enabled:
        standard_request.persistent_session = True
        standard_request.full_prompt = session_plan.full_prompt
        standard_request.prompt = session_plan.prompt
        standard_request.session_message_hashes = session_plan.current_hashes
        standard_request.upstream_chat_id = session_plan.existing_chat_id if session_plan.reuse_chat else None
        if standard_request.bound_account is None and session_plan.account_email:
            standard_request.bound_account = await app.state.account_pool.acquire_wait_preferred(session_plan.account_email, timeout=60)
            if standard_request.bound_account is None:
                raise HTTPException(503, "No available account for persistent session")

    response_id = new_response_id()
    created_at = int(time.time())
    model_name = standard_request.resolved_model or "qwen3.6-plus"

    prompt = standard_request.full_prompt or standard_request.prompt or ""

    # Mock support for development
    mock_upstream = getattr(client.executor, "mock_upstream", None)
    if isinstance(mock_upstream, MockUpstream):
        log.info("[Responses] Using MockUpstream for testing")
        if mock_upstream.stream_mode:
            log.info("[Responses] Mock streaming mode enabled with %s chunks", len(mock_upstream.stream_chunks or []))

    if request.headers.get("accept", "").startswith("text/event-stream") or original_req_data.get("stream", False):
        async def generate():
            translator = ResponsesStreamTranslator(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                request_payload=original_req_data,
                prompt=prompt,
            )
            try:
                translator._ensure_started()
                for chunk in translator.drain():
                    yield chunk
                log.info("[Responses][stream] started response_id=%s", response_id)

                async with app.state.session_locks.hold(session_key):
                    log.info("[Responses][stream] lock_held response_id=%s", response_id)
                    try:
                        update_request_context(stream_attempt=1)

                        delta_event = asyncio.Event()
                        delta_count = 0

                        async def on_delta(evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
                            nonlocal delta_count
                            delta_count += 1
                            phase = evt.get("phase", "")
                            if tool_calls:
                                for tool_call in tool_calls:
                                    translator.on_tool_call(tool_call)
                            elif phase in ("think", "thinking_summary") and text_chunk:
                                translator.on_reasoning_chunk(text_chunk)
                            elif text_chunk:
                                translator.on_text_chunk(text_chunk)
                            delta_event.set()
                            if delta_count <= 5 or delta_count % 50 == 0:
                                log.info("[Responses][stream] delta response_id=%s phase=%s len=%s tool_calls=%s count=%s", response_id, phase, len(text_chunk or ""), len(tool_calls or []), delta_count)

                        # Mock streaming — skip real upstream entirely
                        if isinstance(mock_upstream, MockUpstream) and getattr(mock_upstream, 'stream_mode', False):
                            log.info("[Responses][stream] Simulating mock chunks")
                            simulated = await simulate_mock_stream(translator, mock_upstream, delay=0.12)
                            if simulated:
                                log.info("[Responses][stream] Mock stream simulation completed")
                                payload = build_responses_payload(
                                    response_id=response_id,
                                    created_at=created_at,
                                    model_name=model_name,
                                    prompt=prompt,
                                    execution=type("obj", (object,), {"state": type("s", (object,), {"answer_text": "".join(translator.answer_fragments) if translator.answer_fragments else mock_upstream.default_reply, "tool_calls": []})()})(),
                                    standard_request=standard_request,
                                    request_payload=original_req_data,
                                )
                                for chunk in translator.finalize(payload=payload):
                                    yield chunk
                                log.info("[Responses][stream] completed mock response_id=%s", response_id)
                                _store_response(app, response_id, payload, original_req_data)
                                return

                        # Real upstream streaming
                        bridge_task = asyncio.ensure_future(run_retryable_completion_bridge(
                            client=client,
                            standard_request=standard_request,
                            prompt=prompt,
                            users_db=users_db,
                            token=token,
                            history_messages=history_messages,
                            max_attempts=request_max_attempts(standard_request),
                            usage_delta_factory=build_usage_delta_factory(prompt),
                            allow_after_visible_output=True,
                            capture_events=False,
                            on_delta=on_delta,
                        ))

                        while not bridge_task.done():
                            delta_event.clear()
                            try:
                                await asyncio.wait_for(delta_event.wait(), timeout=_KEEPALIVE_INTERVAL)
                            except asyncio.TimeoutError:
                                pending = translator.drain_pending()
                                if pending:
                                    for chunk in pending:
                                        yield chunk
                                else:
                                    yield "event: ping\ndata: {}\n\n"
                            else:
                                for chunk in translator.drain_pending():
                                    yield chunk

                        try:
                            result = bridge_task.result() if not bridge_task.cancelled() else None
                        except asyncio.InvalidStateError:
                            log.warning("[Responses][stream] bridge_task result not set, result=None")
                            result = None
                        if result is None:
                            for chunk in translator.fail(error={"message": "upstream_streaming_failed"}):
                                yield chunk
                            return


                        log.info("[Responses][stream] bridge_done response_id=%s delta_count=%s answer_len=%s tool_calls=%s", response_id, delta_count, len(result.execution.state.answer_text), len(result.execution.state.tool_calls))
                        execution = result.execution
                        directive = result.directive or build_tool_directive(standard_request, execution.state)
                        assistant_message = build_openai_assistant_history_message(
                            execution=execution,
                            request=standard_request,
                            directive=directive,
                        )
                        await persist_session_turn(
                            app=app,
                            request=standard_request,
                            surface="responses",
                            execution=execution,
                            assistant_message=assistant_message,
                        )
                        payload = build_responses_payload(
                            response_id=response_id,
                            created_at=created_at,
                            model_name=model_name,
                            prompt=result.prompt,
                            execution=execution,
                            standard_request=standard_request,
                            request_payload=original_req_data,
                        )
                        log.info("[Responses][stream] finalize response_id=%s tool_use=%s", response_id, directive.stop_reason == "tool_use")
                        tool_blocks = directive.tool_blocks if directive.stop_reason == "tool_use" else None
                        for chunk in translator.finalize(payload=payload, tool_blocks=tool_blocks):
                            yield chunk
                        log.info("[Responses][stream] completed response_id=%s", response_id)
                        _store_response(app, response_id, payload, original_req_data, owner_token=token)
                        return
                    except HTTPException as he:
                        await clear_invalidated_session_chat(app=app, request=standard_request)
                        log.warning("[Responses][stream] fail response_id=%s reason=http_exception detail=%s", response_id, he.detail)
                        for chunk in translator.fail(error=he.detail):
                            yield chunk
                        return
                    except EmptyUpstreamResponseError as e:
                        await clear_invalidated_session_chat(app=app, request=standard_request)
                        log.warning("[Responses][stream] fail response_id=%s reason=empty_upstream", response_id)
                        for chunk in translator.fail(error={"code": "empty_upstream_response_after_retries", "message": str(e)}):
                            yield chunk
                        return
                    except Exception as e:
                        await clear_invalidated_session_chat(app=app, request=standard_request)
                        log.exception("[Responses][stream] fail response_id=%s reason=unknown", response_id)
                        for chunk in translator.fail(error={"message": str(e)}):
                            yield chunk
                        return
                    except Exception as _guard_exc:
                        log.exception("[Responses][stream] guard_fail response_id=%s", response_id)
                        for chunk in translator.fail(error={"message": str(_guard_exc)}):
                            yield chunk
                        return
            except Exception as outer_exc:
                log.exception("[Responses][stream] outer guard error")
                for chunk in translator.fail(error={"message": str(outer_exc)}):
                    yield chunk
                return

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path — mock upstream bypass
    if isinstance(mock_upstream, MockUpstream):
        log.info("[Responses][non-stream] Using MockUpstream reply")
        mock_answer = mock_upstream.default_reply
        mock_exec = type("MockExec", (), {
            "state": type("MockState", (), {
                "answer_text": mock_answer,
                "tool_calls": [],
                "finish_reason": "stop",
                "emitted_visible_output": True,
            })(),
            "chat_id": None,
            "acc": None,
        })()
        mock_payload = build_responses_payload(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            prompt=prompt,
            execution=mock_exec,
            standard_request=standard_request,
            request_payload=original_req_data,
        )
        log.info("[Responses][non-stream] mock response built, response_id=%s", response_id)
        _store_response(app, response_id, mock_payload, original_req_data)
        return JSONResponse(mock_payload)

    # Non-streaming path
    try:
        async with app.state.session_locks.hold(session_key):
            update_request_context(stream_attempt=1)
            result = await run_retryable_completion_bridge(
                client=client,
                standard_request=standard_request,
                prompt=prompt,
                users_db=users_db,
                token=token,
                history_messages=history_messages,
                max_attempts=request_max_attempts(standard_request),
                usage_delta_factory=build_usage_delta_factory(prompt),
                allow_after_visible_output=True,
            )
            execution = result.execution
            directive = result.directive or build_tool_directive(standard_request, execution.state)
            assistant_message = build_openai_assistant_history_message(
                execution=execution,
                request=standard_request,
                directive=directive,
            )
            await persist_session_turn(
                app=app,
                request=standard_request,
                surface="responses",
                execution=execution,
                assistant_message=assistant_message,
            )
            _payload = build_responses_payload(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                prompt=result.prompt,
                execution=execution,
                standard_request=standard_request,
                request_payload=original_req_data,
            )
            _store_response(app, response_id, _payload, original_req_data, owner_token=token)
            return JSONResponse(_payload)
    except EmptyUpstreamResponseError as e:
        await clear_invalidated_session_chat(app=app, request=standard_request)
        raise HTTPException(
            status_code=502,
            detail={"code": "empty_upstream_response_after_retries", "message": str(e)},
        )
    except Exception as e:
        await clear_invalidated_session_chat(app=app, request=standard_request)
        raise HTTPException(status_code=500, detail=str(e))


# ── Sub-endpoints: get / compact / input_items ────────────────────────────────

@router.get("/responses/{response_id}")
@router.get("/v1/responses/{response_id}")
async def responses_get(response_id: str, request: Request):
    """Retrieve a previously-created response by its ID."""
    store = getattr(request.app.state, "response_store", None)
    if store is None:
        raise HTTPException(503, {"error": {"message": "Response store not available", "type": "server_error"}})
    auth = await resolve_auth_context(request, request.app.state.users_db)
    entry = store.get(response_id)
    if entry is None:
        raise HTTPException(404, {"error": {"message": f"Response '{response_id}' not found", "type": "not_found_error"}})
    await _authorize_stored_response(request, entry, auth.token)
    log.info("[Responses] GET response_id=%s compacted=%s", response_id, entry.compacted)
    return JSONResponse(entry.payload)


@router.post("/responses/{response_id}/compact")
@router.post("/v1/responses/{response_id}/compact")
async def responses_compact(response_id: str, request: Request):
    """Compact / summarise the context of an existing response.

    Phase B: calls upstream model to produce a real context summary.
    Falls back to truncation if upstream is unavailable.
    """
    store = getattr(request.app.state, "response_store", None)
    if store is None:
        raise HTTPException(503, {"error": {"message": "Response store not available", "type": "server_error"}})
    auth = await resolve_auth_context(request, request.app.state.users_db)
    entry = store.get(response_id)
    if entry is None:
        raise HTTPException(404, {"error": {"message": f"Response '{response_id}' not found", "type": "not_found_error"}})
    auth_token = await _authorize_stored_response(request, entry, auth.token)

    if entry.compacted:
        log.info("[Responses] compact already applied for response_id=%s", response_id)
        return JSONResponse(entry.payload)

    # ── Phase B: real compression via upstream model ──────────
    from backend.services.response_compressor import compress_conversation
    client: QwenClient = request.app.state.qwen_client
    preferred_email = getattr(entry, "original_request", {})
    preferred_email = (
        preferred_email.get("metadata", {}).get("account_email")
        if isinstance(preferred_email.get("metadata"), dict)
        else None
    )

    try:
        compacted = await compress_conversation(
            client=client,
            original_request=entry.original_request,
            original_response=entry.payload,
            account_pool=request.app.state.account_pool,
            preferred_account_email=preferred_email,
        )
    except Exception as exc:
        log.exception("[Responses] compact compression failed for response_id=%s", response_id)
        raise HTTPException(502, {"error": {"message": f"Compression failed: {exc}", "type": "server_error"}})

    # Mark the original as compacted
    store.mark_compacted(response_id, compacted)
    # Store the new compacted response under its own ID
    store.put(compacted["id"], compacted, entry.original_request, owner_token=getattr(entry, "owner_token", "") or auth_token)

    log.info("[Responses] compact done: old_id=%s new_id=%s", response_id, compacted["id"])
    return JSONResponse(compacted)


@router.get("/responses/{response_id}/input_items")
@router.get("/v1/responses/{response_id}/input_items")
async def responses_input_items(response_id: str, request: Request):
    """List input items for an existing response (stub).

    Returns the original request's `input` field wrapped in the
    standard list envelope expected by Responses API clients.
    """
    store = getattr(request.app.state, "response_store", None)
    if store is None:
        raise HTTPException(503, {"error": {"message": "Response store not available", "type": "server_error"}})
    auth = await resolve_auth_context(request, request.app.state.users_db)
    entry = store.get(response_id)
    if entry is None:
        raise HTTPException(404, {"error": {"message": f"Response '{response_id}' not found", "type": "not_found_error"}})
    await _authorize_stored_response(request, entry, auth.token)

    raw_input = entry.original_request.get("input", "")
    if isinstance(raw_input, str):
        items = [{
            "id": f"item_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": raw_input}],
            "status": "completed",
        }]
    elif isinstance(raw_input, list):
        items = []
        for item in raw_input:
            if isinstance(item, str):
                items.append({
                    "id": f"item_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": item}],
                    "status": "completed",
                })
            elif isinstance(item, dict):
                # Ensure every item has an id for the list envelope
                normalized = dict(item)
                if "id" not in normalized:
                    normalized["id"] = f"item_{uuid.uuid4().hex[:24]}"
                if "status" not in normalized:
                    normalized["status"] = "completed"
                items.append(normalized)
    else:
        items = []

    return JSONResponse({
        "object": "list",
        "data": items,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
        "has_more": False,
    })

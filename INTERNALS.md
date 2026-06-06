# qwen2API Internal Runtime Notes

This document records the main runtime mechanisms referenced by `README.md`.

## Request flow

1. Compatible API routes normalize client payloads into `StandardRequest`.
2. `AccountPool` selects an available upstream account and enforces concurrency/cooldown.
3. `QwenClient` / `QwenExecutor` create or reuse an upstream chat and stream events.
4. Runtime collectors translate upstream events into OpenAI / Anthropic / Gemini compatible responses.
5. Temporary chats are deleted unless the request is explicitly using a persistent session.

## Chat ID prewarm pool

`ChatIdPool` is optional and controlled by `CHAT_ID_POOL_ENABLED`.
It stores pre-created chat IDs by `(account email, model)`, flushes bad buckets after empty upstream responses, and applies a short failure cooldown to avoid reusing a suspect prewarm batch.

## Responses API store

Completed `/v1/responses` results are kept in an in-memory `ResponseStore` for retrieval, `input_items`, and compact operations. Stored responses include the owner API token so sub-endpoints can enforce authenticated access.

## Logging

Request logs redact authorization material. Health probes are logged at debug level to avoid hiding real warnings.

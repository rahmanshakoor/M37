# OpenRouter fixtures

Synthetic `POST /api/v1/chat/completions` responses for `tests/test_providers.py`,
shaped after the OpenRouter OpenAPI (`ChatResult`, `ChatChoice`, `ChatToolCall`,
`ChatUsage`, `OpenRouterMetadata`) and the tool-calling, structured-output and error
pages the probe read on 2026-09-13. None was recorded from a live call — the probe
ran without a key — and none carries anything but a toy arithmetic conversation
("compute 2+3 and 10+20 with the add tool").

| file | what it stands for |
| --- | --- |
| `tools_round1.json` | `finish_reason: tool_calls` with two parallel `add` calls and signed `reasoning_details` |
| `tools_round2.json` | a second tool turn: one good `add`, one unknown tool, one call whose arguments are not JSON |
| `tools_done.json` | the model stops calling tools (`stop` / `end_turn`) |
| `final_answer.json` | the `response_format: json_schema` answer, JSON as a string in `message.content` |
| `final_truncated.json` | the final answer cut off (`length` / `max_tokens`) |
| `refusal.json` | an Anthropic refusal relayed as `content_filter` with `message.refusal` |
| `error_late_200.json` | HTTP 200 whose body carries only `id` + `error` (late upstream failure) |
| `error_401.json` | the live-verified 401 envelope for an unknown key |
| `error_429.json` | the documented 429 envelope with `metadata.error_type` and a `retry-after` header |
| `error_500.json` | a 500 envelope — retried by the client, like the 429 |
| `error_504.json` | a 504 envelope — a timeout by another name, never re-sent |
| `final_stream.sse` | the final answer as `stream: true` delivers it: keep-alive comments, `delta.content` pieces, the finish chunk with `usage`, `[DONE]` |
| `final_stream_truncated.sse` | a streamed answer cut off (`length` / `max_tokens`) |
| `final_stream_error.sse` | the documented mid-stream failure: a chunk with `error` and `finish_reason: "error"` |
| `final_stream_dropped.sse` | a stream that ends without a finish reason or `[DONE]` |

A `.json` file is `{"status", "headers", "body"}` and the stub transport in the test
returns `body` as the response text, or raises `HttpError` for a status ≥ 400 exactly
as `engine.retrieve.http.Http` does; a `.sse` file is the raw `text/event-stream` body
of a streamed 200, shaped after OpenRouter's streaming guide (chunk objects, the
`: OPENROUTER PROCESSING` keep-alive comment, usage on the last chunk, an `error`
field on a failed chunk) — documented shape, not a recorded exchange.

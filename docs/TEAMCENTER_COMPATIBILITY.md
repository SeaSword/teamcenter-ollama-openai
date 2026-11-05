# Teamcenter compatibility / Teamcenter 兼容性

[English README](../README.md) · [中文说明](../README.zh-CN.md)

## English

The adapter supports two distinct protocol paths:

| Client | Adapter path | Behavior |
| --- | --- | --- |
| BOM Agent / Ollama SDK | `/api/chat` | Translate requests and emit Ollama Chat JSON/NDJSON. |
| Copilot / OpenAI client | `/v1/chat/completions` | Preserve the request body and successful backend response. |

Ollama conversion moves system messages to the beginning and merges multiple system prompts in their original order. Other messages retain their ordering. Tool definitions retain descriptions, nested properties, required fields, enums, and empty schemas. No business prompt is rewritten.

Each response has its own streaming state. Tool fragments are associated by index/ID and emitted once with parsed argument objects after completion. For tool history, retained IDs take priority; otherwise, the adapter generates IDs and matches results by tool name or conversation order. If a client omits both IDs and names, parallel results must retain call order. Ambiguous or unmatched history cannot be reconstructed reliably.

Normal Chat chunks contain `message.role` and `message.content`; reasoning uses `message.thinking`. A final object carries `done:true` and available token counts. Generate keeps its separate `response` field. Errors after streaming begins use a standalone `{"error":"..."}` NDJSON object, as expected by Ollama clients.

`MAX_RETRIES` counts additional attempts after the first request: zero still performs one request. A partially delivered stream is never restarted. Upstream HTTP status errors are detected before an HTTP 200 stream when possible. Diagnostic logs contain IDs, model/path, status, and exception type, without full prompts, arguments, credentials, or raw upstream error bodies.

The Navigation agent may choose ordinary text instead of a tool. The adapter does not turn that text into a fabricated dispatch result. Completion logs expose `tool_call_count`, `tool_count`, and `text_chunk_count`. An explicit `tool_choice` extension is forwarded only when supplied; any production forcing policy must be scoped to the relevant final stage after required confirmation.

The offline suite covers 34 cases, including optional in-memory comparisons against private captures. Raw traffic, backend secrets, runtime output, local deployment mappings, and operational reports are excluded from this public repository. Without private captures, the optional comparison is skipped; the other 33 tests still run.

Live verification should include a fresh BOM conversation, a controlled tool operation, a result-fed follow-up, Navigation behavior, and an existing Copilot knowledge-base question with citations. Mock success alone does not establish live compatibility. Actual token counts and GPU memory requirements depend on the deployed backend and model; use `/tokenize` rather than deriving tokens from request bytes. Thinking configuration is not changed by these fixes.

## 中文

适配器区分 BOM Agent 的 Ollama 路径 `/api/chat` 和 Copilot 的 OpenAI 路径 `/v1/chat/completions`。前者进行请求与响应转换，后者保留原请求正文和成功的后端响应。

Ollama 消息转换把 System 放到开头，多个 System 按原顺序合并；其他消息不改变先后顺序。工具描述、嵌套属性、required、enum 和空 Schema 均保留，不重写业务 Prompt。

每个响应独立维护流状态，按照 index/ID 累积工具片段。完成后解析为参数对象，只输出一次。历史关联优先使用显式 ID；若客户端丢弃 ID，则生成 ID，按工具名称或历史顺序匹配结果。客户端同时缺少 ID 和名称时，并行工具结果必须保持调用顺序，无法可靠恢复已丢失的关联信息。

Chat 数据块使用 `message.role/content`，reasoning 使用独立的 `message.thinking`，最终块含 `done:true` 和可用的 Token 统计。Generate 继续使用 `response`。流启动后发生错误时，输出独立的 `{"error":"..."}` NDJSON 对象。

`MAX_RETRIES` 表示首次请求以外的重试次数，0 仍会请求一次。流已开始后不重新发送，避免重复工具调用。上游 HTTP 错误会尽量在发送 200 前识别。日志只保留 Request ID、模型、路径、状态和异常类型，不记录完整 Prompt、工具参数、凭据或上游错误正文。

Navigation Agent 可能只输出普通文字，代理不会伪造 dispatch_action。可通过完成日志中的 `tool_call_count`、`tool_count`、`text_chunk_count` 区分模型工具选择与协议问题。显式 `tool_choice` 扩展只有调用方提供时才转发；真实业务的强制策略应仅针对完成必要确认后的最终阶段。

离线回归覆盖 34 项，包含可选的私有抓包内存比较。公开仓库不包含原始流量、后端密钥、运行结果、本地模型映射和运维报告。没有私有抓包时，明确跳过该项比较，其他 33 项仍运行。

现场验收需要新建 BOM 对话，完成受控工具操作、结果回传后的下一轮、Navigation 行为，以及既有 Copilot 知识库问答和引用核对。Mock 通过不能替代真实集成结果。实际 Token 与显存需求应在部署后端计数验证；不能用请求字节数推算。本次兼容修复不会改变 Thinking 配置。

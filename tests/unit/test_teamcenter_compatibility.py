"""Offline protocol regression tests, runnable with unittest or pytest.

All requests are synthetic. Optional captures are read in memory, never copied
or printed. No test sends a request to a GPU or uses a real API key.
"""

import copy
import asyncio
import json
import logging
import os
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import FastAPI

with patch.dict(os.environ, {
    "OPENAI_API_BASE_URL": "http://mock-vllm.invalid/v1",
    "OPENAI_API_KEY": "offline-test-key",
    "MAX_RETRIES": "0",
    "REQUEST_TIMEOUT": "10",
}):
    from src.models import OllamaChatRequest, OllamaChatResponse
    from src.routers import chat
    from src.translators.chat import ChatStreamState, ChatTranslator
    from src.utils.http_client import RetryClient


MODEL = "qwen3.8-27b"
DISPATCH = {
    "type": "function",
    "function": {
        "name": "dispatch_action",
        "description": "Dispatch a navigation action (synthetic fixture).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "refreshglobalchangecontext", "none"]},
                "tool_name": {"type": "string"},
                "suggestions": {},
            },
            "required": ["action", "tool_name"],
        },
    },
}


def request_body(stream=True, **kwargs):
    return {"model": MODEL, "messages": [{"role": "user", "content": "在此可做什么？"}], "stream": stream, **kwargs}


def completion(content="你好", calls=None, **extra):
    return {
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {}), **extra}, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": 37, "completion_tokens": 9, "total_tokens": 46},
    }


def event(delta=None, finish=None, **extra):
    return {"model": MODEL, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **extra}


def tool_delta(index=0, name=None, args=None, call_id=None):
    function = {}
    if name is not None:
        function["name"] = name
    if args is not None:
        function["arguments"] = args
    return {"index": index, **({"id": call_id} if call_id else {}), "function": function}


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, parts, failure=None):
        self.parts = parts
        self.failure = failure
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            yield part
        if self.failure:
            raise self.failure

    async def aclose(self):
        self.closed = True


def sse(events, done=True, split=False, failure=None):
    raw = b"".join(("data: " + json.dumps(e, ensure_ascii=False) + "\r\n\r\n").encode("utf-8") for e in events)
    if done:
        raw += b"data: [DONE]\r\n\r\n"
    parts = [raw[i:i+1] for i in range(len(raw))] if split else [raw]
    return ByteStream(parts, failure)


class TeamcenterProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seen = []
        self.replies = []
        self.streams = []
        self.settings = patch.object(chat, "settings", type("Settings", (), {
            "OPENAI_API_BASE_URL": "http://mock-vllm.invalid/v1", "OPENAI_API_KEY": "offline-backend-key", "REQUEST_TIMEOUT": 10,
        })())
        self.settings.start()
        with patch("src.utils.http_client.get_settings") as settings:
            settings.return_value.MAX_RETRIES = 0
            settings.return_value.REQUEST_TIMEOUT = 10
            settings.return_value.DISABLE_SSL_VERIFICATION = False
            self.retry = RetryClient(max_retries=0, base_delay=0, jitter=False)
        await self.retry.client.aclose()

        def backend(req):
            self.seen.append((json.loads(req.content), dict(req.headers)))
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, ByteStream):
                self.streams.append(reply)
                return httpx.Response(200, stream=reply, headers={"content-type": "text/event-stream"})
            status, body = reply
            return httpx.Response(status, json=body)

        self.retry.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))

        @asynccontextmanager
        async def client_context():
            yield self.retry

        self.context = patch.object(chat, "retry_client_context", client_context)
        self.context.start()
        app = FastAPI()
        app.include_router(chat.router, prefix="/api")
        app.include_router(chat.router, prefix="/v1")

        @app.middleware("http")
        async def request_id(req, call_next):
            req.state.request_id = req.headers.get("x-request-id", "offline-request")
            return await call_next(req)

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://adapter")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.retry.close()
        self.context.stop()
        self.settings.stop()

    async def post(self, body, path="/api/chat"):
        return await self.client.post(path, json=body, headers={"authorization": "Bearer ollama", "x-request-id": "offline-request"})

    def chat_lines(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        lines = [json.loads(line) for line in response.text.splitlines() if line]
        self.assertTrue(lines)
        for line in lines:
            if "error" not in line:
                OllamaChatResponse.model_validate(line)
                self.assertIn("message", line)
                self.assertNotIn("response", line)
        return lines

    async def test_T1_nonstream_chat(self):
        self.replies = [(200, completion())]
        response = await self.post(request_body(False))
        self.assertEqual(response.status_code, 200)
        result = OllamaChatResponse.model_validate(response.json())
        self.assertEqual(result.message.content, "你好")
        self.assertEqual(result.prompt_eval_count, 37)
        self.assertTrue(result.done)
        self.assertEqual(self.seen[0][1]["authorization"], "Bearer offline-backend-key")

    async def test_T2_fragmented_utf8_sse_and_final_usage(self):
        self.replies = [sse([
            event({"role": "assistant", "content": None}), event({"content": "你好"}),
            event({"content": "，世界"}), event(finish="stop"),
            {"model": MODEL, "choices": [], "usage": {"prompt_tokens": 37, "completion_tokens": 9}},
        ], split=True)]
        lines = self.chat_lines(await self.post(request_body()))
        self.assertEqual("".join(x["message"]["content"] for x in lines), "你好，世界")
        self.assertEqual(sum(x["done"] for x in lines), 1)
        self.assertTrue(lines[-1]["done"])
        self.assertEqual(lines[-1]["prompt_eval_count"], 37)
        self.assertEqual(lines[-1]["eval_count"], 9)
        self.assertEqual(lines[-1]["done_reason"], "stop")
        self.assertTrue(self.streams[0].closed)

    async def test_T3_system_order_and_merge(self):
        body = request_body(False)
        body["messages"] += [{"role": "system", "content": "first"}, {"role": "assistant", "content": "hello"}, {"role": "system", "content": "second"}]
        original = copy.deepcopy(body)
        self.replies = [(200, completion())]
        self.assertEqual((await self.post(body)).status_code, 200)
        messages = self.seen[0][0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[0]["content"], "first\n\nsecond")
        self.assertEqual(body, original)

    async def test_T4_all_38_tools_and_schemas_survive(self):
        tools = [copy.deepcopy(DISPATCH)]
        for i in range(37):
            tools.append({"type": "function", "function": {"name": f"synthetic_tool_{i}", "description": "Synthetic tool", "parameters": {
                "type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"mode": {"enum": ["a", "b"]}}, "required": ["mode"]}}, "optional": {}},
            }}})
        self.replies = [(200, completion())]
        response = await self.post(request_body(False, tools=tools, options={"temperature": 0, "top_p": 0.8, "num_predict": 50}))
        self.assertEqual(response.status_code, 200)
        upstream = self.seen[0][0]
        self.assertEqual(upstream["tools"], tools)
        self.assertEqual(upstream["temperature"], 0)
        self.assertEqual(upstream["max_tokens"], 50)
        self.assertNotIn("tool_choice", upstream)

    async def test_T5_dispatch_action_stream(self):
        self.replies = [sse([
            event({"tool_calls": [tool_delta(name="dispatch_action", args='{"action":"none","tool_name":""}', call_id="call_dispatch")]}),
            event(finish="tool_calls"),
        ])]
        lines = self.chat_lines(await self.post(request_body(tools=[DISPATCH])))
        calls = [tc for line in lines for tc in line["message"].get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "dispatch_action")
        self.assertEqual(calls[0]["function"]["arguments"], {"action": "none", "tool_name": ""})
        self.assertEqual(calls[0]["function"]["index"], 0)
        self.assertEqual(lines[-1]["done_reason"], "tool_calls")

    async def test_T6_interleaved_tools_and_fragmented_names_arguments(self):
        self.replies = [sse([
            event({"content": "Working", "tool_calls": [tool_delta(0, "dispatch_", '{"action":', "call_a"), tool_delta(1, "lookup", '{"ids":', "call_b")]}),
            event({"tool_calls": [tool_delta(1, args='[1,2]}'), tool_delta(0, "action", '"none","tool_name":""}')] }),
            event(finish="tool_calls"),
        ], split=True)]
        lines = self.chat_lines(await self.post(request_body()))
        calls = [tc for line in lines for tc in line["message"].get("tool_calls", [])]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["function"]["name"], "dispatch_action")
        self.assertEqual(calls[1]["function"]["arguments"], {"ids": [1, 2]})
        self.assertEqual("".join(line["message"]["content"] for line in lines), "Working")

    async def test_T7_tool_result_next_round_without_sdk_preserved_id(self):
        self.replies = [sse([event({"tool_calls": [tool_delta(name="dispatch_action", args='{"action":"none","tool_name":""}', call_id="original_id")]}), event(finish="tool_calls")])]
        first = self.chat_lines(await self.post(request_body(tools=[DISPATCH])))
        call = next(line["message"]["tool_calls"][0] for line in first if line["message"].get("tool_calls"))
        # Ollama SDK versions may retain only function.name/arguments.
        body = request_body(False, tools=[DISPATCH])
        body["messages"] += [{"role": "assistant", "content": "", "tool_calls": [{"function": call["function"]}]}, {"role": "tool", "tool_name": "dispatch_action", "content": '{"ok":true}'}]
        self.replies = [(200, completion("完成"))]
        response = await self.post(body)
        self.assertEqual(response.json()["message"]["content"], "完成")
        history = self.seen[-1][0]["messages"]
        self.assertEqual(history[-1]["tool_call_id"], history[-2]["tool_calls"][0]["id"])
        self.assertEqual(json.loads(history[-2]["tool_calls"][0]["function"]["arguments"]), {"action": "none", "tool_name": ""})
        self.assertEqual(history[-1]["content"], '{"ok":true}')

    async def test_T7_explicit_ids_and_out_of_order_named_results(self):
        body = request_body(False)
        body["messages"] += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "function": {"name": "first", "arguments": {"x": 1}}},
                {"id": "b", "function": {"name": "second", "arguments": {"x": 2}}},
            ]},
            {"role": "tool", "tool_name": "second", "content": "two"},
            {"role": "tool", "tool_call_id": "a", "content": "one"},
        ]
        self.replies = [(200, completion())]
        self.assertEqual((await self.post(body)).status_code, 200)
        history = self.seen[-1][0]["messages"]
        self.assertEqual([m["tool_call_id"] for m in history[-2:]], ["b", "a"])

    async def test_T8_null_content_tools_and_reasoning_nonstream(self):
        calls = [{"id": "a", "type": "function", "function": {"name": "dispatch_action", "arguments": '{"action":"none","tool_name":""}'}}]
        self.replies = [(200, completion(None, calls, reasoning_content="private reasoning"))]
        response = await self.post(request_body(False))
        result = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["message"]["content"], "")
        self.assertEqual(result["message"]["thinking"], "private reasoning")
        self.assertIsInstance(result["message"]["tool_calls"][0]["function"]["arguments"], dict)

    async def test_T8_reasoning_stream_is_separate(self):
        self.replies = [sse([event({"content": None, "reasoning_content": "reason"}), event({"content": "answer"}), event(finish="stop")])]
        lines = self.chat_lines(await self.post(request_body()))
        self.assertEqual("".join(x["message"].get("thinking", "") for x in lines), "reason")
        self.assertEqual("".join(x["message"]["content"] for x in lines), "answer")

    async def test_T9_http_errors_before_200_and_safe_logs(self):
        for streaming in (False, True):
            for code in (400, 401, 422, 500):
                with self.subTest(stream=streaming, code=code):
                    self.replies = [(code, {"error": {"message": "sensitive prompt Bearer credential", "type": "BadRequestError"}})]
                    with self.assertLogs(chat.logger, logging.ERROR) as captured:
                        response = await self.post(request_body(streaming))
                    self.assertEqual(response.status_code, code)
                    self.assertIn("error", response.json())
                    self.assertNotIn("sensitive prompt", str(captured.output) + response.text)
                    record = next(r for r in captured.records if getattr(r, "extra_data", {}).get("upstream_status") == code)
                    self.assertEqual(record.extra_data["request_id"], "offline-request")
                    self.assertEqual(record.extra_data["path"], "/api/chat")
                    self.assertEqual(record.extra_data["model"], MODEL)
                    self.assertIn("exception_type", record.extra_data)

    async def test_T9_network_timeout_empty_and_inband_errors(self):
        for failure, code in [(httpx.ConnectError("private details"), 502), (httpx.ReadTimeout("private details"), 504)]:
            self.replies = [failure]
            response = await self.post(request_body())
            self.assertEqual(response.status_code, code)
            self.assertNotIn("private details", response.text)
        self.replies = [sse([], done=False)]
        self.assertEqual((await self.post(request_body())).status_code, 502)
        self.replies = [sse([{"error": {"message": "private details"}}])]
        self.assertEqual((await self.post(request_body())).status_code, 502)

    async def test_T10_copilot_schema_auth_chinese_and_metadata_unchanged(self):
        body = request_body(False)
        body["messages"] += [{"role": "system", "content": "preserve original order"}]
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}}}
        body["vendor_extension"] = {"citations": True}
        upstream = completion('{"answer":"中文知识库回答"}')
        upstream["choices"][0]["message"]["citations"] = [{"source": "synthetic-source", "page": 3}]
        upstream["retrieval_metadata"] = {"refs": ["synthetic-ref"]}
        self.replies = [(200, upstream)]
        response = await self.post(body, "/v1/chat/completions")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), upstream)
        self.assertEqual(self.seen[0][0], body)
        self.assertEqual(self.seen[0][1]["authorization"], "Bearer offline-backend-key")

    async def test_T10_copilot_sse_bytes_preserved(self):
        stream = sse([event({"content": "中文"}), event(finish="stop")], split=True)
        expected = b"".join(stream.parts)
        self.replies = [stream]
        response = await self.post(request_body(), "/v1/chat/completions")
        self.assertEqual(response.content, expected)

    async def test_T11_eof_after_finish_and_premature_eof(self):
        self.replies = [sse([event({"content": "ok"}), event(finish="stop")], done=False)]
        lines = self.chat_lines(await self.post(request_body()))
        self.assertTrue(lines[-1]["done"])
        self.replies = [sse([event({"content": "partial"})], done=False)]
        lines = self.chat_lines(await self.post(request_body()))
        self.assertIn("error", lines[-1])
        self.assertFalse(any(line.get("done") for line in lines))

    async def test_T11_midstream_failure_not_retried(self):
        self.retry.max_retries = 3
        self.replies = [ByteStream([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'], httpx.ReadError("private details"))]
        lines = self.chat_lines(await self.post(request_body()))
        self.assertIn("error", lines[-1])
        self.assertNotIn("private details", lines[-1]["error"])
        self.assertEqual(len(self.seen), 1)
        self.assertTrue(self.streams[0].closed)

    async def test_T11_invalid_arguments_are_never_executed(self):
        self.replies = [sse([event({"content": "working"}), event({"tool_calls": [tool_delta(name="dispatch_action", args='{"action":', call_id="bad")]}), event(finish="tool_calls")])]
        lines = self.chat_lines(await self.post(request_body()))
        self.assertIn("error", lines[-1])
        self.assertFalse(any(x.get("message", {}).get("tool_calls") for x in lines))

    async def test_generate_keeps_response_field(self):
        self.replies = [sse([event({"content": "text"}), event(finish="stop")])]
        response = await self.post({"model": MODEL, "prompt": "hello", "stream": True}, "/api/generate")
        lines = [json.loads(x) for x in response.text.splitlines()]
        self.assertTrue(all("response" in x and "message" not in x for x in lines))
        self.assertTrue(lines[-1]["done"])

    async def test_navigation_text_is_not_fabricated_as_tool_call(self):
        self.replies = [sse([event({"content": "navigation text"}), event(finish="stop")])]
        with self.assertLogs(chat.logger, logging.INFO) as captured:
            lines = self.chat_lines(await self.post(request_body(tools=[DISPATCH])))
        self.assertFalse(any(line["message"].get("tool_calls") for line in lines))
        self.assertTrue(any(getattr(r, "extra_data", {}).get("tool_call_count") == 0 for r in captured.records))

    async def test_retry_zero_means_one_attempt(self):
        self.replies = [(200, completion())]
        self.assertEqual((await self.post(request_body(False))).status_code, 200)
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.retry.max_retries, 0)

    async def test_retry_budget_and_transient_status(self):
        self.retry.max_retries = 1
        self.replies = [(500, {}), sse([event({"content": "ok"}), event(finish="stop")])]
        self.chat_lines(await self.post(request_body()))
        self.assertEqual(len(self.seen), 2)

    async def test_null_nonstream_without_calls(self):
        self.replies = [(200, completion(None))]
        response = await self.post(request_body(False))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"]["content"], "")

    async def test_invalid_nonstream_arguments_are_upstream_error(self):
        self.replies = [(200, completion(None, [{"function": {"name": "dispatch_action", "arguments": '{"action":'}}]))]
        response = await self.post(request_body(False))
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.json())

    async def test_legacy_function_call_stream(self):
        self.replies = [sse([event({"function_call": {"name": "dispatch_action", "arguments": '{"action":'}}), event({"function_call": {"arguments": '"none","tool_name":""}'}}), event(finish="function_call")])]
        lines = self.chat_lines(await self.post(request_body()))
        calls = [call for line in lines for call in line["message"].get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["arguments"], {"action": "none", "tool_name": ""})

    async def test_client_http_errors_do_not_open_circuit(self):
        self.replies = [(400, {})] * 6 + [(200, completion())]
        for _ in range(6):
            self.assertEqual((await self.post(request_body())).status_code, 400)
        self.assertEqual((await self.post(request_body(False))).status_code, 200)
        self.assertFalse(self.retry.circuit_breaker.is_open)

    async def test_concurrent_streams_keep_separate_arguments(self):
        self.replies = [sse([event({"tool_calls": [tool_delta(name="first", args='{"id":1}', call_id="one")]}), event(finish="tool_calls")], split=True), sse([event({"tool_calls": [tool_delta(name="second", args='{"id":2}', call_id="two")]}), event(finish="tool_calls")], split=True)]
        responses = await asyncio.gather(self.post(request_body()), self.post(request_body()))
        calls = [[tc for line in self.chat_lines(response) for tc in line["message"].get("tool_calls", [])] for response in responses]
        self.assertEqual([len(x) for x in calls], [1, 1])
        self.assertEqual({x[0]["function"]["name"] for x in calls}, {"first", "second"})

    async def test_T7_second_tool_decision_then_final_response(self):
        body = request_body(False)
        for number in range(2):
            self.replies = [(200, completion(None, [{"function": {"name": "lookup", "arguments": json.dumps({"step": number})}}]))]
            response = await self.post(body)
            self.assertEqual(response.status_code, 200)
            body["messages"].append(response.json()["message"])
            body["messages"].append({"role": "tool", "tool_name": "lookup", "content": f"result {number}"})
        self.replies = [(200, completion("done"))]
        self.assertEqual((await self.post(body)).json()["message"]["content"], "done")
        history = self.seen[-1][0]["messages"]
        ids = [m["tool_calls"][0]["id"] for m in history if m.get("tool_calls")]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual([m["tool_call_id"] for m in history if m["role"] == "tool"], ids)

    async def test_sse_multiline_comments_and_no_final_newline(self):
        self.replies = [ByteStream([b': ping\r\nevent: message\r\ndata: {"choices":\r\ndata: [{"delta":{"content":"ok"}}]}\r\n\r\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\r\n\r\ndata:[DONE]'])]
        lines = self.chat_lines(await self.post(request_body()))
        self.assertEqual("".join(x["message"]["content"] for x in lines), "ok")
        self.assertTrue(lines[-1]["done"])

    async def test_generator_close_releases_upstream(self):
        stream = sse([event({"content": "first"}), event(finish="stop")], split=True)
        self.replies = [stream]
        original = OllamaChatRequest(**request_body())
        iterator = chat.stream_response(self.retry, ChatTranslator().translate_request(original), original)
        await anext(iterator)
        self.assertFalse(stream.closed)
        await iterator.aclose()
        self.assertTrue(stream.closed)

    async def test_actual_main_app_middleware(self):
        with patch("src.utils.logging.setup_logging"):
            from src.main import app
        self.replies = [sse([event({"content": "你好"}), event(finish="stop")])]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://adapter") as client:
            response = await client.post("/api/chat", json=request_body(), headers={"X-Request-ID": "main-app-test", "Authorization": "Bearer ollama"})
        self.assertEqual(response.headers["x-request-id"], "main-app-test")
        self.assertTrue(self.chat_lines(response)[-1]["done"])
        self.replies = [sse([event({"content": "hello"}), event(finish="stop")])]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://adapter") as client:
            with self.assertLogs(chat.logger, logging.INFO) as captured:
                response = await client.post("/api/chat", json=request_body())
        completed = next(record for record in captured.records if record.getMessage() == "Backend chat stream completed")
        self.assertEqual(completed.extra_data["request_id"], response.headers["x-request-id"])


class TeamcenterTranslationTests(unittest.TestCase):
    def test_tool_fragments_correlated_by_id_without_index(self):
        translator = ChatTranslator()
        request = OllamaChatRequest(**request_body())
        state = ChatStreamState()
        translator.translate_streaming_response(event({"tool_calls": [
            {"id": "a", "function": {"name": "first", "arguments": '{"id":'}},
            {"id": "b", "function": {"name": "second", "arguments": '{"id":'}},
        ]}), request, state=state)
        translator.translate_streaming_response(event({"tool_calls": [
            {"id": "b", "function": {"arguments": '2}'}},
            {"id": "a", "function": {"arguments": '1}'}},
        ]}), request, state=state)
        result = translator.translate_streaming_response(event(finish="tool_calls"), request, state=state)
        self.assertEqual([c["function"]["arguments"] for c in result["message"]["tool_calls"]], [{"id": 1}, {"id": 2}])
        final = translator.translate_streaming_response("[DONE]", request, state=state)
        self.assertNotIn("tool_calls", final["message"])
        self.assertIsNone(translator.translate_streaming_response("[DONE]", request, state=state))

    def test_tool_choice_is_explicit_and_schema_format_is_preserved(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        request = OllamaChatRequest(**request_body(False, format=schema, tool_choice={"type": "function", "function": {"name": "dispatch_action"}}, tools=[DISPATCH]))
        translated = ChatTranslator().translate_request(request)
        self.assertEqual(translated.response_format["json_schema"]["schema"], schema)
        self.assertEqual(translated.tool_choice, request.tool_choice)

    def test_invalid_tool_field_identified_without_values(self):
        bad = copy.deepcopy(DISPATCH)
        bad["function"]["parameters"]["required"] = "private-invalid-value"
        with self.assertRaises(Exception) as raised:
            ChatTranslator().translate_request(OllamaChatRequest(**request_body(False, tools=[bad])))
        self.assertIn("dispatch_action", str(raised.exception))
        self.assertIn("required", str(raised.exception))
        self.assertNotIn("private-invalid-value", str(raised.exception))

    def test_optional_real_capture_requests_in_memory(self):
        capture_dir = os.environ.get("TEAMCENTER_CAPTURE_DIR")
        if not capture_dir:
            self.skipTest("TEAMCENTER_CAPTURE_DIR is not set")
        for filename, count in [("vllmtoollama1.txt", 38), ("vllmtoollama2.txt", 3)]:
            text = (Path(capture_dir) / filename).read_text(encoding="utf-8-sig")
            body = next(json.loads(line) for line in text.splitlines() if line.startswith('{"model"'))
            request = {"model": body["model"], "messages": body["messages"], "tools": body["tools"], "stream": body["stream"], "options": {k: body[k] for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty") if k in body}}
            original = copy.deepcopy(request)
            translated = ChatTranslator().translate_request(OllamaChatRequest(**request)).model_dump(exclude_none=True)
            self.assertEqual(len(translated["tools"]), count)
            # Avoid assertions that would dump any real tool definitions on failure.
            self.assertTrue(translated["tools"] == body["tools"], "Tool definitions changed")
            self.assertEqual(translated["messages"][0]["role"], "system")
            self.assertTrue(request == original, "Input was mutated")


if __name__ == "__main__":
    unittest.main(verbosity=2)

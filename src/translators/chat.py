"""
Chat translator for converting between Ollama and OpenAI chat formats.

This module handles the translation of chat completion requests and responses
between Ollama and OpenAI formats for Phase 2 (including tool calling and image support).
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from src.models import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    # Ollama models
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIFunction,
    OpenAIMessage,
    OpenAIStreamResponse,
    OpenAITool,
)
from src.translators.base import BaseTranslator
from src.utils.exceptions import TranslationError, ValidationError
from src.utils.logging import get_logger

# Type aliases for clarity
OllamaResponse = Union[OllamaGenerateResponse, OllamaChatResponse]
OllamaStreamResponse = OllamaResponse


@dataclass
class ChatStreamState:
    """State belongs to one response, never to the shared translator."""

    calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    ids: Dict[str, int] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    saw_choice: bool = False
    emitted_calls: bool = False
    finished: bool = False
    text_chunks: int = 0

    def accumulate(self, deltas: List[Dict[str, Any]]) -> None:
        if self.emitted_calls:
            raise TranslationError("Received tool fragments after tool completion")
        for delta in deltas:
            call_id = delta.get("id")
            index = delta.get("index")
            if index is None:
                if call_id in self.ids:
                    index = self.ids[call_id]
                elif call_id:
                    index = max(self.calls, default=-1) + 1
                elif len(self.calls) == 1:
                    index = next(iter(self.calls))
                elif not self.calls:
                    index = 0
            if not isinstance(index, int) or index < 0:
                raise TranslationError("Tool fragment has no unambiguous index or ID")
            if call_id and call_id in self.ids and self.ids[call_id] != index:
                raise TranslationError("Tool fragment ID/index mismatch")
            call = self.calls.setdefault(
                index,
                {
                    "index": index,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if call_id:
                if call.get("id") and call["id"] != call_id:
                    raise TranslationError("Tool fragment ID changed at the same index")
                call["id"] = call_id
                self.ids[call_id] = index
            function = delta.get("function") or {}
            if function.get("name"):
                call["function"]["name"] += function["name"]
            args = function.get("arguments")
            if args is not None:
                if isinstance(args, dict) and not call["function"]["arguments"]:
                    args = json.dumps(args, ensure_ascii=False)
                if not isinstance(args, str):
                    raise TranslationError("Tool argument fragment must be a string")
                call["function"]["arguments"] += args


class ChatTranslator(
    BaseTranslator[
        Union[OllamaGenerateRequest, OllamaChatRequest],
        OpenAIChatRequest,
        Union[OpenAIChatResponse, OpenAIStreamResponse],
        Union[OllamaResponse, OllamaStreamResponse],
    ]
):
    """
    Translator for chat completion requests and responses.

    Handles conversion between Ollama's generate/chat formats and
    OpenAI's chat completion format. Phase 1 supports text-only.
    """

    def __init__(self, model_mappings: Optional[Dict[str, str]] = None):
        """Initialize the chat translator."""
        super().__init__(model_mappings)
        self.logger = get_logger(__name__)

    def get_iso_timestamp(self) -> str:
        """Use actual UTC for the Z suffix on Ollama timestamps."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def handle_translation_error(self, error: Exception, context: str) -> None:
        # Pydantic/JSON exceptions may include prompts and tool argument values.
        self.logger.error(
            "Chat translation failed",
            extra={"extra_data": {"context": context, "error_type": type(error).__name__}},
        )
        if isinstance(error, (TranslationError, ValidationError)):
            raise error
        raise TranslationError(
            f"Failed to translate in {context}",
            details={"error_type": type(error).__name__},
        ) from None

    def translate_request(
        self, ollama_request: Union[OllamaGenerateRequest, OllamaChatRequest]
    ) -> OpenAIChatRequest:
        """
        Translate Ollama request to OpenAI chat completion format.

        Args:
            ollama_request: Either a generate or chat request from Ollama

        Returns:
            OpenAI chat completion request

        Raises:
            TranslationError: If translation fails
            ValidationError: If request contains unsupported features
        """
        try:
            # Validate request
            self._validate_ollama_request(ollama_request)

            # Convert to messages format
            messages = self._convert_to_messages(ollama_request)

            # Map model name
            model = self.map_model_name(ollama_request.model)

            # Extract options
            options = {}
            if ollama_request.options:
                options = self.extract_options(ollama_request.options)

            # Handle tools (Phase 2 feature)
            tools = None
            if isinstance(ollama_request, OllamaChatRequest) and ollama_request.tools:
                tools = self._translate_tools(ollama_request.tools)

            if ollama_request.format == "json":
                options["response_format"] = {"type": "json_object"}
            elif isinstance(ollama_request.format, dict):
                options["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": ollama_request.format},
                }
            if isinstance(ollama_request, OllamaChatRequest) and ollama_request.tool_choice is not None:
                options["tool_choice"] = ollama_request.tool_choice
            if ollama_request.stream:
                options["stream_options"] = {"include_usage": True}

            # Build OpenAI request
            openai_request = OpenAIChatRequest(
                model=model,
                messages=messages,
                stream=ollama_request.stream or False,
                tools=tools,
                **options,
            )

            self.logger.debug(
                "Translated Ollama request to OpenAI format",
                extra={
                    "extra_data": {
                        "model": model,
                        "message_count": len(messages),
                        "stream": openai_request.stream,
                    }
                },
            )

            return openai_request

        except (TranslationError, ValidationError):
            raise
        except Exception as e:
            self.handle_translation_error(e, "translate_request")
            raise  # Re-raise the error after handling

    def translate_response(
        self,
        openai_response: Union[OpenAIChatResponse, OpenAIStreamResponse],
        original_request: Union[OllamaGenerateRequest, OllamaChatRequest],
    ) -> Union[OllamaResponse, OllamaStreamResponse]:
        """
        Translate OpenAI response back to Ollama format.

        Args:
            openai_response: OpenAI chat completion response
            original_request: The original Ollama request for context

        Returns:
            Ollama format response

        Raises:
            TranslationError: If translation fails
        """
        try:
            # Handle streaming response
            if isinstance(openai_response, OpenAIStreamResponse):
                return self._translate_streaming_response(
                    openai_response, original_request
                )

            # Handle non-streaming response
            return self._translate_non_streaming_response(
                openai_response, original_request
            )

        except TranslationError:
            raise
        except Exception as e:
            self.handle_translation_error(e, "translate_response")
            raise  # Re-raise the error after handling

    def translate_streaming_response(
        self,
        openai_chunk: Union[Dict[str, Any], str],
        original_request: Union[OllamaGenerateRequest, OllamaChatRequest],
        is_first_chunk: bool = False,
        is_last_chunk: bool = False,
        state: Optional[ChatStreamState] = None,
    ) -> Optional[Dict[str, Any]]:
        """Translate one SSE event, buffering tool fragments until completion.

        The router supplies a per-request state and emits the terminal object
        after usage/[DONE]. A standalone complete chunk can still be translated.
        """
        standalone = state is None
        state = state if state is not None else ChatStreamState()
        if state.finished:
            return None
        if isinstance(openai_chunk, str):
            if not openai_chunk.strip():
                return None
            if openai_chunk.strip() == "[DONE]":
                if not state.saw_choice and not standalone:
                    raise TranslationError("Upstream returned no chat choices")
                calls = self._completed_stream_calls(state)
                state.finished = True
                result = self._stream_object(original_request, done=True)
                if calls:
                    self._attach_stream_calls(result, original_request, calls)
                result["done_reason"] = state.finish_reason or "stop"
                for source, target in (("prompt_tokens", "prompt_eval_count"), ("completion_tokens", "eval_count")):
                    if state.usage.get(source) is not None:
                        result[target] = state.usage[source]
                return result
            try:
                openai_chunk = json.loads(openai_chunk)
            except json.JSONDecodeError:
                raise TranslationError("Invalid JSON in upstream SSE event") from None
        if not isinstance(openai_chunk, dict):
            raise TranslationError("Upstream SSE event must be an object")
        if "error" in openai_chunk:
            raise TranslationError("Upstream reported an error in the SSE stream")
        state.usage.update(openai_chunk.get("usage") or {})
        choices = openai_chunk.get("choices")
        if choices == []:
            return None  # OpenAI's usage-only event.
        if not isinstance(choices, list) or not choices:
            raise TranslationError("Upstream SSE event has no choices")
        choice = choices[0]
        delta = choice.get("delta") or {}
        state.saw_choice = True
        if state.finish_reason and (delta.get("content") or delta.get("tool_calls")):
            raise TranslationError("Received generation data after finish_reason")
        if delta.get("tool_calls"):
            state.accumulate(delta["tool_calls"])
        elif delta.get("function_call"):
            state.accumulate([{"index": 0, "id": "call_legacy", "function": delta["function_call"]}])
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            state.finish_reason = finish_reason
        content = delta.get("content") or ""
        thinking = delta.get("reasoning_content") or ""
        if not isinstance(content, str) or not isinstance(thinking, str):
            raise TranslationError("Upstream content delta must be text or null")
        if content:
            state.text_chunks += 1
        calls = self._completed_stream_calls(state) if finish_reason else []
        result = self._stream_object(original_request, content=content, thinking=thinking)
        if calls:
            self._attach_stream_calls(result, original_request, calls)
        if standalone and finish_reason:
            result["done"] = True
            result["done_reason"] = finish_reason
        # Skip incomplete tool-only events; never expose JSON string fragments.
        if not content and not thinking and not calls and not delta.get("role"):
            return result if standalone and finish_reason else None
        return result

    def _completed_stream_calls(self, state: ChatStreamState) -> List[Dict[str, Any]]:
        if not state.calls or state.emitted_calls:
            return []
        calls = self._translate_tool_calls([state.calls[i] for i in sorted(state.calls)])
        state.emitted_calls = True
        return calls

    def _stream_object(self, request, content="", thinking="", done=False) -> Dict[str, Any]:
        result = {"model": request.model, "created_at": self.get_iso_timestamp(), "done": done}
        if isinstance(request, OllamaChatRequest):
            result["message"] = {"role": "assistant", "content": content}
            if thinking:
                result["message"]["thinking"] = thinking
        else:
            result["response"] = content
            if thinking:
                result["thinking"] = thinking
        return result

    @staticmethod
    def _attach_stream_calls(result, request, calls):
        if isinstance(request, OllamaChatRequest):
            result["message"]["tool_calls"] = calls
        else:
            result["tool_calls"] = calls

    def _validate_ollama_request(
        self, request: Union[OllamaGenerateRequest, OllamaChatRequest]
    ) -> None:
        """
        Validate Ollama request for Phase 2 support.

        Args:
            request: The Ollama request to validate

        Raises:
            ValidationError: If request contains invalid data
        """
        # Validate model name
        self.validate_model_name(request.model)

    def _translate_tools(self, ollama_tools: List[Dict[str, Any]]) -> List[OpenAITool]:
        """Preserve every definition and schema; reject invalid fields explicitly."""
        result = []
        for index, tool in enumerate(ollama_tools):
            path = f"tools[{index}].function"
            if not isinstance(tool, dict) or tool.get("type", "function") != "function":
                raise ValidationError("Expected a function tool", field=path)
            function = tool.get("function", tool)
            if not isinstance(function, dict):
                raise ValidationError("Expected a function definition", field=path)
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValidationError("Tool name is required", field=path + ".name")
            self._validate_tool_schema(function.get("parameters", {}), path + ".parameters", name)
            # Pydantic errors can include input values, so report only locations.
            try:
                result.append(OpenAITool(type="function", function=OpenAIFunction(**function)))
            except Exception:
                raise ValidationError(f"Invalid definition for tool {name}", field=path) from None
        return result

    def _validate_tool_schema(self, schema, path: str, name: str) -> None:
        def invalid(field):
            raise ValidationError(f"Invalid schema for tool {name}", field=field)

        if isinstance(schema, bool):
            return
        if not isinstance(schema, dict):
            invalid(path)
        if "required" in schema and (not isinstance(schema["required"], list) or not all(isinstance(x, str) for x in schema["required"])):
            invalid(path + ".required")
        if "enum" in schema and not isinstance(schema["enum"], list):
            invalid(path + ".enum")
        if "type" in schema:
            types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not types or not all(isinstance(value, str) and value in {"null", "boolean", "object", "array", "number", "string", "integer"} for value in types):
                invalid(path + ".type")
        for keyword in ("properties", "$defs", "definitions", "patternProperties"):
            if keyword in schema:
                if not isinstance(schema[keyword], dict):
                    invalid(path + "." + keyword)
                for key, value in schema[keyword].items():
                    self._validate_tool_schema(value, f"{path}.{keyword}.{key}", name)
        for keyword in ("items", "additionalProperties", "not", "if", "then", "else", "contains"):
            if keyword in schema:
                value = schema[keyword]
                if keyword == "items" and isinstance(value, list):
                    for i, item in enumerate(value):
                        self._validate_tool_schema(item, f"{path}.{keyword}[{i}]", name)
                else:
                    self._validate_tool_schema(value, f"{path}.{keyword}", name)
        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            if keyword in schema:
                if not isinstance(schema[keyword], list):
                    invalid(path + "." + keyword)
                for i, value in enumerate(schema[keyword]):
                    self._validate_tool_schema(value, f"{path}.{keyword}[{i}]", name)

    def _translate_tool_calls(self, openai_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for index, call in enumerate(openai_tool_calls):
            function = call.get("function") or {}
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise TranslationError(f"Tool call {index} has no function name")
            args = function.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    raise TranslationError(f"Tool call {index} has invalid or incomplete JSON arguments") from None
            if not isinstance(args, dict):
                raise TranslationError(f"Tool call {index} arguments must be an object")
            translated = {"function": {"name": name, "arguments": args}}
            if call.get("id"):
                translated["id"] = call["id"]
            if call.get("type"):
                translated["type"] = call["type"]
            if "index" in call:
                translated["function"]["index"] = call["index"]
            result.append(translated)
        return result

    def _translate_function_call(self, openai_function_call: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._translate_tool_calls([{"id": "call_legacy", "type": "function", "function": openai_function_call}])

    def _convert_message_content(
        self, message: OllamaChatMessage
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Convert Ollama message content to OpenAI multimodal format.

        Args:
            message: Ollama chat message

        Returns:
            Either string content or list of content objects for multimodal

        Raises:
            TranslationError: If content conversion fails
        """
        try:
            # Check if message has images (multimodal)
            if hasattr(message, "images") and message.images:
                # Create multimodal content array
                content_parts: List[Dict[str, Any]] = []

                # Add text content if present
                if message.content:
                    content_parts.append({"type": "text", "text": message.content})

                # Add image content
                for image_data in message.images:
                    # Validate base64 image data
                    if not isinstance(image_data, str):
                        self.logger.warning(
                            f"Invalid image data type: {type(image_data)}"
                        )
                        continue

                    # Create image content object
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": self._format_image_url(image_data)},
                    }
                    content_parts.append(image_content)

                self.logger.debug(
                    f"Converted multimodal message with {len(content_parts)} content parts",
                    extra={
                        "extra_data": {
                            "text_parts": 1 if message.content else 0,
                            "image_parts": len(message.images),
                        }
                    },
                )

                return content_parts
            else:
                # Simple text content
                return message.content or ""

        except Exception as e:
            self.handle_translation_error(e, "convert_message_content")
            raise

    def _format_image_url(self, image_data: str) -> str:
        """
        Format image data as data URL for OpenAI API.

        Args:
            image_data: Base64 encoded image data

        Returns:
            Formatted data URL
        """
        # If already a data URL, return as-is
        if image_data.startswith("data:"):
            return image_data

        # Assume JPEG by default (common for Ollama)
        # TODO: Add image type detection based on header
        return f"data:image/jpeg;base64,{image_data}"

    def _convert_to_messages(
        self, request: Union[OllamaGenerateRequest, OllamaChatRequest]
    ) -> List[OpenAIMessage]:
        """
        Convert Ollama request to OpenAI message format.

        Args:
            request: Ollama request (generate or chat)

        Returns:
            List of OpenAI format messages
        """
        messages = []

        if isinstance(request, OllamaGenerateRequest):
            # For generate requests, create a single user message
            if request.system:
                messages.append(OpenAIMessage(role="system", content=request.system))  # type: ignore[call-arg]

            # Add the prompt as a user message
            messages.append(OpenAIMessage(role="user", content=request.prompt))  # type: ignore[call-arg]

        else:
            # Only system messages move. All other messages retain their order.
            systems = [msg for msg in request.messages if msg.role == "system"]
            if systems:
                messages.append(OpenAIMessage(role="system", content="\n\n".join(msg.content or "" for msg in systems)))
            pending = []
            used_ids = set()
            for message_index, msg in enumerate(request.messages):
                if msg.role == "system":
                    continue
                content = self._convert_message_content(msg)
                calls = None
                tool_call_id = None
                if msg.tool_calls:
                    calls = []
                    for call_index, call in enumerate(msg.tool_calls):
                        # Ollama uses argument objects and may omit IDs entirely.
                        validated = self._translate_tool_calls([call])[0]
                        call_id = call.get("id") or f"call_{message_index}_{call_index}"
                        if call_id in used_ids:
                            raise ValidationError("Duplicate tool call ID in history")
                        used_ids.add(call_id)
                        function = validated["function"]
                        converted = {"id": call_id, "type": "function", "function": {"name": function["name"], "arguments": json.dumps(function["arguments"], ensure_ascii=False)}}
                        calls.append(converted)
                        pending.append(converted)
                if msg.role == "tool":
                    candidates = pending
                    if msg.tool_call_id:
                        candidates = [call for call in pending if call["id"] == msg.tool_call_id]
                    elif msg.tool_name:
                        candidates = [call for call in pending if call["function"]["name"] == msg.tool_name]
                    if not candidates:
                        raise ValidationError("Tool result has no matching assistant tool call", field=f"messages[{message_index}]")
                    matched = candidates[0]
                    tool_call_id = matched["id"]
                    pending.remove(matched)
                messages.append(OpenAIMessage(role=msg.role, content=content, tool_calls=calls, tool_call_id=tool_call_id, reasoning_content=msg.thinking))
        return messages

    def _translate_streaming_response(self, openai_response: OpenAIStreamResponse, original_request) -> OllamaStreamResponse:
        result = self.translate_streaming_response(openai_response.model_dump(exclude_none=True), original_request)
        model = OllamaChatResponse if isinstance(original_request, OllamaChatRequest) else OllamaGenerateResponse
        return model(**(result or self._stream_object(original_request)))

    def _translate_non_streaming_response(
        self,
        openai_response: OpenAIChatResponse,
        original_request: Union[OllamaGenerateRequest, OllamaChatRequest],
    ) -> OllamaResponse:
        """
        Translate non-streaming OpenAI response to Ollama format.

        Args:
            openai_response: OpenAI chat completion response
            original_request: Original Ollama request

        Returns:
            Ollama response
        """
        # Extract content from the first choice
        content = ""
        finish_reason = "stop"
        tool_calls = None
        thinking = None

        if not openai_response.choices:
            raise TranslationError("Upstream returned no chat choices")

        if openai_response.choices:
            choice = openai_response.choices[0]
            if choice.message:
                content = choice.message.content or ""  # type: ignore[assignment]
                reasoning = getattr(choice.message, "reasoning_content", None)
                thinking = reasoning if isinstance(reasoning, str) else None

                # Handle tool calls (Phase 2 feature)
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    tool_calls = self._translate_tool_calls(choice.message.tool_calls)
                elif (
                    hasattr(choice.message, "function_call")
                    and choice.message.function_call
                ):
                    # Handle legacy function_call format
                    tool_calls = self._translate_function_call(
                        choice.message.function_call
                    )

            finish_reason = choice.finish_reason or "stop"

        # Build response - use appropriate response type based on request type
        if isinstance(original_request, OllamaChatRequest):
            # For chat requests, always use OllamaChatResponse with message
            message = OllamaChatMessage(
                role="assistant", content=content, tool_calls=tool_calls, images=None, thinking=thinking
            )
            chat_response = OllamaChatResponse(  # type: ignore[call-arg]
                model=self.reverse_map_model_name(openai_response.model),
                created_at=self.get_iso_timestamp(),
                message=message,
                done=True,
                done_reason=finish_reason,
            )
            response: Union[OllamaChatResponse, OllamaGenerateResponse] = chat_response
        else:
            # For generate requests, use OllamaGenerateResponse
            gen_response = OllamaGenerateResponse(  # type: ignore[call-arg]
                model=self.reverse_map_model_name(openai_response.model),
                created_at=self.get_iso_timestamp(),
                response=content,
                done=True,
                done_reason=finish_reason,
            )
            response = gen_response

        # Add token usage if available
        if openai_response.usage:
            response.prompt_eval_count = openai_response.usage.prompt_tokens
            response.eval_count = openai_response.usage.completion_tokens

            # Upstream does not supply nanosecond timings; do not invent them.

        return response

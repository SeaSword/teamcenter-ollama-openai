"""
Chat and generation endpoints for the Ollama API.

This module handles both Ollama-style (/api) and OpenAI-style (/v1) endpoints
for chat completions and text generation.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Union

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError as PydanticValidationError

from src.config import get_settings
from src.models import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OpenAIChatRequest,
    OpenAIChatResponse,
)
from src.translators.chat import ChatStreamState, ChatTranslator
from src.utils.exceptions import (
    TranslationError,
    UpstreamError,
    ValidationError,
)
from src.utils.http_client import RetryClient, retry_client_context
from src.utils.logging import get_logger, request_id_context
from src.utils.request_body import get_body_json
from src.utils.sse import iter_sse_data

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

# Initialize translator
translator = ChatTranslator()


def _request_id(request: Request) -> str:
    # Logging middleware owns the ID returned to clients. Reuse its context
    # when nested middleware has independently populated request.state.
    supplied = request.headers.get("x-request-id")
    if isinstance(supplied, str) and supplied:
        return supplied
    return request_id_context.get() or getattr(request.state, "request_id", "unknown")


def _upstream_failure(exc, model, request_id="unknown", path="/api/chat"):
    """Log diagnostic metadata only, never bodies, headers or exception text."""
    upstream_status = None
    if isinstance(exc, UpstreamError):
        error = exc
        upstream_status = exc.details.get("upstream_status", exc.status_code)
    elif isinstance(exc, httpx.HTTPStatusError):
        upstream_status = exc.response.status_code
        error = UpstreamError(
            f"Upstream HTTP {upstream_status}: {httpx.codes.get_reason_phrase(upstream_status)}",
            status_code=upstream_status,
        )
    elif isinstance(exc, httpx.TimeoutException):
        error = UpstreamError("Upstream request timed out", status_code=504)
    elif isinstance(exc, (TranslationError, PydanticValidationError, ValueError)):
        upstream_status = 200
        message = (
            exc.message if isinstance(exc, TranslationError)
            else "Invalid upstream chat response"
        )
        error = UpstreamError(message, status_code=502)
    else:
        error = UpstreamError("Upstream connection or response failed", status_code=502)
    logger.error(
        "Backend request failed",
        extra={"extra_data": {
            "request_id": request_id,
            "model": model,
            "path": path,
            "upstream_status": upstream_status,
            "status_code": error.status_code,
            "exception_type": type(exc).__name__,
        }},
    )
    return error


def _ollama_error(error, request_id):
    return JSONResponse(
        status_code=error.status_code,
        content={"error": error.message},
        headers={"X-Request-ID": request_id},
    )


def _translate_backend_response(response, original_request, model, request_id, path):
    try:
        data = response.json()
        # Some backends use null for an empty assistant completion.
        for choice in data.get("choices", []):
            message = choice.get("message") or {}
            if (
                message.get("content") is None
                and not message.get("tool_calls")
                and not message.get("function_call")
            ):
                message["content"] = ""
        translated = translator.translate_response(
            OpenAIChatResponse(**data), original_request
        )
        logger.info(
            "Backend chat completed",
            extra={"extra_data": {
                "request_id": request_id,
                "model": model,
                "path": path,
                "upstream_status": 200,
                "finish_reason": translated.done_reason,
                "tool_call_count": (
                    len(translated.message.tool_calls or [])
                    if isinstance(original_request, OllamaChatRequest) else 0
                ),
            }},
        )
        return translated
    except Exception as exc:
        raise _upstream_failure(exc, model, request_id, path) from None


async def make_openai_request(
    client: RetryClient,
    openai_request: OpenAIChatRequest,
    stream: bool = False,
    request_id="unknown",
    path="/api/chat",
) -> httpx.Response:
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{settings.OPENAI_API_BASE_URL}/chat/completions"
    try:
        response = await client.request_with_retry(
            "POST", url, json=openai_request.model_dump(exclude_none=True), headers=headers
        )
        if response.status_code != 200:
            raise UpstreamError(
                f"Upstream HTTP {response.status_code}: {httpx.codes.get_reason_phrase(response.status_code)}",
                status_code=response.status_code,
            )
        return response
    except Exception as exc:
        raise _upstream_failure(exc, openai_request.model, request_id, path) from None


async def stream_response(
    client: RetryClient,
    openai_request: OpenAIChatRequest,
    original_request: Union[OllamaGenerateRequest, OllamaChatRequest],
    request_id="unknown",
    path="/api/chat",
) -> AsyncGenerator[str, None]:
    """Translate complete SSE events to Ollama NDJSON with per-response state."""
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{settings.OPENAI_API_BASE_URL}/chat/completions"
    state = ChatStreamState()
    try:
        # aclosing also releases the upstream connection on [DONE] or disconnect.
        async with aclosing(client.stream_with_retry(
            "POST", url, json=openai_request.model_dump(exclude_none=True), headers=headers
        )) as chunks:
            async for data in iter_sse_data(chunks):
                result = translator.translate_streaming_response(
                    data, original_request, state=state
                )
                if result:
                    yield json.dumps(result, ensure_ascii=False) + "\n"
                if state.finished:
                    break
        if not state.finished:
            if not state.finish_reason:
                raise TranslationError("Upstream stream ended before completion")
            result = translator.translate_streaming_response(
                "[DONE]", original_request, state=state
            )
            if result:
                yield json.dumps(result, ensure_ascii=False) + "\n"
        logger.info(
            "Backend chat stream completed",
            extra={"extra_data": {
                "request_id": request_id,
                "model": openai_request.model,
                "path": path,
                "upstream_status": 200,
                "finish_reason": state.finish_reason,
                "tool_count": len(openai_request.tools or []),
                "tool_call_count": len(state.calls),
                "text_chunk_count": state.text_chunks,
            }},
        )
    except Exception as exc:
        raise _upstream_failure(exc, openai_request.model, request_id, path) from None


async def _prepared_stream(iterator, request_id, *, openai=False):
    """Read the first useful event before StreamingResponse sends HTTP 200."""
    try:
        first = await anext(iterator)
    except StopAsyncIteration:
        await iterator.aclose()
        raise UpstreamError("Upstream returned an empty stream", status_code=502) from None
    except BaseException:
        await iterator.aclose()
        raise

    async def body():
        try:
            yield first
            async for chunk in iterator:
                yield chunk
        except UpstreamError as exc:
            # Ollama SDK checks `error` before validating ChatResponse.
            if openai:
                yield "data: " + json.dumps({"error": {"message": exc.message, "type": "upstream_error"}}) + "\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield json.dumps({"error": exc.message}) + "\n"
        finally:
            await iterator.aclose()

    return StreamingResponse(
        body(),
        media_type="text/event-stream" if openai else "application/x-ndjson",
        headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"},
    )


# OpenAI-style endpoints (defined first to take precedence)
@router.post("/chat/completions")  # OpenAI-style endpoint
async def openai_chat_completions(
    fastapi_request: Request,
):
    """
    Handle OpenAI-style chat completion requests.

    This endpoint accepts OpenAI-style chat requests and forwards them
    directly to the OpenAI backend without translation.
    """
    request_id = _request_id(fastapi_request)

    logger.debug(
        "OpenAI endpoint handler started",
        extra={
            "extra_data": {
                "request_id": request_id,
                "path": fastapi_request.url.path,
            }
        },
    )

    # Get request body using the utility function
    try:
        request = await get_body_json(fastapi_request)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Invalid request body", extra={"extra_data": {"request_id": request_id, "exception_type": type(e).__name__}})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request body",
        )

    logger.info(
        "OpenAI chat completion request received",
        extra={
            "extra_data": {
                "request_id": request_id,
                "model": request.get("model", "unknown"),
                "message_count": len(request.get("messages", [])),
                "stream": request.get("stream", False),
            }
        },
    )

    try:
        async with retry_client_context() as client:
            # Forward request directly to OpenAI without Pydantic validation
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            url = f"{settings.OPENAI_API_BASE_URL}/chat/completions"

            if request.get("stream", False):
                # Handle streaming
                return await _prepared_stream(
                    openai_stream_response_dict(client, request, request_id, fastapi_request.url.path),
                    request_id, openai=True,
                )
            else:
                # Make non-streaming request
                response = await client.request_with_retry(
                    "POST",
                    url,
                    json=request,
                    headers=headers,
                )

                if response.status_code != 200:
                    raise _upstream_failure(
                        UpstreamError(f"Upstream HTTP {response.status_code}: {httpx.codes.get_reason_phrase(response.status_code)}", status_code=response.status_code),
                        request.get("model"), request_id, fastapi_request.url.path,
                    )

                # Return response directly
                return JSONResponse(
                    content=response.json(),
                    headers={"X-Request-ID": request_id},
                )

    except UpstreamError as e:
        # Re-raise with appropriate status code
        raise HTTPException(
            status_code=e.status_code,
            detail=f"Upstream error: {str(e)}",
        )
    except httpx.TransportError as exc:
        error = _upstream_failure(exc, request.get("model"), request_id, fastapi_request.url.path)
        raise HTTPException(status_code=error.status_code, detail=error.message) from None
    except Exception as e:
        logger.error("Unexpected error in OpenAI chat completions endpoint", extra={"extra_data": {"request_id": request_id, "exception_type": type(e).__name__}})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/completions")  # OpenAI-style text completion endpoint
async def openai_completions(
    fastapi_request: Request,
):
    """
    Handle OpenAI-style text completion requests.

    This endpoint accepts OpenAI-style requests for text completion
    and forwards them to the chat completions endpoint.
    """
    request_id = _request_id(fastapi_request)

    # Get request body using the utility function
    try:
        request = await get_body_json(fastapi_request)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Invalid request body", extra={"extra_data": {"request_id": request_id, "exception_type": type(e).__name__}})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request body",
        )

    logger.info(
        "OpenAI text completion request received",
        extra={
            "extra_data": {
                "request_id": request_id,
                "model": request.get("model", "unknown"),
                "message_count": len(request.get("messages", [])),
                "stream": request.get("stream", False),
            }
        },
    )

    # Forward to chat completions since OpenAI completions API is being phased out
    return await openai_chat_completions(fastapi_request)


# Ollama-style endpoints
@router.post("/generate")
async def generate(
    fastapi_request: Request,
):
    """
    Handle Ollama generate requests (text-only in Phase 1).

    This endpoint accepts Ollama-style generation requests and translates them
    to OpenAI chat completion format for processing.
    """
    request_id = _request_id(fastapi_request)

    # Get request body using the utility function
    try:
        request_dict = await get_body_json(fastapi_request)
        request = OllamaGenerateRequest(**request_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Invalid Ollama request", extra={"extra_data": {"request_id": request_id, "exception_type": type(e).__name__}})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Ollama request body",
        )

    logger.info(
        "Generate request received",
        extra={
            "extra_data": {
                "request_id": request_id,
                "model": request.model,
                "prompt_length": len(request.prompt),
                "stream": request.stream,
            }
        },
    )

    try:
        # Translate to OpenAI format
        openai_request = translator.translate_request(request)

        async with retry_client_context() as client:
            if request.stream:
                # Return streaming response
                return await _prepared_stream(
                    stream_response(client, openai_request, request, request_id, fastapi_request.url.path),
                    request_id,
                )
            else:
                # Make non-streaming request
                response = await make_openai_request(
                    client, openai_request, stream=False, request_id=request_id, path=fastapi_request.url.path
                )

                ollama_response = _translate_backend_response(
                    response, request, openai_request.model, request_id, fastapi_request.url.path
                )

                return JSONResponse(
                    content=ollama_response.model_dump(exclude_none=True),
                    headers={"X-Request-ID": request_id},
                )

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except TranslationError as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Translation error: {str(e)}",
        )
    except UpstreamError as e:
        # Re-raise with appropriate status code
        return _ollama_error(e, request_id)
    except Exception as e:
        logger.error("Unexpected error in generate endpoint", extra={"extra_data": {"request_id": request_id, "exception_type": type(e).__name__}})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/chat")
async def ollama_chat(
    fastapi_request: Request,
):
    """
    Handle Ollama chat requests (text-only in Phase 1).

    This endpoint accepts Ollama-style chat requests with message history
    and processes them through the OpenAI backend.
    """
    request_id = _request_id(fastapi_request)

    # Get request body using the utility function
    try:
        request_dict = await get_body_json(fastapi_request)
        request = OllamaChatRequest(**request_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Invalid Ollama request", extra={"extra_data": {"request_id": request_id, "exception_type": type(e).__name__}})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Ollama request body",
        )

    logger.info(
        "Chat request received",
        extra={
            "extra_data": {
                "request_id": request_id,
                "model": request.model,
                "message_count": len(request.messages),
                "stream": request.stream,
            }
        },
    )

    try:
        # Translate to OpenAI format
        openai_request = translator.translate_request(request)

        async with retry_client_context() as client:
            if request.stream:
                # Return streaming response
                return await _prepared_stream(
                    stream_response(client, openai_request, request, request_id, fastapi_request.url.path),
                    request_id,
                )
            else:
                # Make non-streaming request
                response = await make_openai_request(
                    client, openai_request, stream=False, request_id=request_id, path=fastapi_request.url.path
                )

                ollama_response = _translate_backend_response(
                    response, request, openai_request.model, request_id, fastapi_request.url.path
                )

                return JSONResponse(
                    content=ollama_response.model_dump(exclude_none=True),
                    headers={"X-Request-ID": request_id},
                )

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except TranslationError as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Translation error: {str(e)}",
        )
    except UpstreamError as e:
        # Re-raise with appropriate status code
        return _ollama_error(e, request_id)
    except Exception as e:
        logger.error("Unexpected error in chat endpoint", extra={"extra_data": {"request_id": request_id, "exception_type": type(e).__name__}})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


async def openai_stream_response(client: RetryClient, openai_request: OpenAIChatRequest) -> AsyncGenerator[bytes, None]:
    async for chunk in openai_stream_response_dict(client, openai_request.model_dump(exclude_none=True)):
        yield chunk


async def openai_stream_response_dict(client: RetryClient, request_dict: dict, request_id="unknown", path="/v1/chat/completions") -> AsyncGenerator[bytes, None]:
    """Pass OpenAI SSE bytes through without translating Copilot's payload."""
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"}
    url = f"{settings.OPENAI_API_BASE_URL}/chat/completions"
    try:
        async with aclosing(client.stream_with_retry("POST", url, json=request_dict, headers=headers)) as chunks:
            async for chunk in chunks:
                yield chunk
    except Exception as exc:
        raise _upstream_failure(exc, request_dict.get("model"), request_id, path) from None

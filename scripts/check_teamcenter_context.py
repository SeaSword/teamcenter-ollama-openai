"""Count the normalized capture with vLLM's tokenizer; never print its content."""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.models import OllamaChatRequest
from src.translators.chat import ChatTranslator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.output_tokens < 0:
        parser.error("--output-tokens must be non-negative")
    try:
        text = args.capture.read_text(encoding="utf-8-sig")
        body = next(json.loads(line) for line in text.splitlines() if line.startswith('{"model"'))
        request = OllamaChatRequest(model=body["model"], messages=body["messages"], tools=body.get("tools"), stream=False)
        translated = ChatTranslator().translate_request(request).model_dump(exclude_none=True)
        settings = get_settings()
        base = urlsplit(str(settings.OPENAI_API_BASE_URL))
        # /tokenize is outside /v1. Preserve a possible reverse-proxy prefix.
        prefix = base.path.rstrip("/").removesuffix("/v1")
        url = urlunsplit((base.scheme, base.netloc, prefix + "/tokenize", "", ""))
        payload = {"model": translated["model"], "messages": translated["messages"], "add_generation_prompt": True}
        if translated.get("tools"):
            payload["tools"] = translated["tools"]
        with httpx.Client(timeout=20) as client:
            response = client.post(url, json=payload, headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"})
        if response.status_code != 200:
            print(json.dumps({"tokenization": "unavailable", "upstream_status": response.status_code}))
            return 1
        result = response.json()
        count = result["count"]
        if not isinstance(count, int) or count < 0:
            raise ValueError("Invalid token count")
        print(json.dumps({"model": translated["model"], "tools": len(translated.get("tools", [])), "prompt_tokens": count, "output_token_budget": args.output_tokens, "minimum_context_with_budget": count + args.output_tokens, "backend_max_model_len": result.get("max_model_len"), "method": "vLLM /tokenize with normalized messages and tools"}))
        return 0
    except Exception as exc:
        print(json.dumps({"tokenization": "unavailable", "exception_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

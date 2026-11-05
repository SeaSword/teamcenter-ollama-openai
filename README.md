# Teamcenter Ollama–OpenAI Adapter

English | [简体中文](README.zh-CN.md)

An independent derivative of [eyalrot/ollama_openai](https://github.com/eyalrot/ollama_openai), adding protocol compatibility for Teamcenter Copilot and BOM Agents using an OpenAI-compatible backend such as vLLM.

```text
Teamcenter BOM Agent  → /api/chat             ┐
Teamcenter Copilot    → /v1/chat/completions  ├→ FastAPI adapter → vLLM / OpenAI-compatible backend
Other Ollama clients → /api/generate         ┘
```

This repository contains the adapter, tests, and deployment helpers. Teamcenter, model weights, and the backend inference service are deployed separately.

## Features

- Translate Ollama Chat/Generate requests to OpenAI Chat Completions.
- Normalize system-message placement only for Ollama requests, preserving other conversation order.
- Return Ollama Chat JSON/NDJSON with `message`; retain Generate's `response` format.
- Accumulate fragmented tool names and JSON arguments, support multiple calls, and associate results across conversation turns.
- Keep reasoning in `message.thinking`, separate from normal content; handle null content and tool-only responses.
- Detect upstream failures before starting an HTTP 200 stream where possible, report mid-stream errors, and avoid replaying partially delivered requests.
- Forward Copilot `/v1/chat/completions` payloads and successful responses, retaining JSON Schema, citations, and extension metadata.
- Use the backend API key from server-side environment settings, independently of a client's `Bearer ollama` header.

The upstream model-listing, embedding, monitoring, Docker, and other endpoint implementations remain available. The Teamcenter-specific changes focus on chat and tool calling.

## Validation status

The offline suite contains **34 compatibility tests**, covering 38-tool requests, fragmented UTF-8/SSE, tool arguments, multiple tool rounds, HTTP errors, and mocked Copilot regression. All 34 passed on 2026-10-31 with private captures available. The public-source check runs 33 successfully and skips the private-capture comparison. Synthetic fixtures require no real API key or GPU.

Offline validation does not establish complete Teamcenter integration. Live BOM operations, Navigation tool selection, knowledge-base citations, and context capacity still require testing in your environment. The adapter does not fabricate actual agent tool results.

## Quick start on Windows

Python 3.12 is recommended:

```powershell
git clone https://github.com/SeaSword/teamcenter-ollama-openai.git
Set-Location teamcenter-ollama-openai
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit the local `.env` with your backend configuration. This file is ignored by Git:

```dotenv
OPENAI_API_BASE_URL=http://your-vllm-server:8000/v1
OPENAI_API_KEY=replace-with-your-backend-key
MAX_RETRIES=3
REQUEST_TIMEOUT=180
```

Start the adapter:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app `
  --host 0.0.0.0 `
  --port 11435
```

If you use Windows user environment variables instead of `.env`, an existing terminal may not have inherited them. Reload them before starting, without printing their values:

```powershell
foreach ($name in 'OPENAI_API_BASE_URL','OPENAI_API_KEY','MAX_RETRIES','REQUEST_TIMEOUT') {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if ($value) { [Environment]::SetEnvironmentVariable($name, $value, 'Process') }
}
```

Use `Ctrl+C` to stop a foreground server. Linux uses the same application module:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env before starting.
.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 11435
```

## Main endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/tags` | List backend models |
| `POST /api/chat` | Ollama chat, streaming and tool calling |
| `POST /api/generate` | Ollama text generation |
| `POST /v1/chat/completions` | OpenAI-compatible pass-through |

Example:

```powershell
Invoke-RestMethod http://localhost:11435/api/tags
$payload = @{ model='your-served-model-name'; messages=@(@{role='user';content='Hello'}); stream=$false } | ConvertTo-Json -Depth 10
Invoke-RestMethod http://localhost:11435/api/chat -Method Post `
  -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

## Tests

Run the offline regression using the runtime dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_teamcenter_compatibility -v
```

By default, 33 tests run and one optional local capture check is explicitly skipped. Setting `TEAMCENTER_CAPTURE_DIR` to a directory containing the two expected capture files enables all 34. Captures are not distributed and should never be committed.

With the existing development dependencies installed, the upstream unit suite can also be invoked:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit
```

The recorded 34-test result applies to the compatibility suite. The entire upstream pytest suite was not run in the initial local environment because pytest-related dependencies were unavailable.

## Live checks

These commands call a real model; supply your backend's served model name:

```powershell
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Chat -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Stream -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Tools -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Tools -ForceDispatch -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Copilot -Model 'your-served-model-name'
```

`-ForceDispatch` applies only to that controlled test request. There is no global forced tool choice. Tools mode checks complete object-valued arguments, then submits an explicitly labeled `test_mock` no-op tool result to verify the next turn. It does not execute Teamcenter MCP operations.

Count a private capture using the actual backend tokenizer:

```powershell
.\.venv\Scripts\python.exe scripts/check_teamcenter_context.py `
  'C:\private-captures\vllmtoollama1.txt' --output-tokens 2048
```

The script sends normalized messages and every tool to `/tokenize`. HTTP body size is not a token count. Context capacity must cover input, reserved output, and subsequent tool history; check model and GPU/KV-cache capacity before increasing it.

## Repository layout

- `src/`: FastAPI routes, models, translators, and HTTP client.
- `tests/`: upstream tests and Teamcenter protocol regression.
- `scripts/`: Windows live checks, token counting, and upstream helpers.
- [Compatibility notes](docs/TEAMCENTER_COMPATIBILITY.md): behavior and limits in English and Chinese.
- [Original upstream README](docs/UPSTREAM_README.md): upstream documentation, badges, releases, and image references.
- `.github/upstream-workflows/`: inactive copies of upstream CI/release definitions.

The active workflow runs mocked compatibility regression without real backend credentials. Automatic PyPI, Docker Hub, and model deployment publishing is not configured.

## License and attribution

The upstream [MIT License](LICENSE) and copyright notice are retained. Original project: [eyalrot/ollama_openai](https://github.com/eyalrot/ollama_openai). This is an independent derivative, not an official Siemens product or a Siemens-certified integration.

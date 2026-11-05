# Teamcenter Ollama–OpenAI 适配器

[English](README.md) | 简体中文

基于 [eyalrot/ollama_openai](https://github.com/eyalrot/ollama_openai) 的独立衍生项目，为 Teamcenter Copilot 和 BOM Agent 增强 Ollama 与 OpenAI 兼容后端（例如 vLLM）之间的协议转换。

```text
Teamcenter BOM Agent  → /api/chat             ┐
Teamcenter Copilot    → /v1/chat/completions  ├→ FastAPI 适配器 → vLLM / OpenAI 兼容服务
其他 Ollama 客户端    → /api/generate         ┘
```

本项目提供适配器源码、测试和部署辅助脚本。Siemens Teamcenter、模型权重和 vLLM 需要单独部署。

## 主要功能

- 将 Ollama Chat/Generate 请求转换为 OpenAI Chat Completions。
- 只在 Ollama 请求转换时前置 System Message，合并多个 System，保留其余历史消息顺序。
- `/api/chat` 返回具有 `message` 的 Ollama Chat JSON/NDJSON，`/api/generate` 保留 `response`。
- 累积分段工具名称与 JSON 参数，支持多个工具、纯工具响应和多轮工具结果关联。
- 将 reasoning 内容独立映射到 `message.thinking`，兼容空 content。
- 在发送流式 HTTP 200 前尽量检查上游错误；断流后明确报错，避免重放已经开始的请求。
- Copilot 的 `/v1/chat/completions` 透传原请求和成功响应，保留 JSON Schema 与引用等扩展元数据。
- 使用服务器环境变量中的后端 Key；客户端传入的 `Bearer ollama` 不作为后端凭据。

保留上游的模型列表、Embedding、监控、Docker 和其他接口代码；本次 Teamcenter 兼容性增强主要覆盖 Chat 与 Tool Calling。

## 验证状态

离线协议回归共 **34 项**，覆盖 38 个工具、中文跨字节流、分段参数、多轮工具调用、错误状态和 Copilot Mock 回归。2026-10-31 在私有抓包可用时全部通过；公开源码检查运行 33 项并通过，私有抓包比较明确跳过。固定测试输入为合成数据，不需要真实 API Key 或 GPU。

离线通过不代表完整 Teamcenter 现场验收成功。真实 BOM 操作、Navigation 工具选择、知识库引用与上下文容量仍需在实际环境验证。没有在代理中伪造真实 Agent 工具结果。

## Windows 快速启动

推荐 Python 3.12。在项目目录运行：

```powershell
git clone https://github.com/SeaSword/teamcenter-ollama-openai.git
Set-Location teamcenter-ollama-openai
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑本地 `.env`，配置自己的后端地址和真实 Key。该文件已被 Git 忽略：

```dotenv
OPENAI_API_BASE_URL=http://your-vllm-server:8000/v1
OPENAI_API_KEY=replace-with-your-backend-key
MAX_RETRIES=3
REQUEST_TIMEOUT=180
```

启动适配器：

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app `
  --host 0.0.0.0 `
  --port 11435
```

若使用 Windows 用户环境变量而不是 `.env`，已经打开的终端不一定继承新值。可在启动前重新加载；不要输出变量值：

```powershell
foreach ($name in 'OPENAI_API_BASE_URL','OPENAI_API_KEY','MAX_RETRIES','REQUEST_TIMEOUT') {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if ($value) { [Environment]::SetEnvironmentVariable($name, $value, 'Process') }
}
```

前台服务使用 `Ctrl+C` 停止。Linux 可使用相同模块启动：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 后启动
.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 11435
```

## 常用接口

| 接口 | 用途 |
| --- | --- |
| `GET /api/tags` | 查询后端模型 |
| `POST /api/chat` | Ollama Chat，支持流式和工具调用 |
| `POST /api/generate` | Ollama Generate |
| `POST /v1/chat/completions` | OpenAI 兼容请求透传 |

基本检查：

```powershell
Invoke-RestMethod http://localhost:11435/api/tags
$payload = @{ model='your-served-model-name'; messages=@(@{role='user';content='你好'}); stream=$false } | ConvertTo-Json -Depth 10
Invoke-RestMethod http://localhost:11435/api/chat -Method Post `
  -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

## 自动测试

仅依赖项目运行时依赖的离线回归：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_teamcenter_compatibility -v
```

默认会明确跳过一个可选的本地抓包检查，其余 33 项运行；若提供 `TEAMCENTER_CAPTURE_DIR` 且目录中有约定名称的两份抓包，则运行完整 34 项。原始抓包不随仓库分发，也不应提交到 Git。

已有开发依赖时，可运行原有 pytest 测试集：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit
```

当前记录的 34 项结果来自兼容性回归，不代表原有 pytest 全套已通过。首次上传前的本地环境缺少 pytest 相关依赖。

## 现场检查

以下命令会调用真实模型，请替换模型服务名：

```powershell
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Chat -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Stream -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Tools -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Tools -ForceDispatch -Model 'your-served-model-name'
.\scripts\Test-TeamcenterCompatibility.ps1 -Mode Copilot -Model 'your-served-model-name'
```

`-ForceDispatch` 仅控制本次测试请求；默认不会全局强制工具选择。Tools 模式会验证完整工具参数对象，再提交明确标记为 `test_mock` 的无操作工具结果，检查后续轮次；它不会执行 Teamcenter MCP 操作。

统计本地抓包的实际 Token 数：

```powershell
.\.venv\Scripts\python.exe scripts/check_teamcenter_context.py `
  'C:\private-captures\vllmtoollama1.txt' --output-tokens 2048
```

此脚本调用后端 `/tokenize`，将规范化消息与全部工具计入模板。HTTP 字节数不能直接当作 Token 数。上下文应覆盖输入、预留输出和后续工具历史；扩容前检查模型支持与 GPU/KV Cache 余量。

## 目录与文档

- `src/`：FastAPI 路由、协议模型、转换器和 HTTP 客户端。
- `tests/`：原有测试与 Teamcenter 协议回归。
- `scripts/`：Windows 现场验证、上下文计数及原有辅助脚本。
- `docs/TEAMCENTER_COMPATIBILITY.md`：兼容性说明与已知边界（中英文）。
- `docs/UPSTREAM_README.md`：保留的上游原始说明；其中的徽章、发行版和镜像属于上游。
- `.github/upstream-workflows/`：上游 CI/发布流程的参考副本，不会作为本仓库工作流执行。

本仓库活动 CI 只运行不访问真实后端的兼容性回归。没有配置自动发布 PyPI、Docker Hub 或模型部署。

## 许可与来源

保留上游 [MIT License](LICENSE) 和版权声明。上游项目：[eyalrot/ollama_openai](https://github.com/eyalrot/ollama_openai)。本项目为独立衍生实现，非 Siemens 官方产品，也不代表 Siemens 对该集成作出认证。

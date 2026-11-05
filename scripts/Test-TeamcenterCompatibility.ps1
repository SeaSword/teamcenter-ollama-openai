[CmdletBinding()]
param(
    [ValidateSet('All', 'Tags', 'Chat', 'Stream', 'Tools', 'Copilot')]
    [string]$Mode = 'All',
    [string]$BaseUrl = 'http://localhost:11435',
    [string]$Model = 'qwen3.8-27b',
    [switch]$ForceDispatch
)

$ErrorActionPreference = 'Stop'
$BaseUrl = $BaseUrl.TrimEnd('/')

function Invoke-Adapter([string]$Path, [hashtable]$Payload) {
    $json = ConvertTo-Json -InputObject $Payload -Depth 100 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $response = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl$Path" -Method Post `
        -ContentType 'application/json; charset=utf-8' -Body $bytes `
        -Headers @{ Authorization = 'Bearer ollama'; 'X-Request-ID' = "manual-$([guid]::NewGuid())" } `
        -TimeoutSec 240
    # Decode NDJSON as UTF-8 explicitly, including on Windows PowerShell 5.1.
    $response.RawContentStream.Position = 0
    $reader = [System.IO.StreamReader]::new($response.RawContentStream, [System.Text.Encoding]::UTF8)
    try {
        return [pscustomobject]@{ Content = $reader.ReadToEnd(); StatusCode = $response.StatusCode }
    }
    finally { $reader.Dispose() }
}

function Convert-ChatStream($Response) {
    $lines = @($Response.Content -split '\r?\n' | Where-Object { $_.Trim() } | ForEach-Object { ConvertFrom-Json -InputObject $_ })
    if ($lines.Count -eq 0) { throw 'Empty chat stream' }
    foreach ($line in $lines) {
        if ($line.PSObject.Properties.Name -contains 'error') { throw "Adapter stream error: $($line.error)" }
        if (-not $line.message) { throw 'Chat chunk is missing message' }
        if ($line.PSObject.Properties.Name -contains 'response') { throw 'Generate response received on chat endpoint' }
    }
    if (-not $lines[-1].done) { throw 'Chat stream did not end with done=true' }
    if (@($lines | Where-Object { $_.done }).Count -ne 1) { throw 'Expected exactly one terminal object' }
    return $lines
}

if ($Mode -in @('All', 'Tags')) {
    $tags = Invoke-RestMethod -Uri "$BaseUrl/api/tags" -TimeoutSec 60
    if (-not $tags.models) { throw 'No models returned by /api/tags' }
    $tags | ConvertTo-Json -Depth 12
}

if ($Mode -in @('All', 'Chat')) {
    $result = Invoke-Adapter '/api/chat' @{
        model = $Model; stream = $false
        messages = @(@{ role = 'user'; content = 'Reply briefly in Chinese.' }, @{ role = 'system'; content = 'You are a helpful assistant.' })
        options = @{ temperature = 0; num_predict = 128 }
    }
    $chatResult = ConvertFrom-Json -InputObject $result.Content
    if (-not $chatResult.message -or -not $chatResult.done) { throw 'Invalid non-stream ChatResponse' }
    $chatResult | ConvertTo-Json -Depth 20
}

if ($Mode -in @('All', 'Stream')) {
    $result = Invoke-Adapter '/api/chat' @{
        model = $Model; stream = $true
        messages = @(@{ role = 'user'; content = 'Say hello in Chinese.' })
        options = @{ temperature = 0; num_predict = 128 }
    }
    # WebRequest buffers for assertions. Use curl.exe -N for live display.
    $null = Convert-ChatStream $result
    $result.Content
}

if ($Mode -in @('All', 'Tools')) {
    $dispatchTool = @{
        type = 'function'; function = @{
            name = 'dispatch_action'; description = 'Report the navigation decision for this controlled test.'
            parameters = @{
                type = 'object'; required = @('action', 'tool_name')
                properties = @{ action = @{ type = 'string'; enum = @('open', 'refreshglobalchangecontext', 'none') }; tool_name = @{ type = 'string' } }
            }
        }
    }
    $history = @(
        @{ role = 'system'; content = 'For this controlled protocol test, call dispatch_action exactly once with action="none" and tool_name="". After its result, reply with a short acknowledgement.' },
        @{ role = 'user'; content = 'No navigation is needed. Report the decision using the tool.' }
    )
    $payload = @{ model = $Model; messages = $history; tools = @($dispatchTool); stream = $true; options = @{ temperature = 0; num_predict = 256 } }
    if ($ForceDispatch) {
        # Explicit test-only extension; the adapter never sets this by default.
        $payload.tool_choice = @{ type = 'function'; function = @{ name = 'dispatch_action' } }
    }
    $result = Invoke-Adapter '/api/chat' $payload
    $lines = @(Convert-ChatStream $result)
    $calls = @($lines | ForEach-Object { $_.message.tool_calls } | Where-Object { $null -ne $_ })
    if ($calls.Count -ne 1) { throw "Expected one tool call; got $($calls.Count). Check backend tool selection and adapter tool_call_count." }
    $call = $calls[0]
    if ($call.function.name -ne 'dispatch_action') { throw 'Unexpected tool selected' }
    if ($call.function.arguments -is [string]) { throw 'Tool arguments are a string instead of an object' }
    $argumentFields = @($call.function.arguments.PSObject.Properties.Name)
    if ('action' -notin $argumentFields -or 'tool_name' -notin $argumentFields) { throw 'Tool arguments are missing required fields' }
    if ($call.function.arguments.action -ne 'none' -or $call.function.arguments.tool_name -ne '') { throw 'Unexpected tool arguments' }
    $result.Content

    # This is an explicitly simulated no-op tool result, not a Teamcenter action.
    # Keep only native Ollama fields, as SDK versions may discard call IDs.
    $nextHistory = $history + @(
        @{ role = 'assistant'; content = ''; tool_calls = @(@{ function = @{ name = $call.function.name; arguments = $call.function.arguments } }) },
        @{ role = 'tool'; tool_name = 'dispatch_action'; content = '{"test_mock":true,"action":"none","ok":true}' }
    )
    $next = Invoke-Adapter '/api/chat' @{
        model = $Model; messages = $nextHistory; tools = @($dispatchTool); tool_choice = 'none'
        stream = $false; options = @{ temperature = 0; num_predict = 128 }
    }
    $nextResult = ConvertFrom-Json -InputObject $next.Content
    if (-not $nextResult.message -or -not $nextResult.done) { throw 'Tool-result follow-up failed' }
    $nextResult | ConvertTo-Json -Depth 30
}

if ($Mode -in @('All', 'Copilot')) {
    $result = Invoke-Adapter '/v1/chat/completions' @{
        model = $Model; stream = $false
        messages = @(@{ role = 'user'; content = 'Return a short Chinese answer in the requested JSON schema.' })
        response_format = @{ type = 'json_schema'; json_schema = @{
            name = 'copilot_regression'; strict = $true
            schema = @{ type = 'object'; properties = @{ answer = @{ type = 'string' } }; required = @('answer'); additionalProperties = $false }
        } }
    }
    $completion = ConvertFrom-Json -InputObject $result.Content
    $answer = ConvertFrom-Json -InputObject $completion.choices[0].message.content
    if (-not $answer.answer) { throw 'Copilot JSON schema output is missing answer' }
    $completion | ConvertTo-Json -Depth 30
}

Write-Output "PASS: $Mode (live adapter/model probe; Teamcenter integration still requires its own verification)"

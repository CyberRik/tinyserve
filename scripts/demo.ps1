<#
.SYNOPSIS
    Start TinyServe in the right configuration for each README demo.

.DESCRIPTION
    .\scripts\demo.ps1 page    Config A (n_seq_max=2, defaults otherwise), for Demos 1 and 2.
    .\scripts\demo.ps1 burst   Config B (n_seq_max=4, defaults), for Demo 3.
    .\scripts\demo.ps1 stop    Stop whatever is listening on port 8000.

    page and burst run in the foreground, stream the server log, and stop on Ctrl+C.
    Both warm the model up before announcing ready, so the first take is not a cold one.

    Settings are passed to the server process only. Nothing leaks into your terminal
    session, so switching configs can never inherit a stale TINYSERVE_MAX_QUEUE_DEPTH.

    burst also snapshots benchmarks/results/burst.csv and burst.png on start and puts
    them back on exit. burst.py rewrites both files on every run, so without this each
    recording take would overwrite the sweep that docs/benchmarks.md is built from. To
    publish new benchmark results, run burst.py against a server started without this
    script.

    Renders the terminal demos in docs/media/ via scripts/demo-*.tape.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("page", "burst", "stop")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Port = 8000
$BaseUrl = "http://127.0.0.1:$Port"
$Model = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
$Results = @("benchmarks/results/burst.csv", "benchmarks/results/burst.png")

$Configs = @{
    # Two slots so requests contend and the scheduling policy has choices to make.
    # No queue cap: a browser sends at most ~6 requests at a time, so with a cap
    # low enough to trigger 503s, every rejection instantly frees a connection,
    # the next request finds the queue still full, and ~75% of a burst bounces.
    # Rejection is Demo 3's job, in a terminal, where no connection limit applies.
    page  = @{
        TINYSERVE_N_SEQ_MAX = "2"
    }
    # Matches docs/benchmarks.md #2 exactly. max_queue_depth must be left at its
    # default here: at 3, rejections come back queue_full instead of kv_cache_full
    # and the published 42/22 split does not reproduce.
    burst = @{
        TINYSERVE_N_SEQ_MAX = "4"
    }
}

function Get-PortOwner {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($conn) { return $conn.OwningProcess }
    return $null
}

function Stop-Server {
    $owner = Get-PortOwner
    if (-not $owner) {
        Write-Host "Nothing is listening on port $Port."
        return
    }
    Stop-Process -Id $owner -Force
    Write-Host "Stopped the server on port $Port (pid $owner)."
}

function Wait-Healthy([System.Diagnostics.Process]$proc) {
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { throw "The server exited during startup. Its log is above." }
        try {
            $health = Invoke-RestMethod "$BaseUrl/healthz" -TimeoutSec 2
            if ($health.status -eq "ok") { return }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    throw "The server did not become healthy within 90s."
}

if ($Mode -eq "stop") {
    Stop-Server
    return
}

Set-Location $Repo

if (-not (Test-Path $Model)) {
    throw "Model not found at $Model. Download it into models/ first."
}
$owner = Get-PortOwner
if ($owner) {
    throw "Port $Port is already in use (pid $owner). Run: .\scripts\demo.ps1 stop"
}

# Every TINYSERVE_ variable is set for the child process only. Anything left over
# in this session is blanked for the child, then the session's own values are put
# back, so the child sees exactly $Configs[$Mode] plus the model path.
$wanted = @{ TINYSERVE_MODEL_PATH = $Model } + $Configs[$Mode]
$names = @($wanted.Keys) + @(Get-ChildItem Env: | Where-Object { $_.Name -like "TINYSERVE_*" } |
        ForEach-Object { $_.Name }) | Select-Object -Unique
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }

$backup = $null
$proc = $null
try {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $wanted[$name]) }
    $proc = Start-Process -FilePath "uv" -NoNewWindow -PassThru -ArgumentList @(
        "run", "uvicorn", "tinyserve.api.app:app", "--host", "127.0.0.1", "--port", "$Port")
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }

    Wait-Healthy $proc
    $cfg = Invoke-RestMethod "$BaseUrl/config"
    Invoke-RestMethod "$BaseUrl/generate" -Method Post -ContentType "application/json" `
        -Body '{"prompt":"warm-up","max_tokens":4}' | Out-Null

    if ($Mode -eq "burst") {
        $backup = Join-Path ([IO.Path]::GetTempPath()) "tinyserve-burst-results"
        New-Item -ItemType Directory -Force $backup | Out-Null
        foreach ($f in $Results) {
            if (Test-Path $f) { Copy-Item $f $backup -Force }
        }
    }

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Magenta
    Write-Host " TinyServe ready: $Mode config, warmed up" -ForegroundColor Magenta
    Write-Host "   policy=$($cfg.scheduling_policy)  n_seq_max=$($cfg.n_seq_max)  max_queue_depth=$($cfg.max_queue_depth)"
    if ($Mode -eq "page") {
        Write-Host "   Demo 1: from another terminal, run:  vhs scripts/demo-1.tape"
        Write-Host "   Demo 2: open $BaseUrl/demo, click Send burst, then Compare policies"
    } else {
        Write-Host "   Demo 3: from another terminal, run:  vhs scripts/demo-3.tape"
        Write-Host "   burst.csv/png are snapshotted and restored when this stops."
    }
    Write-Host "   Ctrl+C here to stop."
    Write-Host "================================================================" -ForegroundColor Magenta
    Write-Host ""

    # Poll rather than WaitForExit(): PowerShell can't interrupt a blocking .NET
    # call, so Ctrl+C would not reach the finally block below until the server died.
    while (-not $proc.HasExited) { Start-Sleep -Milliseconds 300 }
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
    $owner = Get-PortOwner
    if ($owner) { Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue }
    if ($backup) {
        foreach ($f in $Results) {
            $copy = Join-Path $backup (Split-Path $f -Leaf)
            if (Test-Path $copy) { Copy-Item $copy $f -Force }
        }
        Write-Host "Restored $($Results -join ', ')."
    }
}

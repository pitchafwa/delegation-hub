# Scheduled-task entry point: runs the daily model refresh (see refresh_all.py) and logs to ingest\logs\scheduled_task.log
$ErrorActionPreference = "Continue"
$ingest = Split-Path -Parent $PSScriptRoot
Set-Location $ingest
New-Item -ItemType Directory -Force -Path (Join-Path $ingest "logs") | Out-Null
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source      # uv on PATH (any PC), else where winget puts it
if (-not $uv) { $uv = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" }
$log = Join-Path $ingest "logs\scheduled_task.log"
# the task also runs at sign-in (2026-10-29: the 6:47am run is skipped when nobody is signed in): only refresh if today's run has not finished yet
$st = Join-Path $ingest "research\data\refresh_status.json"
if (Test-Path $st) {
    $s = Get-Content $st -Raw | ConvertFrom-Json
    if ($s.ok -and $s.finished -and ([datetime]$s.finished).Date -eq (Get-Date).Date) { "=== $(Get-Date -Format s) skipped: today's refresh already finished at $($s.finished) ===" | Out-File -FilePath $log -Append -Encoding utf8; exit 0 }
}
"=== $(Get-Date -Format s) starting ===" | Out-File -FilePath $log -Append -Encoding utf8
& $uv run python research/refresh_all.py 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
"=== $(Get-Date -Format s) finished (exit $LASTEXITCODE) ===" | Out-File -FilePath $log -Append -Encoding utf8

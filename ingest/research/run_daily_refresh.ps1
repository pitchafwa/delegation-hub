# Scheduled-task entry point: runs the daily model refresh (see refresh_all.py) and logs to ingest\logs\scheduled_task.log
$ErrorActionPreference = "Continue"
$ingest = Split-Path -Parent $PSScriptRoot
Set-Location $ingest
New-Item -ItemType Directory -Force -Path (Join-Path $ingest "logs") | Out-Null
$uv = "C:\Users\tommy\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
$log = Join-Path $ingest "logs\scheduled_task.log"
"=== $(Get-Date -Format s) starting ===" | Out-File -FilePath $log -Append -Encoding utf8
& $uv run python research/refresh_all.py 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
"=== $(Get-Date -Format s) finished (exit $LASTEXITCODE) ===" | Out-File -FilePath $log -Append -Encoding utf8

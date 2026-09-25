# Starts the LIVE draft server (see draft_live.py). Add -Lan to reach it from your phone on the same Wi-Fi.
param([switch]$Lan)
$ingest = Split-Path -Parent $PSScriptRoot
Set-Location $ingest
$uv = "C:\Users\tommy\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
if ($Lan) { & $uv run python research/draft_live.py --lan } else { & $uv run python research/draft_live.py }

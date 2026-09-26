# Starts the LIVE draft server (see draft_live.py).
#   -Lan       also reachable from your phone on the same Wi-Fi (prints the exact address to type)
#   -Simulate  TEST MODE: no ESPN, fills the board with fake picks so you can watch the Draft tab update by itself
param([switch]$Lan, [switch]$Simulate)
$ingest = Split-Path -Parent $PSScriptRoot
Set-Location $ingest
$uv = "C:\Users\tommy\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
$args2 = @("run", "python", "research/draft_live.py")
if ($Lan) { $args2 += "--lan" }
if ($Simulate) { $args2 += "--simulate" }
& $uv @args2

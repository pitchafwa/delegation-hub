@echo off
cd /d "C:\Users\tommy\OneDrive\Documents\Claude\fantasy-basketball-hub\ingest"
"C:\Users\tommy\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" run python research/dfs_monitor.py >> logs\dfs_monitor.log 2>&1


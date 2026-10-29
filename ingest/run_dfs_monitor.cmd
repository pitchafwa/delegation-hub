@echo off
rem Runs the DFS monitor (the scheduled task calls this through run_dfs_monitor.vbs). Paths are relative to this file, so it works on any PC.
cd /d "%~dp0"
set "UV=uv"
where uv >nul 2>&1 || set "UV=%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
"%UV%" run python research/dfs_monitor.py >> logs\dfs_monitor.log 2>&1

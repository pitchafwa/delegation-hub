' Runs the DFS monitor with no visible window (the scheduled task calls this through wscript). Finds run_dfs_monitor.cmd next to this file.
Set sh = CreateObject("WScript.Shell")
here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sh.Run "cmd /c """ & here & "\run_dfs_monitor.cmd""", 0, False

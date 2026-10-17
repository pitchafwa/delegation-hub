' Runs the DFS monitor with no visible window (the scheduled task calls this through wscript).
Set sh = CreateObject("WScript.Shell")
sh.Run "cmd /c ""C:\Users\tommy\OneDrive\Documents\Claude\fantasy-basketball-hub\ingest\run_dfs_monitor.cmd""", 0, False

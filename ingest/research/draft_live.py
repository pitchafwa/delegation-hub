"""Local server for the LIVE draft.  Serves the dashboard AND a fast /draft_live.json that reads ESPN's draft picks (and keepers) every few seconds,
so the Draft tab updates by itself while you draft. GitHub Pages can't do this (a static site can't call your private ESPN league), so during the
draft open the LOCAL address this prints instead of the Pages URL.

  uv run python research/draft_live.py              # http://localhost:4180
  uv run python research/draft_live.py --lan        # also reachable from your phone on the same Wi-Fi (prints the address)

Only pick data is exposed; your ESPN cookies never leave this machine.  Ctrl+C to stop.
"""
import json
import socket
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
DASH = Path(__file__).resolve().parent.parent.parent / "dashboard"
PORT = 4180
import os
lg = League(league_id=config.LEAGUE_ID, year=int(os.environ.get("DRAFT_YEAR", "2027")), espn_s2=config.ESPN_S2, swid=config.SWID)
cache = {"t": 0.0, "body": b"{}"}
lock = threading.Lock()


def fetch_picks():
    raw = lg.espn_request.league_get(params={"view": ["mDraftDetail"]})["draftDetail"]
    return json.dumps({"inProgress": bool(raw.get("inProgress")), "drafted": bool(raw.get("drafted")),
                       "picks": [{"n": p["overallPickNumber"], "team": p["teamId"], "keeper": bool(p.get("keeper")),
                                  "player": p["playerId"] if p.get("playerId", -1) > 0 else None} for p in raw["picks"]]}).encode()


class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/draft_live.json":
            with lock:
                if time.time() - cache["t"] > 2.0:          # at most one ESPN call every 2 seconds, however many pages are open
                    try:
                        cache["body"], cache["t"] = fetch_picks(), time.time()
                    except Exception as e:
                        print("ESPN fetch failed:", e)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cache["body"])
            return
        return super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    host = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"
    srv = ThreadingHTTPServer((host, PORT), partial(H, directory=str(DASH)))
    ip = socket.gethostbyname(socket.gethostname()) if host == "0.0.0.0" else "localhost"
    print(f"Draft server running. Open  http://{ip}:{PORT}  (Draft tab).  Ctrl+C to stop.")
    srv.serve_forever()

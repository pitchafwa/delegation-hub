"""Local server for the LIVE draft.  Serves the dashboard AND a fast /draft_live.json that reads ESPN's draft picks (and keepers) every few seconds,
so the Draft tab updates by itself while you draft. GitHub Pages can't do this (a static site can't call your private ESPN league), so during the
draft open the LOCAL address this prints instead of the Pages URL.

  uv run python research/draft_live.py                  # http://localhost:4180
  uv run python research/draft_live.py --lan            # also reachable from your phone on the same Wi-Fi (prints every address this PC has)
  uv run python research/draft_live.py --simulate       # TEST MODE: no ESPN; fills the board one pick every 6 seconds so you can watch the Draft tab update (--simulate 3 = every 3 s)

Only pick data is exposed; your ESPN cookies never leave this machine.  Ctrl+C to stop.
The JSON also carries serverTime / simulated / error so the Draft tab can show whether the connection is alive.
"""
import json
import os
import socket
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
DASH = Path(__file__).resolve().parent.parent.parent / "dashboard"
PORT = 4180
SIM = "--simulate" in sys.argv
SIM_SEC = float(sys.argv[sys.argv.index("--simulate") + 1]) if SIM and len(sys.argv) > sys.argv.index("--simulate") + 1 and sys.argv[sys.argv.index("--simulate") + 1].replace(".", "").isdigit() else 6.0
cache = {"t": 0.0, "data": {}, "error": None, "ok_at": None}
lock = threading.Lock()
START = time.time()

if not SIM:
    import config
    from espn_api.basketball import League
    lg = League(league_id=config.LEAGUE_ID, year=int(os.environ.get("DRAFT_YEAR", "2027")), espn_s2=config.ESPN_S2, swid=config.SWID)
else:
    _dd = json.load(open(DASH / "draft_data.json", encoding="utf-8"))
    _picks = _dd["draft"]["picks"]
    _order = [p["espn_id"] for p in sorted(_dd["players"].values(), key=lambda p: p.get("redraft_rank") or 9999)]     # the "draft" here is just ESPN's redraft rank order


def fetch_picks():
    if SIM:
        kn = _dd["draft"].get("keepers_now") or 3                      # like ESPN: the first rounds are keeper rounds, filled before the draft starts
        n_done = int((time.time() - START) / SIM_SEC)                  # real (non-keeper) picks made so far
        out, k, real = [], 0, 0
        for p in sorted(_picks, key=lambda x: x["n"]):
            if p["r"] <= kn:
                out.append({"n": p["n"], "team": p["team"], "keeper": True, "player": _order[k]})
                k += 1
            else:
                real += 1
                filled = real <= n_done and k < len(_order)
                out.append({"n": p["n"], "team": p["team"], "keeper": False, "player": _order[k] if filled else None})
                if filled:
                    k += 1
        return {"inProgress": real > n_done, "drafted": real <= n_done, "picks": out}
    raw = lg.espn_request.league_get(params={"view": ["mDraftDetail"]})["draftDetail"]
    return {"inProgress": bool(raw.get("inProgress")), "drafted": bool(raw.get("drafted")),
            "picks": [{"n": p["overallPickNumber"], "team": p["teamId"], "keeper": bool(p.get("keeper")),
                       "player": p["playerId"] if p.get("playerId", -1) > 0 else None} for p in raw["picks"]]}


def body():
    with lock:
        if time.time() - cache["t"] > (0.5 if SIM else 2.0):          # at most one ESPN call every 2 seconds, however many pages are open
            try:
                cache["data"], cache["ok_at"], cache["error"] = fetch_picks(), time.time(), None
            except Exception as e:
                cache["error"] = f"{type(e).__name__}: {e}"[:200]
                print("ESPN fetch failed:", cache["error"], flush=True)
            cache["t"] = time.time()
        d = dict(cache["data"])
        d.update({"serverTime": time.time(), "simulated": SIM, "error": cache["error"], "okAt": cache["ok_at"], "filled": sum(1 for p in d.get("picks", []) if p.get("player") and not p.get("keeper"))})
        return json.dumps(d).encode()


class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/draft_live.json":
            b = body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return
        return super().do_GET()

    def log_message(self, *a):
        pass


def lan_addresses():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                       # no packet is sent; this just asks the OS which adapter reaches the internet (your Wi-Fi)
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(i for i in ips if not i.startswith("127.") and not i.startswith("169.254."))


if __name__ == "__main__":
    lan = "--lan" in sys.argv
    host = "0.0.0.0" if lan else "127.0.0.1"
    srv = ThreadingHTTPServer((host, PORT), partial(H, directory=str(DASH)))
    print("=" * 66, flush=True)
    print("DRAFT SERVER RUNNING" + ("  (TEST MODE: simulated picks, ESPN is not contacted)" if SIM else ""), flush=True)
    print(f"  On this PC:      http://localhost:{PORT}", flush=True)
    if lan:
        for ip in lan_addresses():
            print(f"  On your phone:   http://{ip}:{PORT}   <- type exactly this (same Wi-Fi)", flush=True)
    else:
        print("  (phone access is OFF: restart with -Lan)", flush=True)
    print("  Then open the Draft tab. Ctrl+C stops the server.", flush=True)
    print("=" * 66, flush=True)
    srv.serve_forever()

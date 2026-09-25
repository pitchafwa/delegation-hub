"""One daily refresh of everything the dashboard shows that comes from the local model.  Run from ingest/:

    uv run python research/refresh_all.py            # normal daily run (what the scheduled task does)
    uv run python research/refresh_all.py --full     # also rebuild the model chain even if no new games came in
    uv run python research/refresh_all.py --no-push  # do everything but skip git (testing)

Order matters:
  0. FREEZE the preseason breakout ledger on opener day (2026-10-20) using the candidates built BEFORE the first tip, before anything else
     can change them. Refuses (and says so) if the candidates file was built after the first tip.
  1. Market/ESPN inputs: Hashtag dynasty ranks (MKT), ESPN ADP for the live season, ESPN positions/injury status.
  2. In season only (new game logs arrived): pull the current season's game logs -> unify -> Kalman inputs -> Kalman trajectories (updates
     every player's true-talent estimate with the new games) -> prospect trajectories -> asset value (5-8 min).
     NOTE: the empirical (ridge) half of the projection is anchored to last completed season; only the Kalman half moves with in-season games.
  3. Breakout models (they use ADP; skipped once the ledger is frozen because the dashboard then shows the frozen values).
  4. Rebuild dashboard/hub_data.json + breakout_history.json.
  5. Commit and push (GitHub Pages redeploys). A failed step is logged and skipped where safe; a failed model chain aborts the rebuild so a
     half-updated model is never published.
Log: ingest/logs/refresh_<date>.log ; machine-readable status: research/data/refresh_status.json
"""
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
ING = R.parent
REPO = ING.parent
D = R / "data"
LOGDIR = ING / "logs"
LOGDIR.mkdir(exist_ok=True)
ET = ZoneInfo("America/New_York")
OPENER = date(2026, 10, 20)
FIRST_TIP = datetime(2026, 10, 20, 15, 0, tzinfo=ET)     # opening night tips at 3pm ET (RotoWire schedule); conservative
LEDGER_CSV = D / "breakout_ledger" / "2026-27.csv"
args = set(sys.argv[1:])
FULL, NO_PUSH = "--full" in args, "--no-push" in args
now = datetime.now(ET)
past_opener = now.date() >= OPENER
in_season = past_opener or bool(os.environ.get("REFRESH_FORCE_INSEASON"))   # env override is for the plumbing test only

log = open(LOGDIR / f"refresh_{now.date().isoformat()}.log", "a", encoding="utf-8")
status = {"started": now.isoformat(timespec="seconds"), "steps": {}}


def say(msg):
    line = f"[{datetime.now(ET).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def run(name, script, *a, critical=False, timeout=3 * 3600):
    t0 = time.time()
    say(f"START {name}: {script} {' '.join(a)}")
    try:
        p = subprocess.run([sys.executable, str(R / script), *a], cwd=ING, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        ok = p.returncode == 0
        tail = (p.stdout or "")[-600:] + (p.stderr or "")[-600:]
    except Exception as e:                       # timeout etc.
        ok, tail = False, repr(e)
    status["steps"][name] = {"ok": ok, "seconds": round(time.time() - t0)}
    say(f"{'OK   ' if ok else 'FAIL '}{name} ({round(time.time() - t0)}s)" + ("" if ok else f"\n{tail}"))
    if not ok and critical:
        raise RuntimeError(f"critical step failed: {name}")
    return ok


def finish(ok=True, note=""):
    status.update({"finished": datetime.now(ET).isoformat(timespec="seconds"), "ok": ok, "note": note})
    (D / "refresh_status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    say(f"DONE ok={ok} {note}")


try:
    # 0. preseason breakout ledger freeze (opener day or later, once)
    if past_opener and not LEDGER_CSV.exists():
        cand = D / "breakout_candidates.csv"
        built = datetime.fromtimestamp(cand.stat().st_mtime, ET) if cand.exists() else None
        if built and built < FIRST_TIP:
            run("freeze_ledger", "freeze_breakout_ledger.py")
        else:
            say(f"!! NOT freezing: breakout candidates were built {built}, after the first tip. The preseason ledger would not be honest. Decide manually.")
            status["steps"]["freeze_ledger"] = {"ok": False, "seconds": 0, "note": "candidates newer than first tip"}
    ledger_frozen = LEDGER_CSV.exists()

    # 0b. pick up the GitHub Action's latest rosters / weekly plan (the trade finder reads them)
    pl = subprocess.run(["git", "pull", "--rebase", "--autostash", "--quiet"], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    say("git pull: " + ("ok" if pl.returncode == 0 else f"failed ({(pl.stderr or '').strip()[:120]})"))

    # 1. inputs
    run("hashtag_mkt", "pull_hashtag_dynasty.py")
    run("espn_adp", "pull_espn_adp.py", "2027")
    run("espn_positions", "pull_espn_positions.py")

    # 2. model chain, only when new games could have arrived (or --full)
    model_ok = True
    if in_season or FULL:
        games_before = (D / "game_logs_unified.csv").stat().st_size if (D / "game_logs_unified.csv").exists() else 0
        try:
            if in_season:
                run("gamelogs_pull", "pull_recent_gamelogs.py")       # an NBA-site hiccup just means no new games today; the model chain is skipped
                run("gamelogs_unify", "build_game_logs.py")
            grew = (D / "game_logs_unified.csv").stat().st_size != games_before
            if grew or FULL:
                run("kalman_inputs", "prep_kalman_input.py", critical=True)
                run("kalman_trajectories", "kalman_vor.py", critical=True)
                run("prospect_trajectories", "project_prospect_trajectory.py", critical=True)
                run("asset_value", "asset_value_v2.py", critical=True)
            else:
                say("no new games since the last refresh: model chain not rerun")
        except RuntimeError as e:
            model_ok = False
            say(f"!! model chain aborted: {e}. hub_data will NOT be rebuilt from a half-updated model.")

    # 3. breakout models
    if not ledger_frozen:
        run("breakout_model", "breakout_model.py")
        run("rookie_breakout_model", "rookie_breakout_model.py")

    # 4. dashboard data
    if model_ok:
        run("build_hub_data", "build_hub_data.py", critical=True)
        run("build_trades", "build_trades.py")            # reads hub_data, league rosters, weekly plan and the Hashtag values
    else:
        finish(False, "model chain failed; nothing published")
        sys.exit(1)

    # 5. publish
    if NO_PUSH:
        say("--no-push: skipping git")
    else:
        def git(*g):
            return subprocess.run(["git", *g], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        git("add", "dashboard/hub_data.json", "dashboard/trade_ideas.json", "dashboard/breakout_history.json", "ingest/research/data/breakout_ledger", "ingest/research/data/breakout_validation.json",
            "ingest/research/data/rookie_breakout_validation.json", "ingest/research/espn_id_map.json")
        if git("diff", "--cached", "--quiet").returncode == 0:
            say("nothing changed: no commit")
        else:
            msg = f"Daily model refresh {now.date().isoformat()}\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
            c = git("commit", "-m", msg)
            say(f"commit: {(c.stdout or c.stderr).strip()[:120]}")
            git("pull", "--rebase", "--quiet")
            pu = git("push", "--quiet")
            say("pushed" if pu.returncode == 0 else f"PUSH FAILED: {pu.stderr.strip()[:200]}")
            status["steps"]["push"] = {"ok": pu.returncode == 0}
    finish(True)
except Exception as e:
    finish(False, repr(e)[:300])
    raise

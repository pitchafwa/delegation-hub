"""One daily refresh of everything the dashboard shows that comes from the local model.  Run from ingest/:

    uv run python research/refresh_all.py            # normal daily run (what the scheduled task does)
    uv run python research/refresh_all.py --full     # also rebuild the model chain even if no new games came in
    uv run python research/refresh_all.py --no-push  # do everything but skip git (testing)
    HUB_CLOUD=1 ... refresh_all.py --no-push         # the laptop-free test (cloud-test.yml): box scores from ESPN (pull_espn_box.py) instead of
                                                     # stats.nba.com, which blocks GitHub; no git, no PC heartbeat

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
import season_calendar as SCAL
from season import current_season_id, current_season_str
OPENER = SCAL.opener()                                     # derived from the NBA schedule
_o, _t = SCAL.first_tip()
FIRST_TIP = (datetime.fromisoformat(f"{_o.isoformat()}T{_t}:00+00:00").astimezone(ET) if _t else datetime(OPENER.year, OPENER.month, OPENER.day, 15, 0, tzinfo=ET))   # earliest tip on opening day
LEDGER_CSV = D / "breakout_ledger" / f"{current_season_str()}.csv"
args = set(sys.argv[1:])
FULL, NO_PUSH = "--full" in args, "--no-push" in args
CLOUD = bool(os.environ.get("HUB_CLOUD"))
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
    if past_opener and not LEDGER_CSV.exists() and not CLOUD:
        cand = D / "breakout_candidates.csv"
        built = datetime.fromtimestamp(cand.stat().st_mtime, ET) if cand.exists() else None
        if built and built < FIRST_TIP:
            run("freeze_ledger", "freeze_breakout_ledger.py")
        else:
            say(f"!! NOT freezing: breakout candidates were built {built}, after the first tip. The preseason ledger would not be honest. Decide manually.")
            status["steps"]["freeze_ledger"] = {"ok": False, "seconds": 0, "note": "candidates newer than first tip"}
    ledger_frozen = LEDGER_CSV.exists()

    # 0b. pick up the GitHub Action's latest rosters / weekly plan (the trade finder reads them)
    if not CLOUD:
        pl = subprocess.run(["git", "pull", "--rebase", "--autostash", "--quiet"], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        say("git pull: " + ("ok" if pl.returncode == 0 else f"failed ({(pl.stderr or '').strip()[:120]})"))

    # 1. inputs
    run("hashtag_mkt", "pull_hashtag_dynasty.py")
    run("espn_adp", "pull_espn_adp.py", str(current_season_id()))
    run("espn_positions", "pull_espn_positions.py")

    # 2. model chain, only when new games could have arrived (or --full)
    model_ok = True
    if in_season or FULL:
        games_before = (D / "game_logs_unified.csv").stat().st_size if (D / "game_logs_unified.csv").exists() else 0
        try:
            if in_season and CLOUD:
                run("injury_reports", "pull_injury_reports_hourly.py", "update")   # first: the ESPN box scores take their inactive list from it
                run("espn_box", "pull_espn_box.py", current_season_str(), "--prefix", "")   # both NBA-format files at once
            elif in_season:
                run("gamelogs_pull", "pull_recent_gamelogs.py")       # an NBA-site hiccup just means no new games today; the model chain is skipped
                run("box_v3", "pull_box_v3.py", current_season_str())   # did-not-play reasons, for the rotation status below
            if in_season:
                run("rotation_status", "build_rotation_status.py")   # healthy players their coach is not using (build_week_plan.py ROT_TABLE); non-critical
                run("gamelogs_unify", "build_game_logs.py")
                run("form_split", "build_form_split.py")             # "why is he hot": non-critical; the weekly plan ignores a form file older than 4 days
                run("matchup_context", "build_matchup_context.py")   # opponent defense/positional-defense for the matchup adjustment; non-critical
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

    # 2b. injury tiers and tags (in season: pull the newest official reports, rebuild episodes, refit tiers). Non-critical: the old files stay if it fails.
    if in_season:
        if not CLOUD:
            run("injury_reports", "pull_injury_reports_hourly.py", "update")
        run("injury_rows", "injury_playrate_study.py")
        run("injury_episodes", "injury_recurrence_study.py")
    run("injury_risk", "build_injury_risk.py")

    # 3. breakout models
    if not ledger_frozen:
        run("breakout_model", "breakout_model.py")
        run("rookie_breakout_model", "rookie_breakout_model.py")

    # 4. dashboard data
    if model_ok:
        run("build_hub_data", "build_hub_data.py", critical=True)
        run("build_trades", "build_trades.py")            # reads hub_data, league rosters, weekly plan and the Hashtag values
        run("darko_log", "pull_darko.py")                 # DARKO projections (2026-11-09): logged privately for the Dec 14 blend study, and shown as "DARKO" on the player card / Redraft board; non-critical
        run("build_redraft", "build_redraft.py")          # rest-of-season board (ESPN projections, expected games, last-10 logs); non-critical
        run("pull_player_news", "pull_player_news.py")    # news blurbs for the player card (also runs in the game-day workflow); non-critical
    else:
        finish(False, "model chain failed; nothing published")
        sys.exit(1)

    # 4b. trade analyzer context (team posture / odds slopes and the market table): weekly; it needs the local Hashtag file (on the cloud: from the stored data)
    def _age(f):           # days since this weekly file was built; on the cloud a fresh checkout resets file times, so a stamp kept with the data counts
        st = D / f"_built_{f.stem}.stamp" if CLOUD else f
        return (time.time() - st.stat().st_mtime) / 86400 if st.exists() and f.exists() else 99
    _tc = REPO / "dashboard" / "trade_context.json"
    if _age(_tc) > 6 and run("trade_context", "build_trade_context.py") and CLOUD:
        (D / f"_built_{_tc.stem}.stamp").touch()
    # 4b'. Trades Command Center (2026-11-09; replaces the weekly find_trades.py ideas): tags, Target scores, the page's quick-estimate weights re-fit against the full
    # analyzer, and the day's best ideas through the full analyzer -> dashboard/trade_board.json. Daily, on the laptop and in the cloud test. Non-critical.
    run("trade_board", "build_trade_board.py")
    # 4c. a heartbeat the cloud alerts can see: the morning brief warns when this local refresh has stopped running
    if not CLOUD:
        (REPO / "dashboard" / "local_refresh_status.json").write_text(json.dumps({"finished": datetime.now(ET).isoformat(timespec="seconds"),
            "failed": [k for k, v in status["steps"].items() if not v.get("ok")]}), encoding="utf-8")

    # 4d. laptop-free test (2026-11-01, until Tommy decides): ESPN's box scores next to the NBA site's, game by game, for the cloud test's scorecard;
    # also teaches the cloud new players' NBA ids (espn_id_map_learned.json). Non-critical.
    if not CLOUD and in_season:
        if run("cloud_test_learn", "pull_espn_box.py", "--learn") and run("cloud_test_espn", "pull_espn_box.py"):
            run("cloud_test_box", "compare_espn_nba.py", "--out", str(REPO / "dashboard" / "cloud_test_box.json"))
        run("cloud_test_dff", "compare_dff.py")                  # the cloud's DFF monitor log vs this PC's (dashboard/cloud_test_dff.json)

    # 5. publish
    if NO_PUSH:
        say("--no-push: skipping git")
    else:
        def git(*g):
            return subprocess.run(["git", *g], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        git("add", "dashboard/redraft_data.json", "dashboard/season_calendar.json", "dashboard/player_news.json", "dashboard/form_split.json", "dashboard/team_matchup.json", "dashboard/hub_data.json", "dashboard/keeper_history.json", "dashboard/trade_ideas.json", "dashboard/breakout_history.json", "ingest/research/data/breakout_ledger", "ingest/research/data/breakout_validation.json",
            "ingest/research/data/rookie_breakout_validation.json", "ingest/research/espn_id_map.json",
            "dashboard/local_refresh_status.json", "dashboard/rotation_status.json", "dashboard/trade_context.json", "dashboard/trade_market.json", "dashboard/trade_ideas_v2.json", "dashboard/trade_board.json",
            "dashboard/cloud_test_box.json", "dashboard/cloud_test_dff.json", "ingest/research/espn_id_map_learned.json")
        if git("diff", "--cached", "--quiet").returncode == 0:
            say("nothing changed: no commit")
        else:
            msg = f"Daily model refresh {now.date().isoformat()}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
            c = git("commit", "-m", msg)
            say(f"commit: {(c.stdout or c.stderr).strip()[:120]}")
            pr = git("pull", "--rebase", "--autostash", "-X", "theirs", "--quiet")    # a clash on a file the cloud also writes (player_news.json, 2026-11-01): this run's copy wins
            if pr.returncode != 0:
                git("rebase", "--abort")
                say(f"pull before push failed: {(pr.stderr or '').strip()[:200]}")
            pu = git("push", "--quiet")
            say("pushed" if pu.returncode == 0 else f"PUSH FAILED: {pu.stderr.strip()[:200]}")
            status["steps"]["push"] = {"ok": pu.returncode == 0}
    # 6. back up the PC-only pieces (Claude memory and scheduled tasks, Windows tasks) into the OneDrive restore folder (../_restore/RESTORE.md); non-critical
    _bk = REPO.parent / "_restore" / "backup_now.py"
    if _bk.exists() and not CLOUD:
        _b = subprocess.run([sys.executable, str(_bk)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        status["steps"]["restore_backup"] = {"ok": _b.returncode == 0}
        say(((_b.stdout or "") + (_b.stderr or "")).strip()[-300:])
    finish(True)
except Exception as e:
    finish(False, repr(e)[:300])
    raise

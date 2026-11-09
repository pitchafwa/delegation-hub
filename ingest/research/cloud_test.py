"""Laptop-free test: the daily scorecard (2026-11-01, Tommy: run the cloud version side by side for about a month before deciding).
Runs on GitHub (cloud-test.yml) right after the cloud refresh (HUB_CLOUD=1 refresh_all.py --no-push) has rebuilt the dashboard files in this
checkout. Nothing here touches main or the site: the record goes to the cloud-test branch (cloud_test/history.json, latest.json).
  1. the laptop's dashboard files (origin/main) and the cloud's (this checkout) side by side -> compare_cloud_pc.models
  2. the weekly plan built twice, back to back, from the same ESPN league state: on the laptop's files, then on the cloud's -> compare_cloud_pc.plans
  3. what the laptop checked this morning (on main): box scores ESPN vs the NBA site (cloud_test_box.json), DFF monitor logs (cloud_test_dff.json)
  4. how the cloud run itself went (refresh_status.json: finished?, failed steps, minutes)
Run from ingest/:  python research/cloud_test.py OUT_DIR [--recheck]
--recheck (cloud-test-recheck.yml): the same comparison again after the laptop finishes, on this morning's cloud files, replacing today's record.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
REPO = R.parent.parent
DASH = REPO / "dashboard"
sys.path.insert(0, str(R))
import compare_cloud_pc as CMP

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ET = ZoneInfo("America/New_York")
MODEL_FILES = ["hub_data.json", "redraft_data.json", "rotation_status.json", "form_split.json", "team_matchup.json", "trade_ideas.json",
               "trade_context.json", "trade_market.json", "trade_ideas_v2.json", "trade_board.json", "player_news.json", "local_refresh_status.json", "cloud_test_box.json", "cloud_test_dff.json"]
PC_ONLY = ("local_refresh_status.json", "cloud_test_box.json", "cloud_test_dff.json")


def git_show(path):
    p = subprocess.run(["git", "show", f"origin/main:{path}"], cwd=REPO, capture_output=True)
    return p.stdout if p.returncode == 0 else None


tmp = Path(tempfile.mkdtemp())
pc, cl = tmp / "pc", tmp / "cloud"
pc.mkdir()
cl.mkdir()
subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=REPO)
for f in MODEL_FILES:
    b = git_show(f"dashboard/{f}")
    if b is not None:
        (pc / f).write_bytes(b)
    if (DASH / f).exists() and f not in PC_ONLY:
        shutil.copy(DASH / f, cl / f)

rec = {"date": datetime.now(ET).date().isoformat(), "at": datetime.now(ET).isoformat(timespec="minutes"), "recheck": "--recheck" in sys.argv}
st = json.loads((R / "data" / "refresh_status.json").read_text(encoding="utf-8")) if (R / "data" / "refresh_status.json").exists() else {}
if (st.get("started") or "")[:10] != rec["date"]:              # the run died before writing its status: the file is an older run's
    st = {"ok": False, "note": "the cloud run stopped before it finished"}
t0, t1 = st.get("started"), st.get("finished")
rec["cloud_run"] = {"ok": st.get("ok"), "note": st.get("note"), "failed": [k for k, v in st.get("steps", {}).items() if not v.get("ok")],
                    "minutes": round((datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds() / 60, 1) if t0 and t1 else None,
                    "steps": {k: v.get("seconds") for k, v in st.get("steps", {}).items()}}
lr = CMP._load(pc / "local_refresh_status.json") or {}
rec["laptop_run"] = {"finished": lr.get("finished"), "failed": lr.get("failed"),
                     "today": bool(lr.get("finished") and lr["finished"][:10] == rec["date"])}
box = CMP._load(pc / "cloud_test_box.json")
rec["box"] = box if box and box.get("date") == rec["date"] else ({"stale": box.get("date")} if box else None)
dff = CMP._load(pc / "cloud_test_dff.json")
rec["dff"] = dff if dff and dff.get("date") == rec["date"] else ({"stale": dff.get("date")} if dff else None)
rec["models"] = CMP.models(pc, cl)


def plan_on(src):
    """Build the weekly plan with the model files from src in dashboard/, return the plan's path."""
    for f in MODEL_FILES:
        if (src / f).exists() and f not in PC_ONLY:
            shutil.copy(src / f, DASH / f)
    p = subprocess.run([sys.executable, str(R / "build_week_plan.py")], cwd=REPO / "ingest", capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return None, (p.stderr or p.stdout)[-400:]
    dst = src / "week_plan.json"
    shutil.copy(DASH / "week_plan.json", dst)
    return dst, None


cloud_files = {f: (DASH / f).read_bytes() for f in MODEL_FILES if (DASH / f).exists() and f not in PC_ONLY}
pp, e1 = plan_on(pc)
cp, e2 = plan_on(cl)
rec["plan"] = CMP.plans(pp, cp) if pp and cp else {"error": e1 or e2}
for f, b in cloud_files.items():                    # leave the cloud's own files in place (the stored data goes back to R2 as the cloud built it)
    (DASH / f).write_bytes(b)

hist_p = OUT / "history.json"
hist = json.loads(hist_p.read_text(encoding="utf-8")) if hist_p.exists() else []
hist = [h for h in hist if h.get("date") != rec["date"]] + [rec]
hist_p.write_text(json.dumps(hist, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
(OUT / "latest.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
shutil.copytree(cl, OUT.parent / "cloud_dash", dirs_exist_ok=True)                # the cloud's model files, kept in R2 for the recheck
for f in ("hub_data.json", "redraft_data.json", "rotation_status.json"):          # the cloud's own copies, for a closer look when something differs
    if (cl / f).exists():
        shutil.copy(cl / f, OUT / f"cloud_{f}")
print(json.dumps({k: rec[k] for k in ("cloud_run", "laptop_run")}, indent=1))
print("plan:", json.dumps(rec["plan"])[:600])

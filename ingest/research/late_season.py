"""Late-season motivation (2026-10-25): how much each NBA team's core players and young players are expected to play in OUR fantasy playoffs (weeks 20-22),
given where the team will sit in the standings and what it is playing for. Writes dashboard/late_season.json; read by build_week_plan.py (playoff-week games in the
long-run costs) and the site (season value, a flag on player cards).

Evidence (late_season_study.py, 2012-26, the matching stretch of each season: last 34 to 14 days of the NBA regular season): stars on teams with a lottery reason to
lose lost about 20% of their fantasy points per team game against teams in a race, all from games missed (minutes when playing were normal); teams with a weaker
reason lost about 10%. Behaviour followed the incentive: after the 2019 lottery reform cut the bottom-3 edge, their gap shrank (0.79 -> 0.91) while the 4th-10th
worst, newly favoured, rested more (0.89 -> 0.81). Young players (23 or younger) on bottom teams gained about 4-12%.

THIS SEASON'S RULES (2027 draft, '3-2-1' lottery): the three worst teams are relegated to 2 balls (5.4% for #1, worse than 4th-10th at 3 balls / 8.1%, all equal), so no
team gains by sinking to the bottom and nobody in 4th-10th gains by losing more. The multipliers below are therefore small: what is left is the 'nothing to play for'
rest that happens without a lottery reason (health shutdowns), set at about half of the smallest gap ever measured. A team that is visibly tanking anyway (for example
one that does not own its pick) can be set by hand in OVERRIDES. Relative to a team in the race (= 1.0):
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from team_abbr import canon    # noqa: E402

HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
CORE_MULT = {"secure": 1.0, "race": 1.0, "bottom4_10": 0.95, "bottom3": 0.96}
YOUNG_MULT = {"secure": 1.0, "race": 1.0, "bottom4_10": 1.04, "bottom3": 1.04}
OVERRIDES = {}      # e.g. {"LAC": {"core": 0.85, "note": "tanking: pick owed"}}  -- set by hand once behaviour is visible (February)
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
PRIOR_GAMES = 20     # early season: a team's record is pulled toward its roster strength with this many games' weight


def main():
    cal = json.load(open(HUB / "season_calendar.json", encoding="utf-8"))
    sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))["games"]
    po = [p for p in cal["periods"] if p.get("playoff")]
    w0 = date.fromisoformat(po[0]["start"])
    today = date.today()
    # games each team still plays before the window opens
    left = {}
    for d, gs in sched.items():
        if today.isoformat() <= d < w0.isoformat():
            for a, h, _ in gs:
                for t in (canon(a), canon(h)):
                    left[t] = left.get(t, 0) + 1
    # current records
    js = requests.get("https://site.api.espn.com/apis/v2/sports/basketball/nba/standings", timeout=20).json()
    rec = {}
    for c in js["children"]:
        for e in c["standings"]["entries"]:
            st = {s["name"]: s.get("value") for s in e["stats"]}
            rec[canon(e["team"]["abbreviation"])] = (int(st.get("wins") or 0), int(st.get("losses") or 0))
    # roster strength prior from the redraft board: top-8 projected points per game
    RD = json.load(open(HUB / "redraft_data.json", encoding="utf-8"))
    by_team = {}
    for p in RD["players"]:
        by_team.setdefault(canon(p.get("team") or ""), []).append(p)
    strength = {t: sum(sorted((x["ppg"] for x in v), reverse=True)[:8]) for t, v in by_team.items() if t in rec}
    sv = np.array(list(strength.values()))
    prior = {t: float(np.clip(0.5 + 0.12 * (s - sv.mean()) / sv.std(), 0.2, 0.8)) for t, s in strength.items()}
    teams = sorted(rec)
    pw = np.array([(rec[t][0] + PRIOR_GAMES * prior.get(t, 0.5)) / (sum(rec[t]) + PRIOR_GAMES) for t in teams])
    rng = np.random.default_rng(1)
    n_sim = 4000
    status_cnt = {t: {"secure": 0, "race": 0, "bottom4_10": 0, "bottom3": 0} for t in teams}
    for _ in range(n_sim):
        q = np.clip(pw + rng.normal(0, 0.05, len(teams)), 0.05, 0.95)          # uncertainty in true strength
        wins = np.array([rec[t][0] for t in teams]) + rng.binomial([left.get(t, 0) for t in teams], q)
        games = np.array([sum(rec[t]) + left.get(t, 0) for t in teams])
        pct = wins / np.maximum(games, 1) + rng.uniform(0, 1e-6, len(teams))
        order = np.argsort(pct)                                                 # worst first
        lg_rank = {teams[i]: r + 1 for r, i in enumerate(order)}
        for conf in (True, False):
            idx = [i for i, t in enumerate(teams) if (t in EAST) == conf]
            for seed, i in enumerate(sorted(idx, key=lambda i: -pct[i]), start=1):
                t = teams[i]
                r = lg_rank[t]
                status_cnt[t]["bottom3" if r <= 3 else ("bottom4_10" if r <= 10 else ("secure" if seed <= 5 else "race"))] += 1
    # core players (top 3 by projected ppg on the team) and young players (23 or younger)
    out_teams = {}
    for t in teams:
        pr = {k: v / n_sim for k, v in status_cnt[t].items()}
        core_m = sum(pr[k] * CORE_MULT[k] for k in pr)
        young_m = sum(pr[k] * YOUNG_MULT[k] for k in pr)
        ov = OVERRIDES.get(t, {})
        pl = sorted(by_team.get(t, []), key=lambda x: -x["ppg"])
        out_teams[t] = {"record": rec[t], "p": {k: round(v, 3) for k, v in pr.items()}, "core_mult": round(ov.get("core", core_m), 3), "young_mult": round(ov.get("young", young_m), 3),
                        "core": [x["id"] for x in pl[:3]], "young": [x["id"] for x in pl if x.get("age") is not None and x["age"] <= 23.5],
                        "note": ov.get("note", "")}
    res = {"generated": date.today().isoformat(), "window": [po[0]["start"], po[-1]["end"]], "playoff_weeks": [p["id"] for p in po],
           "core_mult_by_status": CORE_MULT, "young_mult_by_status": YOUNG_MULT, "teams": out_teams,
           "note": "expected share of normal playing time in the fantasy playoff weeks; late_season.py / late_season_study.py"}
    (HUB / "late_season.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    worst = sorted(out_teams.items(), key=lambda kv: kv[1]["core_mult"])[:6]
    print("wrote late_season.json; lowest core multipliers:", [(t, v["core_mult"], v["record"]) for t, v in worst])


if __name__ == "__main__":
    main()

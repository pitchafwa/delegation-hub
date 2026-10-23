"""Grade the DailyFantasyFuel projections against what happened, and measure how fast they react to injury news. Reads the local logs written by dfs_monitor.py.

  uv run python research/pull_recent_gamelogs.py       (refresh the box scores first)
  uv run python research/grade_dfs.py [--since 2026-10-20]      (from ingest/)

PART 1 - ACCURACY. For every listed player-game, the projection in force at tip-off = the last logged value before his team's tip (dashboard/nba_schedule.json). Actual = this league's
fantasy points from the box score (players who played). Compared with ESPN's level and our own model's level on the same player-games (the shadow log dashboard/projection_log.csv, when it has
them): mean error (bias), mean absolute error, correlation; split by DFF injury tag. Also: of players tagged Questionable, how many played, and was the projection about what they scored when
they did (tests that the number is conditional on playing).
PART 2 - SPEED. For every player the ESPN feed marked Out/Doubtful before his game: minutes from ESPN's own update time to the first DFF snapshot showing him out (projection 0 or tag O), as a
distribution; how many were still projected to play at the last snapshot before tip ('stale at tip'). Snapshots are about 10 minutes apart, so the speed numbers have that resolution.
Writes data/dfs/grade_report.txt and prints it.
"""
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data" / "dfs"
HUB = ROOT.parent.parent / "dashboard"


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


def tips_for(day):
    """team -> tip-off (UTC datetime) of the game on that ET date, from the NBA schedule file"""
    sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))
    out = {}
    y, m, d = map(int, day.split("-"))
    for a, h, t in sched.get("games", {}).get(day, []):
        try:
            hh, mm = int(t[:2]), int(t[3:5])
        except (TypeError, ValueError):
            continue
        dt = datetime(y, m, d, hh, mm, tzinfo=timezone.utc)
        if hh < 10:
            dt += timedelta(days=1)
        out[a] = out[h] = dt
    return out


def main():
    since = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--since"), "2026-10-20")
    log = pd.read_csv(D / "log.csv")
    log["t"] = pd.to_datetime(log["fetched_utc"], utc=True)
    log["key"] = log["name"].map(norm)
    lines = []
    P = lines.append
    import form_common as F
    games = F.load_all(cache=False)
    games["key"] = games["name"].map(norm)
    games["day"] = games["date"].dt.strftime("%Y-%m-%d")
    act = games.drop_duplicates(["day", "key"]).set_index(["day", "key"])["fp"]
    proj_log = HUB / "projection_log.csv"
    sh = pd.read_csv(proj_log) if proj_log.exists() else pd.DataFrame(columns=["date", "name", "espn", "model"])
    sh["key"] = sh["name"].map(norm)
    shs = sh.drop_duplicates(["date", "key"]).set_index(["date", "key"])[["espn", "model"]].apply(pd.to_numeric, errors="coerce")
    # ---------------- PART 1
    rows = []
    for day, g in log[log.game_date >= since].groupby("game_date"):
        tips = tips_for(day)
        for (key, name), h in g.groupby(["key", "name"]):
            tip = tips.get(h["team"].iloc[0])
            if tip is None:
                continue
            before = h[h.t < tip].sort_values("t")
            if before.empty:
                continue
            last = before.iloc[-1]
            played = (day, key) in act.index
            rows.append({"day": day, "name": name, "tag": last["inj"] if isinstance(last["inj"], str) else "", "dff": last["league_proj"], "dk": last["dk_proj"], "played": played,
                         "actual": float(act.loc[(day, key)]) if played else np.nan,
                         "espn": float(shs.loc[(day, key), "espn"]) if (day, key) in shs.index else np.nan,
                         "model": float(shs.loc[(day, key), "model"]) if (day, key) in shs.index else np.nan})
    df = pd.DataFrame(rows)
    P(f"DFF projections graded since {since}: {len(df)} listed player-games, {int(df.played.sum()) if len(df) else 0} played")

    def stats(name, col, d):
        d = d.dropna(subset=[col, "actual"])
        d = d[d[col] > 0]
        if len(d) < 5:
            return
        e = d["actual"] - d[col]
        P(f"  {name:<10} n={len(d):<5} bias {e.mean():+5.2f}  MAE {e.abs().mean():5.2f}  corr {np.corrcoef(d['actual'], d[col])[0, 1]:.3f}")

    if len(df):
        pl = df[(df.played) & (df.dff > 0)]
        P("\nPART 1 - accuracy on players who played (error = actual - projection):")
        stats("DFF", "dff", pl)
        P("same player-games where ESPN's level is logged:")
        c = pl.dropna(subset=["espn"])
        stats("DFF", "dff", c)
        stats("ESPN", "espn", c)
        P("same player-games where our model's level is logged:")
        c = pl.dropna(subset=["model"])
        stats("DFF", "dff", c)
        stats("our model", "model", c)
        P("\nby DFF injury tag at tip (players who played):")
        for tag, d in pl.groupby("tag"):
            if len(d) >= 5:
                e = d["actual"] - d["dff"]
                P(f"  tag '{tag or '-'}': n={len(d)} bias {e.mean():+.2f} MAE {e.abs().mean():.2f}")
        q = df[(df.tag.str.upper().isin(["Q", "GTD"])) & (df.dff > 0)]
        if len(q) >= 5:
            P(f"\nQuestionable-tagged: {len(q)} listed, {int(q.played.sum())} played ({q.played.mean():.0%}); among those who played, projection {q[q.played].dff.mean():.1f} vs actual {q[q.played].actual.mean():.1f}")
            P("  (near-zero bias = the number is 'if he plays'; actual far below = it is already discounted for the risk of sitting)")
    # ---------------- PART 2
    P("\nPART 2 - reaction speed to injury news:")
    ip = D / "inj_log.csv"
    if ip.exists():
        inj = pd.read_csv(ip)
        inj["t"] = pd.to_datetime(inj["fetched_utc"], utc=True)
        inj["key"] = inj["name"].map(norm)
        lag, stale, n_out = [], [], 0
        for day, g in log[log.game_date >= since].groupby("game_date"):
            tips = tips_for(day)
            for key, h in g.groupby("key"):
                tip = tips.get(h["team"].iloc[0])
                if tip is None:
                    continue
                ii = inj[(inj.key == key) & (inj.status.isin(["Out", "Doubtful"])) & (inj.t < tip) & (inj.t >= tip - timedelta(hours=20))]
                if ii.empty:
                    continue
                n_out += 1
                first = ii.sort_values("t").iloc[0]
                et = pd.to_datetime(first["espn_time"], utc=True, errors="coerce")
                news_t = et if (pd.notna(et) and et >= tip - timedelta(hours=20)) else first["t"]
                h = h.sort_values("t")
                is_out = h.dk_proj <= 0                                   # a zero projection = he is out (the tag is often missing)
                pre = h[h.t < tip]
                if pre.empty or not bool(is_out.loc[pre.index[-1]]):
                    stale.append((day, h["name"].iloc[0], float(pre.iloc[-1].dk_proj) if len(pre) else None))
                else:
                    # when did he become out for good before tip (the start of the final run of zero rows); negative lag = DFF had him out before ESPN's news
                    flags = is_out.loc[pre.index].tolist()
                    k = len(flags) - 1
                    while k > 0 and flags[k - 1]:
                        k -= 1
                    lag.append((pre.iloc[k].t - news_t).total_seconds() / 60)
        P(f"  players ESPN marked Out/Doubtful before tip and listed by DFF: {n_out}")
        if lag:
            a = np.array(lag)
            P(f"  DFF showed them out after: median {np.median(a):.0f} min, 75th pct {np.percentile(a, 75):.0f}, 90th {np.percentile(a, 90):.0f}, max {a.max():.0f}  (n={len(a)}; negative = DFF was first)")
        P(f"  STALE AT TIP (DFF still projected them to play at the last snapshot before tip): {len(stale)}" + (": " + ", ".join(f"{n} {d} ({p})" for d, n, p in stale[:8]) if stale else ""))
    else:
        P("  no ESPN injury log yet")
    txt = "\n".join(lines)
    (D / "grade_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

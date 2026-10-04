"""Backtest of the Draft tab's "chance he is still available at your next pick" simulation against this league's real drafts (2021-2026).

For every real pick t and every team's NEXT pick (g picks later), simulate the picks in between with a candidate model, then compare the model's
"still available" probability for each of the top-60 remaining players with whether he really was. Scored with Brier score (lower is better) plus a
calibration table (when it says 30%, how often was he really there?).

Models compared (see MODELS): the current page logic (ADP order, tight noise) vs wider noise fit to this league vs the rookie fix (price rookies by their NBA
draft slot, because ESPN ADP badly under-ranks rookies here).  Leave-one-year-out for everything that is fit.
  uv run python research/draft_avail_backtest.py        (from ingest/)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
rng = np.random.default_rng(7)


def key(n):
    return re.sub(r"[^a-z]", "", str(n).lower().replace(" jr.", "").replace(" iii", "").replace(" ii", ""))


d = pd.read_csv(R / "data" / "draft_history_2021_2026.csv")
d = d[~d.keeper].sort_values(["year", "n"]).copy()
d["rp"] = d.groupby("year").cumcount() + 1
d["k"] = d.name.map(key)
links = pd.read_csv(R / "data" / "college_to_nba_links.csv", encoding="latin1")
links["k"] = links.college_player.map(key)
d = d.merge(links[["k", "real_draft_year", "real_draft_number"]].drop_duplicates("k"), on="k", how="left")
d["nba_pick"] = np.where(d.real_draft_year == d.year - 1, d.real_draft_number, np.nan)
d["adp_ok"] = d.adp.where(d.adp < 130)
d["prk"] = d.groupby("year").prior_tot.rank(ascending=False)
d["pool_n"] = d.groupby("year").rp.transform("max")
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def fit_rookie_curve(train):
    """log(real pick) ~ a + b*log(NBA pick) on rookies with a known NBA slot (2023+ only: the league's rookie pricing changed once keepers/dynasty took hold)."""
    r = train[(train.rookie) & train.nba_pick.notna() & (train.year >= 2023)]
    if len(r) < 8:
        return None
    b, a = np.polyfit(np.log(r.nba_pick), np.log(r.rp), 1)
    resid = np.log(r.rp) - (a + b * np.log(r.nba_pick))
    return a, b, resid.std()


def fit_prod_map(train):
    """map prior-production rank -> ADP units for veterans with no ADP: log(adp) ~ a + b*log(prk)"""
    v = train[(~train.rookie) & train.adp_ok.notna() & train.prk.notna()]
    b, a = np.polyfit(np.log(v.prk), np.log(v.adp_ok), 1)
    return a, b


def scores(g, curve, pmap, rookie_fix):
    """ADP-unit price per player (smaller = goes earlier); NaN price -> 'never in the simulated pool' like the page does for players without ADP"""
    s = g.adp_ok.copy()
    if pmap is not None:
        miss = s.isna() & (~g.rookie) & g.prk.notna()
        s[miss] = np.exp(pmap[0] + pmap[1] * np.log(g.prk[miss]))
    if rookie_fix and curve is not None:
        rk = g.rookie & g.nba_pick.notna()
        s[rk] = np.exp(curve[0] + curve[1] * np.log(g.nba_pick[rk]))
    return s


def simulate(prices, taken_mask, gap, c0, c1, n_sims=120):
    """prices: array (nan = excluded); returns P(still available after `gap` picks) per player"""
    idx = np.where(~taken_mask & ~np.isnan(prices))[0]
    idx = idx[np.argsort(prices[idx])][: gap + 60]
    cnt = np.zeros(len(prices))
    for _ in range(n_sims):
        gone = np.zeros(len(prices), bool)
        for _g in range(gap):
            best, bv, seen = -1, -1e9, 0
            for j in idx:
                if gone[j]:
                    continue
                seen += 1
                if seen > 40:
                    break
                u = -seen + (-np.log(-np.log(rng.random() + 1e-12) + 1e-12)) * (c0 + c1 * seen)
                if u > bv:
                    bv, best = u, j
            if best >= 0:
                gone[best] = True
        cnt[idx] += ~gone[idx]
    return {j: cnt[j] / n_sims for j in idx}


MODELS = {
    "current page (ADP only, tight noise)":      dict(c0=1.5, c1=0.12, pmap=False, rookie=False),
    "+ wider noise":                              dict(c0=4.0, c1=0.25, pmap=False, rookie=False),
    "+ much wider noise":                         dict(c0=6.0, c1=0.35, pmap=False, rookie=False),
    "+ wider noise, vets without ADP priced":     dict(c0=4.0, c1=0.25, pmap=True, rookie=False),
    "+ rookies priced by NBA slot (full fix)":    dict(c0=4.0, c1=0.25, pmap=True, rookie=True),
}


def evaluate(cfg, years=YEARS, step=3, gaps=(10, 22)):
    """Squared error summed over the SAME players for every model: everyone who is picked within the next gap+60 picks. A player the model never simulates
    (no price) counts as 'still available' (p=1), which is exactly what the page shows for him today."""
    sse, n, cal = 0.0, 0, []
    for y in years:
        train = d[d.year != y]
        g = d[d.year == y].reset_index(drop=True)
        curve, pmap = fit_rookie_curve(train), fit_prod_map(train)
        pr = scores(g, curve, pmap if cfg["pmap"] else None, cfg["rookie"]).to_numpy(float)
        for t in range(0, len(g) - 30, step):
            taken = (g.rp <= t).to_numpy()
            for gap in gaps:
                if t + gap >= len(g):
                    continue
                p = simulate(pr, taken, gap, cfg["c0"], cfg["c1"], n_sims=60)
                uni = np.where((g.rp > t) & (g.rp <= t + gap + 60))[0]
                for j in uni:
                    pa = p.get(j, 1.0)
                    a = 0.0 if (g.rp[j] <= t + gap) else 1.0
                    sse += (pa - a) ** 2
                    n += 1
                    cal.append((pa, a, 1.0 if g.rookie[j] else 0.0))
    return sse, n, np.array(cal)


if __name__ == "__main__":
    print("rookie curve (all years, 2023+):", fit_rookie_curve(d))
    yrs = [2022, 2023, 2024, 2025, 2026]
    for name, cfg in MODELS.items():
        b, n, cal = evaluate(cfg, years=yrs)
        bins = np.digitize(cal[:, 0], [0.1, 0.3, 0.5, 0.7, 0.9])
        tab = " ".join(f"[{['<10','10-30','30-50','50-70','70-90','>90'][i]}%: said {cal[bins==i,0].mean()*100:.0f} real {cal[bins==i,1].mean()*100:.0f} n={int((bins==i).sum())}]" for i in range(6) if (bins == i).any())
        rk = cal[cal[:, 2] == 1]
        print(f"\n{name}\n  ALL: squared error {b:.0f} over {n:,} predictions (lower is better)\n  ROOKIES ONLY: squared error {((rk[:,0]-rk[:,1])**2).sum():.1f} over {len(rk)} (model said {rk[:,0].mean()*100:.0f}% available on average, really {rk[:,1].mean()*100:.0f}%)\n  {tab}")

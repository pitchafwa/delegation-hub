"""Why is he hot?  For every player with 10+ games this season: split his recent form (last 10 games) vs his baseline (previous 365 days) into minutes / shooting luck / shot volume / rebounds / assists /
steals+blocks / other, apply the fitted persistence weights (form_model.json, out-of-sample tested on 2024-25 and 2025-26), and write dashboard/form_split.json keyed by normalised name.
  uv run python research/build_form_split.py                       (uses the newest game logs; the daily local refresh runs it)
  FORM_AS_OF=2026-01-15 uv run python research/build_form_split.py  (test: pretend it is that date in a finished season)"""
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("FORM_OUT") or HERE.parent.parent / "dashboard" / "form_split.json")
M = json.load(open(HERE / "form_model.json"))
C = M["coef"]
W = 10
BLEND_B = M["blend"]

norm = lambda n: re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower().replace(" jr.", "").replace(" jr", "").replace(" iii", "").replace(" ii", ""))
LABEL = {"minutes": "minutes", "luck3": "3-pt shooting", "luck2": "2-pt shooting", "luckf": "free-throw shooting", "volume": "shot volume", "o_reb": "rebounding", "o_ast": "assists", "o_stk": "steals/blocks", "o_misc": "turnovers/other"}
STICK = {"minutes": "lasts", "volume": "mostly lasts", "o_ast": "mostly lasts", "o_reb": "half lasts", "o_stk": "fades", "o_misc": "fades", "luck3": "fades", "luck2": "mostly fades", "luckf": "mostly fades"}

g = F.load_all(cache=False)
asof = pd.Timestamp(os.environ["FORM_AS_OF"]) if os.environ.get("FORM_AS_OF") else g["date"].max()
g = g[g["date"] <= asof]
season = g.loc[g["date"].idxmax(), "season"]
print("as of", str(asof)[:10], "season", season)
out = {}
for pid, pl in g.groupby("pid", sort=False):
    pl = pl.reset_index(drop=True)
    cur = pl[pl["season"] == season]
    if len(cur) < 5 or (asof - cur["date"].iloc[-1]).days > 21:
        continue
    name, team = pl["name"].iloc[-1], pl["team"].iloc[-1]
    w = pl.iloc[-W:] if len(cur) >= W else pl.iloc[-len(cur):]
    s0 = w.index[0]
    b = pl[(pl["date"] >= w["date"].iloc[0] - pd.Timedelta(days=365)) & (pl.index < s0)]
    rec = {"n": int(len(w)), "fp": round(float(w.fp.mean()), 1), "m": round(float(w["min"].mean()), 1), "ppm": round(float(w.fp.sum() / w["min"].sum()), 2), "team": team,
           "m3": round(float(pl["min"].iloc[-3:].mean()), 1), "fp3": round(float(pl.fp.iloc[-3:].mean()), 1)}
    if len(b) >= 30 and b["min"].mean() >= 8 and len(cur) >= W:
        r = F.split(w, b)
        rec.update({"fp_b": round(r["fp_b"], 1), "m_b": round(r["m_b"], 1), "ppm_b": round(r["fp_b"] / r["m_b"], 2)})
        x = {k: r[k] for k in ("minutes", "luck3", "luck2", "luckf", "volume", "o_reb", "o_ast", "o_stk")}
        x["o_misc"] = r["o_tov"] + r["o_rest"]
        x["trend"] = float(pl.fp.iloc[-3:].mean() - w.fp.mean())
        x["mtrend"] = float(pl["min"].iloc[-3:].mean() - w["min"].mean())
        x["young"], x["old"] = 0.0, 0.0                       # age is left to ESPN's projection; only the composition of the form is used for the level adjustment
        d = r["fp_w"] - r["fp_b"]
        pred_comp = sum(C[k] * x[k] for k in M["features"])
        chips = sorted(({"k": LABEL[k], "v": round(float(x[k]), 1), "p": STICK[k]} for k in LABEL if abs(x[k]) >= 0.6), key=lambda c: -abs(c["v"]))[:4]
        rec.update({"d": round(d, 1), "keep": round(pred_comp + C["int"], 1), "chips": chips})
    out[norm(name)] = rec
Path(OUT).write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "as_of": str(asof)[:10], "season": season, "players": out}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
n_split = sum(1 for v in out.values() if "comp" in v)
print(f"wrote {OUT.name}: {len(out)} players, {n_split} with a full split")
hot = sorted((v for v in out.values() if "d" in v and v["fp"] >= 25), key=lambda v: -v["d"])[:6]
for k, v in sorted(out.items(), key=lambda kv: -kv[1].get("d", -99))[:8]:
    print(k, "recent", v["fp"], "base", v.get("fp_b"), "d", v.get("d"), "->keep", v.get("keep"), [(c["k"], c["v"]) for c in v.get("chips", [])])

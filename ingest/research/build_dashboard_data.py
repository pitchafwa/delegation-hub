"""Build the rookie-prospect dashboard's data file, mirroring the exact
WRPI/RUPI scores.json shape so the cloned dashboard HTML can render it
unmodified. Scores EVERY player in the dataset (including current 2025-26
draft classes with no resolved outcome yet), but percentiles/star/diamond
are always computed against the frozen, fully-matured reference pool
(real_draft_year <= 2018) -- never let unproven recent classes shift
historical percentiles, same convention as WRPI/RUPI.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT.parent.parent / "dashboard" / "rookie_scores.json"

# ---------------------------------------------------------------------------
# Load + rebuild every feature exactly as fit_rookie_model_a2.py does, but
# on the FULL unfiltered dataset (that module restricts to real_draft_year
# <= 2018 and drops unresolved rows at import time -- not usable here).
# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
POWER_CONFS = {"ACC", "B10", "B12", "BE", "P10", "P12", "SEC", "Amer"}
df["power_conf"] = df["conf"].isin(POWER_CONFS).astype(int)
df["fg_pct_filled"] = df["fg_pct"].fillna(df["fg_pct"].median())
df["three_pct_filled"] = df["three_pct"].fillna(df["three_pct"].median())
df["ft_pct_filled"] = df["ft_pct"].fillna(df["ft_pct"].median())
df["ppg_c"] = df["ppg"]
df["apg_c"] = df["apg"]
df["had_combine"] = df["LANE_AGILITY_TIME_PCTILE"].notna().astype(int)
for c in ["WINGSPAN_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE"]:
    df[c] = df[c].fillna(df[c].median())
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["breakout_age_filled"] = df["breakout_age_filled"].fillna(df["breakout_age_filled"].max())
df["never_broke_out"] = df["never_broke_out"].fillna(1).astype(int)
for c in ["bpm", "porpag", "usg", "ts", "ortg"]:
    df[c] = df[c].fillna(df[c].median())
# undrafted players (no real NBA draft slot) -- treat as just past the draft
# for the capital curve, same convention RUPI used for its UDFA pool
df["pick_filled"] = df["real_draft_number"].fillna(61.0)

frozen = json.loads((ROOT / "data" / "output_a_model.frozen.json").read_text())
PRE = frozen["pre_draft_model"]
POST = frozen["post_draft_model"]
CAL = frozen["star_diamond_calibration"]
DIAMOND = CAL["diamond"]


def score_with_components(params, data, feature_list):
    idx = 0
    components = {}
    total = np.zeros(len(data))
    for f in feature_list:
        lo_x, span, weight = params[idx], params[idx + 1], params[idx + 2]
        idx += 3
        hi_x = lo_x + span
        x = data[f].to_numpy(dtype=float)
        frac = np.clip((x - lo_x) / max(hi_x - lo_x, 1e-6), 0.0, 1.0)
        contrib = weight * frac
        components[f] = contrib
        total += contrib
    intercept = params[idx]
    total += intercept
    return total, components, intercept


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


# ---------------------------------------------------------------------------
# Pre-draft score (single model, clean per-feature decomposition)
# ---------------------------------------------------------------------------
pre_total, pre_components, pre_intercept = score_with_components(PRE["params"], df, PRE["features"])
df["pre_draft_score"] = pre_total

# ---------------------------------------------------------------------------
# Post-draft ensemble: rank-average of 3 components. Keep the "champion"
# component's own raw score + decomposition for the detail view (cleanest
# single-model story), plus each component's own rank for transparency.
# ---------------------------------------------------------------------------
component_scores = {}
component_ranks = []
champion_components = None
champion_intercept = None
champion_capital_params = None
for name, cfg in POST["components"].items():
    n_pre = cfg["n_pre_draft_params"]
    p = cfg["params"]
    talent_total, comps, intercept = score_with_components(p[:n_pre], df, cfg["features"])
    cap_part = capital_curve(p[n_pre:], df["pick_filled"].to_numpy(dtype=float))
    raw = talent_total + cap_part
    component_scores[name] = raw
    component_ranks.append(rankdata(raw))
    if name == "champion":
        champion_components = comps
        champion_intercept = intercept
        champion_capital_params = p[n_pre:]
        champion_capital_contrib = cap_part

post_rank_avg = np.mean(component_ranks, axis=0)
df["post_draft_score"] = post_rank_avg  # already an ensemble rank-average, used as the "raw" display value
df["post_draft_score_champion_raw"] = component_scores["champion"]
df["post_draft_score_rec_variant_raw"] = component_scores["rec_variant"]
df["post_draft_score_breakout_variant_raw"] = component_scores["breakout_variant"]

# ---------------------------------------------------------------------------
# Percentiles -- ALWAYS against the frozen, fully-matured reference pool,
# never the full (including-recent-classes) population.
# ---------------------------------------------------------------------------
REFERENCE_MAX_YEAR = 2018
ref_mask = (df["real_draft_year"] <= REFERENCE_MAX_YEAR) & df["age_22_29_best3"].notna()
ref_pool = df[ref_mask]
print(f"Reference pool: {len(ref_pool)} players, classes {int(ref_pool['real_draft_year'].min())}-{int(ref_pool['real_draft_year'].max())}")


def to_percentile(x, ref):
    ref_sorted = np.sort(ref)
    return np.searchsorted(ref_sorted, x, side="right") / len(ref_sorted)


df["pre_draft_pctl"] = to_percentile(df["pre_draft_score"].to_numpy(), ref_pool["pre_draft_score"].to_numpy())
df["post_draft_pctl"] = to_percentile(df["post_draft_score"].to_numpy(), ref_pool["post_draft_score"].to_numpy())
df["tier"] = np.ceil(df["post_draft_pctl"] * 10).clip(1, 10).astype(int)

STAR_PCTL = CAL["star"]["star_pctl"]
df["is_star_pre"] = (df["pre_draft_pctl"] >= STAR_PCTL).astype(int)
df["is_star_post"] = (df["post_draft_pctl"] >= STAR_PCTL).astype(int)

# ---------------------------------------------------------------------------
# Diamond -- apply the frozen weights/means/stds/threshold exactly as fit
# ---------------------------------------------------------------------------
CUT_PICK = DIAMOND["chosen_cut_pick"]
FLAG_THRESHOLD = DIAMOND["flag_threshold"]
WEIGHTS = DIAMOND["weights"]
Z_MEANS = DIAMOND["z_means"]
Z_STDS = DIAMOND["z_stds"]
SIGNS = DIAMOND["indicator_signs"]

dscore = np.zeros(len(df))
for f, w in WEIGHTS.items():
    if w == 0:
        continue
    x = df[f].to_numpy(dtype=float)
    z = SIGNS[f] * (x - Z_MEANS[f]) / Z_STDS[f]
    dscore += w * np.nan_to_num(z, nan=0.0)
df["diamond_score"] = dscore
df["is_diamond"] = ((df["pick_filled"] >= CUT_PICK) & (df["diamond_score"] >= FLAG_THRESHOLD)).astype(int)

# ---------------------------------------------------------------------------
# Real outcome percentile (only for resolved players)
# ---------------------------------------------------------------------------
resolved = df["age_22_29_best3"].notna()
df["actual_fantasy_pctl"] = np.nan
df.loc[resolved, "actual_fantasy_pctl"] = to_percentile(
    df.loc[resolved, "age_22_29_best3"].to_numpy(), ref_pool["age_22_29_best3"].to_numpy()
)

# a "hit" (used only for the calibration table below, not shown per-player)
HIT_BAR = CAL["hit_bar_fantasy_ppg"]
df["hit"] = np.where(resolved, (df["age_22_29_best3"] >= HIT_BAR).astype(float), np.nan)

df["low_conf"] = (df["had_combine"] == 0).astype(int)
df["low_conf_reason"] = np.where(df["had_combine"] == 0, "no NBA combine testing on record", "")

# ---------------------------------------------------------------------------
# Decile calibration tables (recomputed here with avg_outcome_pctl included,
# for the dashboard's "how it works" tab)
# ---------------------------------------------------------------------------
def decile_table(pctl_col):
    d = df[ref_mask].copy()
    d["_pctl"] = to_percentile(d[pctl_col].to_numpy(), ref_pool[pctl_col].to_numpy())
    d["_t"] = np.ceil(d["_pctl"] * 10).clip(1, 10).astype(int)
    g = d.groupby("_t").agg(
        n=("hit", "size"), hit_rate=("hit", "mean"), avg_outcome_pctl=("actual_fantasy_pctl", "mean")
    ).reset_index()
    return g.to_dict("records")


pre_decile = decile_table("pre_draft_score")
post_decile = decile_table("post_draft_score")

# ---------------------------------------------------------------------------
# Assemble scored[] rows
# ---------------------------------------------------------------------------
def round_or_none(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


# ---------------------------------------------------------------------------
# Prospect-profile comps: nearest neighbours over model inputs + body type +
# outputs, mirroring WRPI/RUPI's exact similarity methodology (score_v2.py)
# -- weighted z-scored features, Euclidean distance, converted to a 0-100
# similarity via 100*exp(-dist/scale). Body size weighted heavily so a
# 6'2" guard never gets matched to a 6'11" center. This is PROSPECT-PROFILE
# similarity (measurables/production), not playstyle -- the simpler of the
# two Tommy asked for, done first.
df["weight_filled"] = df["WEIGHT"].fillna(df["WEIGHT"].median())
SIM_W = {
    "bpm": 1.0, "porpag": 1.0, "draft_age_filled": 1.0, "breakout_age_filled": 1.0,
    "rec_filled": 0.8, "usg": 0.6,
    "inches": 2.2, "weight_filled": 1.8,
    "pre_draft_score": 0.6, "post_draft_score": 0.6, "diamond_score": 0.6,
}
_M = pd.DataFrame(index=df.index)
for c, wgt in SIM_W.items():
    v = pd.to_numeric(df[c], errors="coerce")
    v = v.fillna(v.median())
    z = (v - v.mean()) / (v.std() + 1e-9)
    _M[c] = z * wgt
_Marr = _M.to_numpy()
players_arr = df["player"].to_numpy()
years_arr = df["real_draft_year"].to_numpy()
fant_arr = df["actual_fantasy_pctl"].to_numpy()
similar_col = []
for i in range(len(_Marr)):
    dist = np.sqrt(((_Marr - _Marr[i]) ** 2).sum(1))
    order = np.argsort(dist)
    top = [j for j in order if j != i][:6]
    scale = np.median(dist[dist > 0]) if (dist > 0).any() else 1.0
    similar_col.append([
        {"p": str(players_arr[j]), "y": int(years_arr[j]) if pd.notna(years_arr[j]) else None,
         "sim": round(float(100 * np.exp(-dist[j] / scale)), 0),
         "fant": None if pd.isna(fant_arr[j]) else round(float(fant_arr[j]), 3)}
        for j in top
    ])
df["similar"] = similar_col

scored = []
for i, row in df.iterrows():
    comp_pre = {k: round_or_none(v[i], 2) for k, v in pre_components.items()}
    comp_pre["intercept"] = round_or_none(pre_intercept, 2)
    comp_post = {k: round_or_none(v[i], 2) for k, v in champion_components.items()}
    comp_post["capital curve"] = round_or_none(champion_capital_contrib[i], 2)
    comp_post["intercept"] = round_or_none(champion_intercept, 2)

    scored.append({
        "Player": row["player"],
        "Year": int(row["real_draft_year"]) if pd.notna(row["real_draft_year"]) else None,
        "pos": row["pos"] if pd.notna(row["pos"]) else None,
        "pick": int(row["pick_filled"]) if row["pick_filled"] < 61 else 999,
        "tier": int(row["tier"]),
        "rpi_post": round_or_none(row["post_draft_pctl"]),
        "rpi_pre": round_or_none(row["pre_draft_pctl"]),
        "raw_post": round_or_none(row["post_draft_score"] / len(df) * 100, 1),
        "raw_pre": round_or_none(row["pre_draft_score"], 2),
        "diamond_score": round_or_none(row["diamond_score"], 3),
        "is_diamond": int(row["is_diamond"]),
        "is_star_pre": int(row["is_star_pre"]),
        "is_star_post": int(row["is_star_post"]),
        "low_conf": int(row["low_conf"]),
        "low_conf_reason": row["low_conf_reason"],
        "actual_fantasy_pctl": round_or_none(row["actual_fantasy_pctl"]),
        "bpm": round_or_none(row["bpm"], 2),
        "fg_pct": round_or_none(row["fg_pct_filled"], 1),
        "porpag": round_or_none(row["porpag"], 2),
        "draft_age": round_or_none(row["draft_age_filled"], 2),
        "breakout_age": round_or_none(row["breakout_age"], 2) if pd.notna(row["breakout_age"]) else None,
        "never_broke_out": int(row["never_broke_out"]),
        "rec": round_or_none(row["rec"], 1) if pd.notna(row["rec"]) else None,
        "exp": row["exp"] if pd.notna(row["exp"]) else None,
        "agility_pctile": round_or_none(row["LANE_AGILITY_TIME_PCTILE"]) if row["had_combine"] == 1 else None,
        "ensemble_components": {
            "champion_raw": round_or_none(row["post_draft_score_champion_raw"], 2),
            "rec_variant_raw": round_or_none(row["post_draft_score_rec_variant_raw"], 2),
            "breakout_variant_raw": round_or_none(row["post_draft_score_breakout_variant_raw"], 2),
        },
        "comp_pre": comp_pre,
        "comp_post": comp_post,
        "similar": row["similar"],
    })

out = {
    "generated": pd.Timestamp.now().isoformat(),
    "draft_status": {"class": 2026, "provisional": False},
    "model": {
        "target": "best-3-of-any-3-consecutive-seasons average fantasy PPG, ages 22-29",
        "post_cv_spearman": 0.5921,
        "pre_cv_spearman": 0.4527,
        "pick_alone_spearman": 0.5809,
        "reference_years": [int(ref_pool["real_draft_year"].min()), int(ref_pool["real_draft_year"].max())],
        "reference_n": len(ref_pool),
        "star_pctl": STAR_PCTL,
        "diamond": {
            "cut_pick": CUT_PICK,
            "lift5": round(DIAMOND["lift"], 2),
            "base_rate": round(DIAMOND["late_pool_base_rate"], 3),
        },
    },
    "calibration": {"post": post_decile, "pre": pre_decile},
    "scored": scored,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out, indent=None, allow_nan=False))
print(f"Wrote {len(scored)} scored prospects to {OUT_PATH}")
print(f"Star (post) count: {df['is_star_post'].sum()}, Diamond count: {df['is_diamond'].sum()}")

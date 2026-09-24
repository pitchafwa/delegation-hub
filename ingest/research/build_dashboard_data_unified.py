"""Build the rookie-prospect dashboard's data file from the UNIFIED
(college + international) model. Mirrors build_dashboard_data.py's
structure; diamond stays college-only (disclosed scope limit -- its
indicator set is heavily college-specific), everything else covers both
pathways.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT.parent.parent / "dashboard" / "rookie_scores.json"

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")

# real, near-universal height/weight/position (covers both pathways; torvik
# "inches" and combine WEIGHT are patchy/college-only or combine-only)
bio = pd.read_csv(ROOT / "data" / "player_bio.csv")


def parse_height(h):
    if not isinstance(h, str) or "-" not in h:
        return np.nan
    try:
        ft, inch = h.split("-")
        return int(ft) * 12 + int(inch)
    except ValueError:
        return np.nan


bio["height_in"] = bio["HEIGHT"].apply(parse_height)
df = df.merge(
    bio[["PERSON_ID", "height_in", "WEIGHT", "POSITION"]].rename(
        columns={"PERSON_ID": "PLAYER_ID", "WEIGHT": "bio_weight", "POSITION": "bio_position"}
    ), on="PLAYER_ID", how="left",
)
# real, universal position fallback -- fills the gap for international rows
# (torvik "pos" is college-only) and any college rows missing it too
df["pos_display"] = df["pos"].fillna(df["bio_position"])

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
POWER_CONFS = {"ACC", "B10", "B12", "BE", "P10", "P12", "SEC", "Amer"}
df["power_conf"] = df["conf"].isin(POWER_CONFS).astype(int) if "conf" in df.columns else 0
df["fg_pct_filled"] = df["fg_pct"].fillna(df["fg_pct"].median())
df["three_pct_filled"] = df["three_pct"].fillna(df["three_pct"].median()) if "three_pct" in df.columns else np.nan
df["had_combine"] = df["LANE_AGILITY_TIME_PCTILE"].notna().astype(int)
for c in ["WINGSPAN_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["breakout_age_filled"] = df["breakout_age"].fillna(df["breakout_age"].median())
df["never_broke_out"] = df["never_broke_out"].fillna(1).astype(int)
for c in ["bpm", "porpag", "usg", "ts", "ortg"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
df["talent_pctile"] = df["talent_pctile"].fillna(df["talent_pctile"].median())

frozen = json.loads((ROOT / "data" / "output_a_model_unified.frozen.json").read_text(encoding="utf-8"))
PRE = frozen["pre_draft_model"]
POST = frozen["post_draft_model"]
STAR = frozen["star_calibration"]

# diamond stays college-only -- load the college-only frozen model just for
# its diamond calibration (a disclosed scope limit, not a bug)
college_frozen = json.loads((ROOT / "data" / "output_a_model.frozen.json").read_text(encoding="utf-8"))
DIAMOND = college_frozen["star_diamond_calibration"]["diamond"]


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


pre_total, pre_components, pre_intercept = score_with_components(PRE["params"], df, PRE["features"])
df["pre_draft_score"] = pre_total

component_scores = {}
component_ranks = []
champion_components = None
champion_intercept = None
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
        champion_capital_contrib = cap_part

post_rank_avg = np.mean(component_ranks, axis=0)
df["post_draft_score"] = post_rank_avg
df["post_draft_score_champion_raw"] = component_scores["champion"]
df["post_draft_score_rec_variant_raw"] = component_scores["rec_variant"]
df["post_draft_score_breakout_variant_raw"] = component_scores["breakout_variant"]

REFERENCE_MAX_YEAR = 2018
ref_mask = (df["real_draft_year"] <= REFERENCE_MAX_YEAR) & df["age_22_29_best3"].notna()
ref_pool = df[ref_mask]
print(f"Reference pool: {len(ref_pool)} players (college: {(ref_pool['data_source']=='college').sum()}, "
      f"international: {(ref_pool['data_source']=='international').sum()}), classes "
      f"{int(ref_pool['real_draft_year'].min())}-{int(ref_pool['real_draft_year'].max())}")


def to_percentile(x, ref):
    ref_sorted = np.sort(ref)
    return np.searchsorted(ref_sorted, x, side="right") / len(ref_sorted)


df["pre_draft_pctl"] = to_percentile(df["pre_draft_score"].to_numpy(), ref_pool["pre_draft_score"].to_numpy())
df["post_draft_pctl"] = to_percentile(df["post_draft_score"].to_numpy(), ref_pool["post_draft_score"].to_numpy())
df["tier"] = np.ceil(df["post_draft_pctl"] * 10).clip(1, 10).astype(int)

STAR_PCTL = STAR["star_pctl"]
df["is_star_pre"] = (df["pre_draft_pctl"] >= STAR_PCTL).astype(int)
df["is_star_post"] = (df["post_draft_pctl"] >= STAR_PCTL).astype(int)

# --- diamond: college rows only ---
CUT_PICK = DIAMOND["chosen_cut_pick"]
FLAG_THRESHOLD = DIAMOND["flag_threshold"]
WEIGHTS = DIAMOND["weights"]
Z_MEANS = DIAMOND["z_means"]
Z_STDS = DIAMOND["z_stds"]
SIGNS = DIAMOND["indicator_signs"]

dscore = np.full(len(df), np.nan)
college_idx = df["data_source"] == "college"
for f, w in WEIGHTS.items():
    if w == 0 or f not in df.columns:
        continue
    x = df.loc[college_idx, f].to_numpy(dtype=float)
    z = SIGNS[f] * (x - Z_MEANS[f]) / Z_STDS[f]
    dscore[college_idx.to_numpy().nonzero()[0]] = np.nan_to_num(dscore[college_idx.to_numpy().nonzero()[0]], nan=0.0) + w * np.nan_to_num(z, nan=0.0)
df["diamond_score"] = dscore
df["is_diamond"] = ((df["data_source"] == "college") & (df["pick_filled"] >= CUT_PICK) & (df["diamond_score"] >= FLAG_THRESHOLD)).astype(int)

resolved = df["age_22_29_best3"].notna()
df["actual_fantasy_pctl"] = np.nan
df.loc[resolved, "actual_fantasy_pctl"] = to_percentile(
    df.loc[resolved, "age_22_29_best3"].to_numpy(), ref_pool["age_22_29_best3"].to_numpy()
)

HIT_BAR = STAR["hit_bar_fantasy_ppg"]
df["hit"] = np.where(resolved, (df["age_22_29_best3"] >= HIT_BAR).astype(float), np.nan)

df["low_conf"] = ((df["had_combine"] == 0) | (df["data_source"] == "international")).astype(int)
df["low_conf_reason"] = np.where(
    df["data_source"] == "international",
    "international prospect -- talent read uses a Game Score proxy (no BPM equivalent exists for non-NCAA leagues), breakout age and position-normalized combine unavailable",
    np.where(df["had_combine"] == 0, "no NBA combine testing on record", ""),
)


def decile_table(pctl_col):
    d = df[ref_mask].copy()
    d["_pctl"] = to_percentile(d[pctl_col].to_numpy(), ref_pool[pctl_col].to_numpy())
    d["_t"] = np.ceil(d["_pctl"] * 10).clip(1, 10).astype(int)
    g = d.groupby("_t").agg(n=("hit", "size"), hit_rate=("hit", "mean"), avg_outcome_pctl=("actual_fantasy_pctl", "mean")).reset_index()
    return g.to_dict("records")


pre_decile = decile_table("pre_draft_score")
post_decile = decile_table("post_draft_score")


def round_or_none(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


# --- similarity comps: talent_pctile stands in for bpm/porpag so both
# pathways can compare on a shared scale ---
df["weight_filled"] = df["WEIGHT"].fillna(df["WEIGHT"].median()) if "WEIGHT" in df.columns else 0
SIM_W = {
    "talent_pctile": 1.4, "draft_age_filled": 1.0, "rec_filled": 0.8,
    "inches": 2.2, "weight_filled": 1.8,
    "pre_draft_score": 0.6, "post_draft_score": 0.6,
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

# ---------------------------------------------------------------------------
# Stylistic comps ("what do they look like, what do they do on the court")
# -- separate from the prospect-profile comps above. Real, disclosed gap:
# no "transition scorer" proxy exists in this data (season aggregates, not
# play-type-tagged) -- left out rather than faked with a weak signal.
# College gets the full, rich version (real shot-location splits, DBPM,
# "stops", usage-adjusted assist rate). International gets a cruder version
# (basic box-score rates only) -- disclosed via is_low_conf_style below.
# ---------------------------------------------------------------------------
raw_intl = pd.read_csv(ROOT / "data" / "international_player_seasons.csv")
for c in ["FGA", "3PA", "2PA"]:
    raw_intl[c] = pd.to_numeric(raw_intl[c], errors="coerce")
raw_intl["_is_final_year"] = raw_intl.groupby("PERSON_ID")["season_end_year"].transform("max") == raw_intl["season_end_year"]
final_intl = raw_intl[raw_intl["_is_final_year"]].sort_values("G", ascending=False).drop_duplicates(subset=["PERSON_ID"], keep="first")
df = df.merge(final_intl[["PERSON_ID", "FGA", "3PA", "2PA"]].rename(columns={"PERSON_ID": "PLAYER_ID"}), on="PLAYER_ID", how="left")

c_mask = df["data_source"] == "college"
i_mask = df["data_source"] == "international"

df["lankiness"] = df["WINGSPAN"] - df["height_in"]
df["three_rate"] = np.nan
df.loc[c_mask, "three_rate"] = df.loc[c_mask, "three_a"] / df.loc[c_mask, "fga"].replace(0, np.nan)
df.loc[i_mask, "three_rate"] = df.loc[i_mask, "3PA"] / df.loc[i_mask, "FGA"].replace(0, np.nan)
df["rim_rate"] = np.nan
df.loc[c_mask, "rim_rate"] = df.loc[c_mask, "rim_a"] / df.loc[c_mask, "fga"].replace(0, np.nan)
df["playmaking"] = np.nan
df.loc[c_mask, "playmaking"] = df.loc[c_mask, "ast"] * (df.loc[c_mask, "usg"] / 20.0)
df.loc[i_mask, "playmaking"] = df.loc[i_mask, "apg"] / df.loc[i_mask, "tov"].replace(0, np.nan)
df["defense_score"] = np.nan
df.loc[c_mask, "defense_score"] = df.loc[c_mask, "dbpm"]
df.loc[i_mask, "defense_score"] = df.loc[i_mask, "spg"].fillna(0) + df.loc[i_mask, "bpg"].fillna(0)
df["rebounding_style"] = np.nan
df.loc[c_mask, "rebounding_style"] = df.loc[c_mask, "oreb_rate"] + df.loc[c_mask, "dreb_rate"]
df.loc[i_mask, "rebounding_style"] = df.loc[i_mask, "rpg"]

STYLE_W = {
    "height_in": 2.0, "bio_weight": 1.6, "lankiness": 1.4,
    "three_rate": 1.6, "rim_rate": 1.3, "playmaking": 1.3, "defense_score": 1.2, "rebounding_style": 0.9,
}
_SM = pd.DataFrame(index=df.index)
for c, wgt in STYLE_W.items():
    v = pd.to_numeric(df[c], errors="coerce")
    v = v.fillna(v.median())
    z = (v - v.mean()) / (v.std() + 1e-9)
    _SM[c] = z * wgt
_SMarr = _SM.to_numpy()
sources_arr = df["data_source"].to_numpy()
style_col = []
for i in range(len(_SMarr)):
    dist = np.sqrt(((_SMarr - _SMarr[i]) ** 2).sum(1))
    order = np.argsort(dist)
    top = [j for j in order if j != i][:6]
    scale = np.median(dist[dist > 0]) if (dist > 0).any() else 1.0
    style_col.append([
        {"p": str(players_arr[j]), "y": int(years_arr[j]) if pd.notna(years_arr[j]) else None,
         "sim": round(float(100 * np.exp(-dist[j] / scale)), 0),
         "intl": bool(sources_arr[j] == "international")}
        for j in top
    ])
df["style_similar"] = style_col
df["style_low_conf"] = (df["data_source"] == "international").astype(int)

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
        "pos": row["pos_display"] if pd.notna(row["pos_display"]) else None,
        "pick": int(row["pick_filled"]) if row["pick_filled"] < 61 else 999,
        "tier": int(row["tier"]),
        "data_source": row["data_source"],
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
        "talent_pctile": round_or_none(row["talent_pctile"], 3),
        "bpm": round_or_none(row["bpm"], 2) if row["data_source"] == "college" else None,
        "game_score": round_or_none(row.get("game_score"), 2) if row["data_source"] == "international" else None,
        "draft_age": round_or_none(row["draft_age_filled"], 2),
        "breakout_age": round_or_none(row["breakout_age"], 2) if pd.notna(row["breakout_age"]) else None,
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
        "style_similar": row["style_similar"],
        "style_low_conf": int(row["style_low_conf"]),
    })

out = {
    "generated": pd.Timestamp.now().isoformat(),
    "draft_status": {"class": 2026, "provisional": False},
    "model": {
        "target": "best-3-of-any-3-consecutive-seasons average fantasy PPG, ages 22-29",
        "post_cv_spearman": POST["loco_cv_rho_mean_across_5_seeds"],
        "pre_cv_spearman": PRE["loco_cv_rho"],
        "pick_alone_spearman": POST["baseline_draft_capital_alone_loco_cv_rho"],
        "reference_years": [int(ref_pool["real_draft_year"].min()), int(ref_pool["real_draft_year"].max())],
        "reference_n": len(ref_pool),
        "star_pctl": STAR_PCTL,
        "diamond": {"cut_pick": CUT_PICK, "lift5": round(DIAMOND["lift"], 2), "base_rate": round(DIAMOND["late_pool_base_rate"], 3),
                    "note": "college-only -- diamond's indicator set wasn't adapted for international prospects in this pass"},
        "international_note": "121 real international/non-college prospects included (out of 179 real gap-population "
            "draftees 2008-2026 with no US college record); talent read uses a percentile-ranked Hollinger Game "
            "Score in place of BPM (no BPM-equivalent exists for non-NCAA leagues).",
    },
    "calibration": {"post": post_decile, "pre": pre_decile},
    "scored": scored,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out, indent=None, allow_nan=False), encoding="utf-8")
print(f"Wrote {len(scored)} scored prospects to {OUT_PATH}")
print(f"College: {(df['data_source']=='college').sum()}, International: {(df['data_source']=='international').sum()}")
print(f"Star (post) count: {df['is_star_post'].sum()}, Diamond count: {df['is_diamond'].sum()}")

for name in ["Victor Wembanyama", "Luka Dončić"]:
    row = df[df["player"] == name]
    if row.empty:
        print(f"\n{name}: NOT FOUND")
    else:
        r = row.iloc[0]
        print(f"\n{name}: tier={r['tier']}, post_pctl={r['post_draft_pctl']:.3f}, pre_pctl={r['pre_draft_pctl']:.3f}, "
              f"is_star_post={r['is_star_post']}, is_star_pre={r['is_star_pre']}, pick={r['pick_filled']}")

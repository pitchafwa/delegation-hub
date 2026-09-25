"""Team drafting profiles from six years of this league's ESPN drafts (2020-21 .. 2025-26) -> dashboard/team_profiles.json.

What ESPN gives us per pick: overall pick, round, team, player, and whether he was a KEEPER. From that plus NBA data we describe how each team builds:
  * KEEPERS  : how many they keep, how old, how productive the year before, how often rookies/young players, and whether they keep the same guys year to year
  * DRAFT    : age of the players drafted, share of rookies (and how early), REACH vs production (took a player earlier than his previous-season fantasy points say
               he should go, negative = reach), positions (guards vs bigs) early and overall, injury gambles (missed a lot of last season)
  * OWNERSHIP: teams that changed hands (ESPN owner ids differ by season) are flagged; their profile uses only the CURRENT owner's seasons and says how many.
Baseline for "reach": among the players actually drafted (non-keepers) each year, rank by previous-season league-scored TOTAL points (or by ESPN ADP when it exists for
that year) and compare with the order picks were really made. League-scored points = PTS + 1.5 REB + 2 AST + 3 STL + 3 BLK + 3PM + 2 FTM - FTA - TOV.
Nothing here is published about owners as people: no names, only team identity, owner-change flags and tendencies.
Run from ingest/:  uv run python research/build_team_profiles.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
HUB = R.parent.parent / "dashboard"
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]     # ESPN seasonId = year the season ends (2026 = the 2025-26 season, drafted fall 2025)
N_TEAMS = 12


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


# ---------------- NBA data
logs = pd.read_csv(R / "data" / "game_logs_unified.csv", usecols=["PLAYER_ID", "PLAYER_NAME", "SEASON", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
logs["fp"] = logs.PTS + 1.5 * logs.REB + 2 * logs.AST + 3 * logs.STL + 3 * logs.BLK + logs.FG3M + 2 * logs.FTM - logs.FTA - logs.TOV + 3 * logs.TD3
logs["end"] = logs.SEASON.str[:4].astype(int) + 1
logs["key"] = logs.PLAYER_NAME.map(norm)
ss = logs.groupby(["key", "end"]).agg(gp=("fp", "size"), fpg=("fp", "mean"), tot=("fp", "sum"), pid=("PLAYER_ID", "first")).reset_index()
stats = {(r.key, r.end): (r.gp, r.fpg, r.tot) for r in ss.itertuples()}
first_end = ss.groupby("key").end.min().to_dict()
pid_by_key = ss.groupby("key").pid.first().to_dict()
bd = pd.read_csv(R / "data" / "birthdates_all.csv")
bd["BIRTHDATE"] = pd.to_datetime(bd["BIRTHDATE"], errors="coerce")
born = bd.dropna(subset=["BIRTHDATE"]).drop_duplicates("PERSON_ID").set_index("PERSON_ID")["BIRTHDATE"]
pos_csv = pd.read_csv(R / "data" / "espn_positions.csv")
pos_by = {norm(r.espn_name): str(r.espn_position) for r in pos_csv.itertuples()}
adp = pd.read_csv(R / "data" / "espn_adp.csv")
adp["key"] = adp["name"].map(norm)
adp_by = {(r.key, int(r.season_id)): r.adp for r in adp.dropna(subset=["adp"]).itertuples()}


def pos_group(p):
    toks = [t for t in re.split(r"[/,\s]+", p or "") if t in ("PG", "SG", "SF", "PF", "C", "G", "F")]
    if not toks:
        return None
    g = sum(t in ("PG", "SG", "G") for t in toks) / len(toks)
    b = sum(t in ("PF", "C") for t in toks) / len(toks)
    return "G" if g >= 0.5 else ("B" if b >= 0.5 else "W")


# ---------------- ESPN drafts + owners
rows, owners = [], {}
for yr in YEARS:
    lg = League(league_id=config.LEAGUE_ID, year=yr, espn_s2=config.ESPN_S2, swid=config.SWID)
    for t in lg.teams:
        owners[(t.team_id, yr)] = tuple(sorted(o.get("id", "") for o in (t.owners or [])))
    for p in lg.draft:
        rows.append(dict(year=yr, team=p.team.team_id, abbrev=p.team.team_abbrev, rnd=p.round_num, n=(p.round_num - 1) * N_TEAMS + p.round_pick, keeper=bool(p.keeper_status), name=p.playerName))
d = pd.DataFrame(rows)
d["key"] = d.name.map(norm)
d["owners"] = [owners.get((t, y), ()) for t, y in zip(d.team, d.year)]
lg27 = League(league_id=config.LEAGUE_ID, year=2027, espn_s2=config.ESPN_S2, swid=config.SWID)
current = {t.team_id: {"abbrev": t.team_abbrev, "owners": set(o.get("id", "") for o in (t.owners or []))} for t in lg27.teams}


def feats(r):
    prior = stats.get((r.key, r.year - 1))
    pid = pid_by_key.get(r.key)
    age = None
    if pid in born.index:
        age = (pd.Timestamp(year=r.year - 1, month=10, day=20) - born[pid]).days / 365.25
    rookie = first_end.get(r.key) == r.year
    return pd.Series({"prior_gp": prior[0] if prior else np.nan, "prior_fpg": prior[1] if prior else np.nan, "prior_tot": prior[2] if prior else np.nan,
                      "age": age, "rookie": rookie, "adp": adp_by.get((r.key, r.year), np.nan), "pg": pos_group(pos_by.get(r.key))})


d = pd.concat([d, d.apply(feats, axis=1)], axis=1)
print(f"{len(d)} picks; matched prior-season stats for {d.prior_tot.notna().mean():.0%}, age for {d.age.notna().mean():.0%}, positions for {d.pg.notna().mean():.0%}")

# reach: among each year's DRAFTED (non-keeper) picks, expected order = ADP if available (else previous-season total points); actual order = pick number
dr = d[~d.keeper].copy()
dr["rel"] = dr.groupby("year").n.rank(method="first")            # pick number among the REAL (non-keeper) picks of that draft
dr["rrnd"] = np.ceil(dr.rel / N_TEAMS)                           # real round (keepers occupy the first board rounds)
dr["reach"] = np.nan
for yr, g in dr.groupby("year"):
    g = g.sort_values("n")
    val = g.adp.where(g.adp.notna(), np.nan)
    have = g[g.prior_tot.notna() | g.adp.notna()].copy()
    # expected rank: ADP when known, else rank by total points (higher = earlier), scaled to the same 1..N range
    by_tot = have.prior_tot.rank(ascending=False, method="first")
    have["exp"] = np.where(have.adp.notna(), have.adp.rank(method="first"), by_tot)
    have["exp"] = have.exp.rank(method="first")
    have["act"] = have.n.rank(method="first")
    dr.loc[have.index, "reach"] = have.act - have.exp        # negative = drafted earlier than the baseline order says (a reach)

# ---------------- per-team profile
teams = sorted(current)
abbr = {t: current[t]["abbrev"] for t in teams}
cur_owner_years = {}
for t in teams:      # identity = ESPN owner id (owners change hands and can move between team slots), not the team id
    mine_ = d[d.owners.map(lambda o: bool(set(o) & current[t]["owners"]))]
    cur_owner_years[t] = sorted(mine_.year.unique().tolist())
    others = d[(d.team == t) & ~d.year.isin(cur_owner_years[t])].year.nunique()
    print(abbr[t], "seasons drafted by the CURRENT owner:", cur_owner_years[t], f"| other owners held this team in {others} seasons")


def summarize(sub_dr, sub_kp):
    m = {}
    top5 = sub_dr[sub_dr.n <= 60]                  # each team's first ~5 real picks are inside the first 60 overall picks in a 12-team snake
    m["picks"] = len(sub_dr)
    m["age_mean"] = sub_dr.age.mean()
    m["age_early"] = sub_dr.sort_values("n").groupby("year").head(4).age.mean()
    m["young_share"] = (sub_dr.age <= 23).mean() if sub_dr.age.notna().any() else np.nan
    m["rookie_share"] = sub_dr.rookie.mean()
    m["rookie_avg_round"] = sub_dr[sub_dr.rookie].rrnd.mean() if sub_dr.rookie.any() else np.nan
    m["reach_mean"] = sub_dr.reach.mean()
    m["reach_early"] = sub_dr[sub_dr.rrnd <= 4].reach.mean()
    m["guard_share"] = (sub_dr.pg == "G").sum() / max(sub_dr.pg.notna().sum(), 1)
    m["big_share"] = (sub_dr.pg == "B").sum() / max(sub_dr.pg.notna().sum(), 1)
    m["big_early"] = (sub_dr[sub_dr.rrnd <= 4].pg == "B").sum() / max(sub_dr[sub_dr.rrnd <= 4].pg.notna().sum(), 1)
    m["injury_gamble"] = ((sub_dr.prior_gp < 45) & (sub_dr.prior_fpg > 28)).sum() / max(len(sub_dr), 1)
    m["keepers_per_year"] = len(sub_kp) / max(sub_kp.year.nunique(), 1) if len(sub_kp) else 0
    m["keeper_age"] = sub_kp.age.mean()
    m["keeper_young_share"] = (sub_kp.age <= 23).mean() if sub_kp.age.notna().any() else np.nan
    m["keeper_rookie_share"] = sub_kp.rookie.mean() if len(sub_kp) else np.nan
    m["keeper_prod"] = sub_kp.prior_tot.rank(pct=True).mean() if False else sub_kp.prior_fpg.mean()
    # loyalty: share of a year's keepers who were also kept the year before
    kk = sub_kp.groupby("year").key.apply(set).to_dict()
    both = tot = 0
    for y, ks in kk.items():
        if y - 1 in kk:
            tot += len(ks)
            both += len(ks & kk[y - 1])
    m["keeper_repeat"] = both / tot if tot else np.nan
    return m


all_dr, all_kp = dr, d[d.keeper]
league = summarize(all_dr, all_kp)
league["keepers_per_year"] = len(all_kp) / (len(YEARS) * N_TEAMS)
prof, rows_out = {}, []
for t in teams:
    yrs = cur_owner_years[t]
    own = current[t]["owners"]
    sd = all_dr[all_dr.owners.map(lambda o: bool(set(o) & own))]
    sk = all_kp[all_kp.owners.map(lambda o: bool(set(o) & own))]
    prof[t] = summarize(sd, sk)
    prof[t]["seasons"] = yrs
    prof[t]["owner_changed"] = bool(len(d[(d.team == t) & ~d.year.isin(yrs)]))
metrics = ["age_mean", "age_early", "young_share", "rookie_share", "reach_mean", "reach_early", "guard_share", "big_share", "big_early", "injury_gamble", "keeper_age", "keeper_young_share", "keeper_repeat", "keeper_prod"]
dist = {k: pd.Series([prof[t][k] for t in teams], dtype=float) for k in metrics}
z = {t: {k: (prof[t][k] - dist[k].mean()) / (dist[k].std() or 1) if pd.notna(prof[t][k]) else 0.0 for k in metrics} for t in teams}

LABELS = {
    "age_mean": ("Drafts younger players", "Drafts older, established players"),
    "age_early": ("Goes young with early picks", "Spends early picks on veterans"),
    "rookie_share": ("Loves rookies", "Rarely drafts rookies"),
    "reach_early": ("Reaches early (takes players before production says)", "Takes the best producers on the board early"),
    "guard_share": ("Guard-heavy", "Light on guards"),
    "big_share": ("Big-man heavy", "Light on bigs"),
    "big_early": ("Bigs with the early picks", "Guards/wings with the early picks"),
    "injury_gamble": ("Gambles on injured players", "Avoids injury risk"),
    "keeper_age": ("Keeps young players", "Keeps veterans"),
    "keeper_repeat": ("Loyal: keeps the same players year to year", "Churns keepers"),
    "keeper_prod": ("Keeps high producers", "Keeps lower producers (upside/prospects)"),
}
SIGN = {"age_mean": -1, "age_early": -1, "rookie_share": 1, "reach_early": -1, "guard_share": 1, "big_share": 1, "big_early": 1, "injury_gamble": 1, "keeper_age": -1, "keeper_repeat": 1, "keeper_prod": 1}
out = []
for t in teams:
    tags = []
    for k, (hi, lo) in LABELS.items():
        zz = z[t][k] * SIGN[k]
        if abs(zz) >= 0.9:
            tags.append({"key": k, "label": hi if zz > 0 else lo, "z": round(float(zz), 2)})
    tags.sort(key=lambda x: -abs(x["z"]))
    seasons_n = len(prof[t]["seasons"])
    if prof[t]["picks"] < 8:
        tags = []                              # too little history for a tendency (new owners, or only one season)
    own = current[t]["owners"]
    detail = []
    for y in prof[t]["seasons"]:
        yd = d[(d.year == y) & d.owners.map(lambda o: bool(set(o) & own))].sort_values("n")
        detail.append({"year": int(y), "keepers": [{"name": r.name, "age": None if pd.isna(r.age) else round(float(r.age), 1)} for r in yd[yd.keeper].itertuples()],
                       "early": [{"name": r.name, "pick": int(dr.loc[r.Index, "rel"]) if r.Index in dr.index else None, "age": None if pd.isna(r.age) else round(float(r.age), 1)}
                                 for r in yd[~yd.keeper].head(4).itertuples()]})
    out.append({"id": int(t), "by_season": detail, "abbrev": abbr.get(t), "seasons": seasons_n, "years": [int(y) for y in prof[t]["seasons"]], "owner_changed": bool(prof[t]["owner_changed"]),
                "confidence": "none" if seasons_n == 0 else "low" if seasons_n <= 2 else ("medium" if seasons_n <= 4 else "high"),
                "metrics": {k: (None if pd.isna(prof[t][k]) else round(float(prof[t][k]), 3)) for k in metrics + ["picks", "keepers_per_year", "rookie_avg_round"]},
                "tags": tags[:5]})
league_out = {k: (None if pd.isna(league[k]) else round(float(league[k]), 3)) for k in metrics + ["keepers_per_year", "rookie_avg_round"]}
(HUB / "team_profiles.json").write_text(json.dumps({"years": YEARS, "league": league_out, "teams": out}, separators=(",", ":")), encoding="utf-8")
d.to_csv(R / "data" / "draft_history_2021_2026.csv", index=False)
print("\nleague:", {k: round(v, 2) if v is not None else None for k, v in league_out.items()})
for o in out:
    print(f"\n{o['abbrev']} ({o['seasons']} seasons with current owner{', OWNER CHANGED' if o['owner_changed'] else ''}; confidence {o['confidence']})")
    for tg in o["tags"]:
        print(f"   {tg['label']}  (z {tg['z']:+.1f})")
    m = o["metrics"]
    f = lambda v, fmt: "n/a" if v is None else format(v, fmt)
    print(f"   picks {m['picks']} | age {f(m['age_mean'],'.1f')} | rookies {f(m['rookie_share'],'.0%')} | reach(early) {f(m['reach_early'],'+.1f')} | G/B {f(m['guard_share'],'.0%')}/{f(m['big_share'],'.0%')} | keepers/yr {f(m['keepers_per_year'],'.1f')}, keeper age {f(m['keeper_age'],'.1f')}, repeat {f(m['keeper_repeat'],'.2f')}")

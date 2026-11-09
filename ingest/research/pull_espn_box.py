"""Box scores from ESPN instead of stats.nba.com (laptop-free test, 2026-11-01, Tommy): stats.nba.com blocks GitHub's servers, ESPN does not.
Writes the same two files the model chain reads from the NBA site, in the same columns:
  nba_api_<season>.csv  (pull_recent_gamelogs.py: one row per player who played)
  box_v3_<season>.csv   (pull_box_v3.py: every listed player, with the did-not-play comment)
Ids: players are mapped ESPN -> NBA with espn_id_map.json (+ the crosswalk and earlier game logs by name); a player still unmapped gets
and is counted (a stable negative id from his name). Games get "0" + the ESPN event id (10 digits, never clashes with NBA's 0022...). One cached summary per finished game in
data/espn_box_cache/, pulled again until REPULL_DAYS after the game: official stat corrections reach ESPN after the morning-after pull (2026-11-09: 22 of 316 player lines from 11-04).
Names: a player the NBA files already know keeps the NBA's spelling, as on the laptop. ESPN writes Nikola Vucevic, last season's logs Nikola Vučević; build_redraft.py then took
his "current" DELCO and minutes from last season's rows under the other spelling (found 2026-11-09).
Run from ingest/:  python research/pull_espn_box.py [season] [--out DIR] [--prefix espn_]
  default writes data/game_logs/espn_nba_api_<season>.csv and espn_box_v3_<season>.csv (side by side with the NBA files, for the comparison);
  --prefix "" writes the real file names (the cloud run).
"""
import json
import re
import zlib
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent))
from team_abbr import canon
from season import current_season_str
import season_calendar as SCAL

args = sys.argv[1:]
OUT = Path(args[args.index("--out") + 1]) if "--out" in args else R / "data" / "game_logs"
PREFIX = args[args.index("--prefix") + 1] if "--prefix" in args else "espn_"
season = next((a for a in args if re.fullmatch(r"\d{4}-\d{2}", a)), current_season_str())
CACHE = R / "data" / "espn_box_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
ET = ZoneInfo("America/New_York")
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
TEAMS = {"ATL": (1610612737, "Atlanta Hawks"), "BOS": (1610612738, "Boston Celtics"), "BKN": (1610612751, "Brooklyn Nets"), "CHA": (1610612766, "Charlotte Hornets"),
         "CHI": (1610612741, "Chicago Bulls"), "CLE": (1610612739, "Cleveland Cavaliers"), "DAL": (1610612742, "Dallas Mavericks"), "DEN": (1610612743, "Denver Nuggets"),
         "DET": (1610612765, "Detroit Pistons"), "GSW": (1610612744, "Golden State Warriors"), "HOU": (1610612745, "Houston Rockets"), "IND": (1610612754, "Indiana Pacers"),
         "LAC": (1610612746, "LA Clippers"), "LAL": (1610612747, "Los Angeles Lakers"), "MEM": (1610612763, "Memphis Grizzlies"), "MIA": (1610612748, "Miami Heat"),
         "MIL": (1610612749, "Milwaukee Bucks"), "MIN": (1610612750, "Minnesota Timberwolves"), "NOP": (1610612740, "New Orleans Pelicans"), "NYK": (1610612752, "New York Knicks"),
         "OKC": (1610612760, "Oklahoma City Thunder"), "ORL": (1610612753, "Orlando Magic"), "PHI": (1610612755, "Philadelphia 76ers"), "PHX": (1610612756, "Phoenix Suns"),
         "POR": (1610612757, "Portland Trail Blazers"), "SAC": (1610612758, "Sacramento Kings"), "SAS": (1610612759, "San Antonio Spurs"), "TOR": (1610612761, "Toronto Raptors"),
         "UTA": (1610612762, "Utah Jazz"), "WAS": (1610612764, "Washington Wizards")}
# ESPN's did-not-play reasons -> the NBA box score's comment (build_rotation_status.py reads the words Injury/Illness/Not With Team/Suspen/Personal)
REASON = [("COACH", "DNP - Coach's Decision"), ("G LEAGUE", "Inactive - G League - Two-Way"), ("TWO-WAY", "Inactive - G League - Two-Way"),
          ("NOT WITH TEAM", "Inactive - Not With Team"), ("SUSPEN", "Inactive - Suspension"), ("PERSONAL", "Inactive - Personal Reasons"),
          ("ILLNESS", "Inactive - Injury/Illness"), ("INJUR", "Inactive - Injury/Illness")]


def fold(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


def get(url, **p):
    for i in range(4):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.ok:
                return r.json()
        except Exception:
            pass
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"ESPN did not answer: {url} {p}")


# ESPN -> NBA player ids
e2n = {int(k): int(v) for k, v in json.load(open(R / "espn_id_map.json")).items()}
LEARNED = R / "espn_id_map_learned.json"           # new players' NBA ids, learned on the laptop from the NBA site (--learn) while it still runs
_learned = json.load(open(LEARNED)) if LEARNED.exists() else {}
e2n.update({int(k): int(v) for k, v in _learned.items() if k.isdigit()})
cw = R / "data" / "espn_nba_id_crosswalk.csv"
if cw.exists():
    x = pd.read_csv(cw, usecols=["NBAID", "ESPNID"], encoding="latin-1").dropna()
    for e, n in zip(x.ESPNID, x.NBAID):
        e2n.setdefault(int(e), int(n))
REPULL_DAYS = 3
nba_spelling = {}                              # NBA id -> the name in the earlier game logs
by_name = {}                                   # last resort for new players: NBA ids by folded name from the earlier game logs
if cw.exists():
    x = pd.read_csv(cw, usecols=["NBAName", "NBAID"], encoding="latin-1").dropna()
    by_name.update({fold(n): int(p) for p, n in zip(x.NBAID, x.NBAName)})
for f in sorted((R / "data" / "game_logs").glob("box_v3_*.csv")):
    if season in f.name:                       # this season's file is what we are building (on the cloud it is ESPN's own)
        continue                                                              # earlier seasons' box scores include inactive two-way players who never got on the floor
    g = pd.read_csv(f, usecols=["personId", "firstName", "familyName"]).drop_duplicates()
    by_name.update({fold(f"{a} {b}"): int(p) for p, a, b in zip(g.personId, g.firstName, g.familyName)})
for f in sorted((R / "data" / "game_logs").glob("nba_api_*.csv")):
    if season in f.name:
        continue
    g = pd.read_csv(f, usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
    by_name.update({fold(n): int(p) for p, n in zip(g.PLAYER_ID, g.PLAYER_NAME)})
    nba_spelling.update({int(p): n for p, n in zip(g.PLAYER_ID, g.PLAYER_NAME)})
by_name.update({k[2:]: int(v) for k, v in _learned.items() if k.startswith("n:")})     # learned names (inactive players ESPN never lists)


def name_id(nm):
    """A player nobody has an NBA id for yet: a stable negative id from his name, so ESPN's box score and the injury report agree."""
    return -(zlib.crc32(fold(nm).encode()) % 10 ** 9)

start = SCAL.opener()
today_et = datetime.now(ET).date()
events = []                                    # (date, event id)
d = start
while d <= today_et:
    j = get(SB, dates=d.strftime("%Y%m%d"))
    for ev in j.get("events", []):
        st = ev["competitions"][0]["status"]["type"]
        if st.get("completed") and ev.get("season", {}).get("type", 2) == 2:
            events.append((d, ev["id"]))
    d += timedelta(days=1)
print(f"{season}: {len(events)} finished regular-season games on ESPN since {start}", flush=True)

_rost = {}


def roster(team_id):
    """ESPN's current roster for one team: [(espn id, name)]. Players on it who are in neither the box score nor the injury report were inactive
    by the coach's decision (the NBA box score's "Inactive - Coach's Decision")."""
    if team_id not in _rost:
        _rost[team_id] = [(int(a["id"]), a["displayName"]) for a in get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster").get("athletes", [])]
    return _rost[team_id]


rows, box, unmapped, rosters = [], [], {}, {}
for gd, eid in events:
    cp = CACHE / f"{eid}.json"
    j = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else None
    if j is None or j.get("fetched", "") < (gd + timedelta(days=REPULL_DAYS)).isoformat():
        try:
            f = get(SUM, event=eid)
            keep = {"boxscore": {"players": f["boxscore"]["players"]}, "header": {"competitions": [{"competitors": f["header"]["competitions"][0]["competitors"]}]}, "fetched": today_et.isoformat()}
            keep["rosters"] = j["rosters"] if j and "rosters" in j else {c["team"]["abbreviation"]: roster(c["team"]["id"]) for c in keep["header"]["competitions"][0]["competitors"]}   # as of the morning after
            cp.write_text(json.dumps(keep), encoding="utf-8")
            j = keep
            time.sleep(0.25)
        except (RuntimeError, KeyError):
            if j is None:                      # no copy at all; a copy we already hold is still usable
                raise
    if "rosters" not in j:                     # summaries cached before rosters were kept (2026-11-01 backfill): today's roster stands in
        j["rosters"] = {c["team"]["abbreviation"]: roster(c["team"]["id"]) for c in j["header"]["competitions"][0]["competitors"]}
        cp.write_text(json.dumps(j), encoding="utf-8")
    comp = j["header"]["competitions"][0]["competitors"]
    side = {canon(c["team"]["abbreviation"]): c for c in comp}
    home = next(canon(c["team"]["abbreviation"]) for c in comp if c["homeAway"] == "home")
    away = next(canon(c["team"]["abbreviation"]) for c in comp if c["homeAway"] == "away")
    gid = "0" + str(eid)
    for ab, lst in j["rosters"].items():
        rosters[(gd.isoformat(), canon(ab))] = lst
    for tb in j["boxscore"]["players"]:
        tm = canon(tb["team"]["abbreviation"])
        tid, tname = TEAMS[tm]
        opp = away if tm == home else home
        matchup = f"{tm} vs. {opp}" if tm == home else f"{tm} @ {opp}"
        wl = "W" if side[tm].get("winner") else "L"
        st = tb["statistics"][0]
        keys = st["keys"]
        for a in st["athletes"]:
            ath = a["athlete"]
            eid_p = int(ath["id"])
            nm = ath["displayName"]
            pid = e2n.get(eid_p) or by_name.get(fold(nm))
            if not pid:
                unmapped[eid_p] = nm
                pid = name_id(nm)
            first, _, last = nm.partition(" ")
            base = dict(gameId=gid, teamId=tid, teamTricode=tm, personId=pid, firstName=first, familyName=last, position=a.get("position", {}).get("abbreviation", "") if a.get("starter") else "",
                        game_date=gd.isoformat(), season_year=season)
            if a.get("didNotPlay") or not a.get("stats"):
                why = (a.get("reason") or "").upper()
                cm = next((v for k, v in REASON if k in why), "DNP - " + (a.get("reason") or "").title()) if why else "DNP"
                box.append(dict(base, comment=cm, minutes=None))
                continue
            s = dict(zip(keys, a["stats"]))
            mk = lambda k: tuple(int(v) for v in s[k].split("-")) if "-" in s.get(k, "") else (0, 0)
            fgm, fga = mk("fieldGoalsMade-fieldGoalsAttempted")
            tpm, tpa = mk("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
            ftm, fta = mk("freeThrowsMade-freeThrowsAttempted")
            num = lambda k: int(str(s.get(k, "0")).replace("+", "") or 0) if str(s.get(k, "0")).lstrip("+-").isdigit() else 0
            mn, pts, reb, ast, stl, blk, tov, oreb, dreb, pf, pm = (num(k) for k in ("minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers",
                                                                                         "offensiveRebounds", "defensiveRebounds", "fouls", "plusMinus"))
            rows.append(dict(SEASON_ID=int("2" + season[:4]), PLAYER_ID=pid, PLAYER_NAME=nba_spelling.get(pid, nm), TEAM_ID=tid, TEAM_ABBREVIATION=tm, TEAM_NAME=tname, GAME_ID=gid,
                             GAME_DATE=gd.isoformat(), MATCHUP=matchup, WL=wl, MIN=mn, FGM=fgm, FGA=fga, FG_PCT=round(fgm / fga, 3) if fga else None, FG3M=tpm, FG3A=tpa,
                             FG3_PCT=round(tpm / tpa, 3) if tpa else None, FTM=ftm, FTA=fta, FT_PCT=round(ftm / fta, 3) if fta else None, OREB=oreb, DREB=dreb, REB=reb,
                             AST=ast, STL=stl, BLK=blk, TOV=tov, PF=pf, PTS=pts, PLUS_MINUS=pm, FANTASY_PTS=None, VIDEO_AVAILABLE=0))
            box.append(dict(base, comment="", minutes=f"{mn}:00", fieldGoalsMade=fgm, fieldGoalsAttempted=fga, threePointersMade=tpm, threePointersAttempted=tpa,
                            freeThrowsMade=ftm, freeThrowsAttempted=fta, reboundsOffensive=oreb, reboundsDefensive=dreb, reboundsTotal=reb, assists=ast, steals=stl,
                            blocks=blk, turnovers=tov, foulsPersonal=pf, points=pts, plusMinusPoints=pm))

if "--learn" in args and unmapped:
    cur = {}
    for f in (R / "data" / "game_logs").glob(f"*_{season}.csv"):
        if f.name.startswith("espn_"):
            continue
        g = pd.read_csv(f)
        if "PLAYER_ID" in g:
            cur.update({fold(n): int(p) for p, n in zip(g.PLAYER_ID, g.PLAYER_NAME)})
        elif "personId" in g:
            cur.update({fold(f"{a} {b}"): int(p) for p, a, b in zip(g.personId, g.firstName, g.familyName)})
    got = {e: cur[fold(n)] for e, n in unmapped.items() if fold(n) in cur}
    if got:
        old = json.load(open(LEARNED)) if LEARNED.exists() else {}
        old.update({str(k): v for k, v in got.items()})
        LEARNED.write_text(json.dumps(old, indent=0, sort_keys=True), encoding="utf-8")
        print(f"learned {len(got)} new NBA ids from this season's NBA files; rerun to use them")
if not rows:
    print("no finished games yet; nothing written")
    sys.exit(0)
# ESPN's box score leaves out the inactive list (injured, two-way / G League, not with team). The NBA box score's inactive comments come from the
# official injury report, so add every player that report had out (or doubtful/questionable) for the game who is not in ESPN's box score.
added, unnamed = 0, set()
bx = pd.DataFrame(box)
listed = {(g, t): set(s.personId) for (g, t), s in bx.groupby(["game_date", "teamTricode"])}
gid_of = {(g, t): i for g, t, i in zip(bx.game_date, bx.teamTricode, bx.gameId)}
ir = R / "data" / "injury_reports_hourly.csv"
if ir.exists():
    rep = pd.read_csv(ir, usecols=["report_date", "slot", "game_date", "team", "player_key", "player_name", "status", "reason"])
    rep = rep[rep.game_date >= start.isoformat()].sort_values(["report_date", "slot"]).drop_duplicates(["game_date", "team", "player_key"], keep="last")
    rep = rep[rep.status.isin(["Out", "Doubtful", "Questionable"])]
    full2abbr = {n: a for a, (_, n) in TEAMS.items()} | {"Los Angeles Clippers": "LAC", "LA Lakers": "LAL"}
    for r in rep.itertuples():
        tm = full2abbr.get(r.team)
        if not tm or (r.game_date, tm) not in listed:
            continue
        pid = by_name.get(fold(r.player_key))
        if not pid:
            last, _, first = str(r.player_name).partition(", ")
            if re.match(r"(Jr|Sr|II|III|IV)\.?,", first):
                first = first.split(", ", 1)[-1]
            pid = by_name.get(fold(f"{first} {last}"))
            if not pid:
                unnamed.add(fold(f"{first} {last}"))
                pid = name_id(f"{first} {last}")
        if pid in listed[(r.game_date, tm)]:
            continue
        why = str(r.reason)
        cm = ("Inactive - G League - Two-Way" if "G League" in why else "Inactive - Not With Team" if "Not With Team" in why else
              "Inactive - Suspension" if "Suspen" in why else "Inactive - Personal Reasons" if "Personal" in why else "Inactive - Injury/Illness")
        last, _, first = str(r.player_name).partition(", ")
        box.append(dict(gameId=gid_of[(r.game_date, tm)], teamId=TEAMS[tm][0], teamTricode=tm, personId=pid, firstName=first, familyName=last, position="",
                        game_date=r.game_date, season_year=season, comment=cm, minutes=None))
        listed[(r.game_date, tm)].add(pid)
        added += 1
coach = 0
for (gdt, tm), lst in rosters.items():
    if (gdt, tm) not in gid_of:
        continue
    for e, nm in lst:
        pid = e2n.get(e) or by_name.get(fold(nm)) or name_id(nm)
        if pid in listed[(gdt, tm)]:
            continue
        first, _, last = nm.partition(" ")
        box.append(dict(gameId=gid_of[(gdt, tm)], teamId=TEAMS[tm][0], teamTricode=tm, personId=pid, firstName=first, familyName=last, position="",
                        game_date=gdt, season_year=season, comment="Inactive - Coach's Decision", minutes=None))
        listed[(gdt, tm)].add(pid)
        coach += 1
if "--learn" in args and unnamed:
    cur = {}
    for f in (R / "data" / "game_logs").glob(f"box_v3_{season}.csv"):
        g = pd.read_csv(f, usecols=["personId", "firstName", "familyName"]).drop_duplicates()
        cur.update({fold(f"{a} {b}"): int(p) for p, a, b in zip(g.personId, g.firstName, g.familyName)})
    got = {f"n:{n}": cur[n] for n in unnamed if n in cur}
    if got:
        old = json.load(open(LEARNED)) if LEARNED.exists() else {}
        old.update(got)
        LEARNED.write_text(json.dumps(old, indent=0, sort_keys=True), encoding="utf-8")
        print(f"learned {len(got)} inactive players' NBA ids by name; rerun to use them")
pd.DataFrame(rows).to_csv(OUT / f"{PREFIX}nba_api_{season}.csv", index=False)
pd.DataFrame(box).to_csv(OUT / f"{PREFIX}box_v3_{season}.csv", index=False)
print(f"wrote {len(rows)} player-games, {len(box)} box rows ({added} inactives from the injury report, {coach} more from the roster) from {len(events)} games; "
      f"{len(unmapped)} players with no NBA id: {list(unmapped.values())[:10]}")

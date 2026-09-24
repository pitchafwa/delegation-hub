"""Did experts' preseason dynasty premiums for young players pay off?
Two real preseason lists:
  RotoWire 2025-26 (published 2025-08-19; top 100 with ages)   -> outcomes: 2025-26 season
  FantraxHQ 2023-24 preseason top 100                          -> outcomes: 2023-24, 2024-25, 2025-26 seasons
For each: compare the expert's rank with the player's REALIZED rank in real fantasy pts/g, split by age / rookie status.
A positive "premium" means the expert ranked a group better than its realized production justified over the window
(dynasty value also includes years beyond the window, so this is a floor on what the premium must be buying).
"""
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"


def nn_(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


ROTO_2025 = """Victor Wembanyama 21|Nikola Jokic 30|Shai Gilgeous-Alexander 27|Luka Doncic 26|Cooper Flagg 18|Cade Cunningham 23|Anthony Edwards 24|Karl-Anthony Towns 29|Giannis Antetokounmpo 30|Chet Holmgren 23|Scottie Barnes 24|Trae Young 26|Paolo Banchero 22|Amen Thompson 22|LaMelo Ball 23|Jayson Tatum 27|Evan Mobley 24|Jalen Johnson 23|Jalen Williams 24|Devin Booker 28|Tyrese Haliburton 25|Donovan Mitchell 28|Domantas Sabonis 29|Jaylen Brown 28|De'Aaron Fox 27|Tyrese Maxey 24|Franz Wagner 23|Bam Adebayo 28|Tyler Herro 25|Jaren Jackson 25|Alperen Sengun 23|Anthony Davis 32|Jalen Brunson 28|Dyson Daniels 22|Josh Giddey 22|Desmond Bane 27|Darius Garland 25|Jamal Murray 28|Brandon Miller 22|Dylan Harper 19|Ausar Thompson 22|Trey Murphy 25|Ja Morant 26|James Harden 35|Kevin Durant 36|Stephen Curry 37|Pascal Siakam 31|Derrick White 31|Zion Williamson 25|LeBron James 40|Austin Reaves 27|Ace Bailey 19|Matas Buzelis 20|Ivica Zubac 28|VJ Edgecombe 20|Alex Sarr 20|Myles Turner 29|Jarrett Allen 27|Walker Kessler 24|Jalen Duren 21|Onyeka Okongwu 24|Dereck Lively 21|Deni Avdija 24|Jordan Poole 26|Coby White 25|Zach LaVine 30|Lauri Markkanen 28|Zach Edey 23|Donovan Clingan 21|Isaiah Hartenstein 27|Mikal Bridges 28|OG Anunoby 28|Miles Bridges 27|Kyrie Irving 33|Joel Embiid 31|Stephon Castle 20|Reed Sheppard 21|Scoot Henderson 21|Devin Vassell 24|Josh Hart 30|Toumani Camara 25|Julius Randle 30|Kel'el Ware 21|Jaden McDaniels 24|Shaedon Sharpe 22|Jalen Suggs 24|Immanuel Quickley 26|Dejounte Murray 28|Jalen Green 23|Cam Thomas 23|Brandon Ingram 27|Anfernee Simons 26|Fred VanVleet 31|Jimmy Butler 35|Rudy Gobert 33|Jakob Poeltl 29|Naz Reid 25|Nic Claxton 26|Zaccharie Risacher 20|Kon Knueppel 20"""
FANTRAX_2023 = """Nikola Jokic|Luka Doncic|Jayson Tatum|Giannis Antetokounmpo|Anthony Edwards|LaMelo Ball|Tyrese Haliburton|Joel Embiid|Shai Gilgeous-Alexander|Devin Booker|Karl-Anthony Towns|Ja Morant|Trae Young|Domantas Sabonis|Evan Mobley|Darius Garland|Cade Cunningham|Paolo Banchero|Bam Adebayo|Jaren Jackson Jr|Donovan Mitchell|De'Aaron Fox|Victor Wembanyama|Zion Williamson|Mikal Bridges|Anthony Davis|Brandon Ingram|Jaylen Brown|Dejounte Murray|Deandre Ayton|Jarrett Allen|Jalen Brunson|Damian Lillard|Jamal Murray|Tyrese Maxey|RJ Barrett|Stephen Curry|Scottie Barnes|Nicolas Claxton|Tyler Herro|James Harden|Pascal Siakam|OG Anunoby|Lauri Markkanen|Desmond Bane|Jaden Ivey|Alperen Sengun|Kevin Durant|Scoot Henderson|Julius Randle|Jabari Smith Jr|Kristaps Porzingis|Myles Turner|Zach LaVine|Fred VanVleet|Josh Giddey|Keegan Murray|Paul George|Jordan Poole|Anfernee Simons|Bradley Beal|Jalen Williams|Jimmy Butler|Jalen Green|Kawhi Leonard|John Collins|Robert Williams|Kyrie Irving|Walker Kessler|Austin Reaves|Devin Vassell|D'Angelo Russell|Franz Wagner|Michael Porter Jr|Amen Thompson|Rudy Gobert|CJ McCollum|Keldon Johnson|Onyeka Okongwu|Wendell Carter Jr|DeMar DeRozan|Kyle Kuzma|Cameron Johnson|LeBron James|Jrue Holiday|AJ Griffin|Ausar Thompson|Mitchell Robinson|Kevin Huerter|Brandon Miller|Jalen Duren|Bennedict Mathurin|Khris Middleton|Aaron Gordon|Mark Williams|Jalen Suggs|Tobias Harris|Nikola Vucevic|Jeremy Sochan|Herbert Jones"""

sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["fpg"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"] - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
sb["norm"] = sb["PLAYER_NAME"].apply(nn_)
bio = pd.read_csv(D / "player_bio.csv")
bio["norm"] = (bio["PLAYER_FIRST_NAME"] + " " + bio["PLAYER_LAST_NAME"]).apply(nn_)
draft = bio.drop_duplicates("norm").set_index("norm")["DRAFT_YEAR"]


def realized(norm, years):
    v = sb[(sb["norm"] == norm) & sb["yr"].isin(years) & (sb["GP"] >= 15)]["fpg"]
    return float(v.mean()) if len(v) else np.nan


def analyze(rows, years, label, young_fn):
    df = pd.DataFrame(rows, columns=["rank", "player", "age"])
    df["norm"] = df["player"].apply(nn_)
    df["real"] = df["norm"].apply(lambda n: realized(n, years))
    df = df.dropna(subset=["real"]).copy()
    df["real_rank"] = df["real"].rank(ascending=False)
    df["exp_rank"] = df["rank"].rank()
    df["young"] = df.apply(young_fn, axis=1)
    df["disp"] = df["real_rank"] - df["exp_rank"]   # + = realized worse than the expert ranked him
    print(f"\n=== {label}: n={len(df)}; Spearman(expert rank, realized rank) = {-spearmanr(df['rank'], df['real'])[0]:.3f}")
    for name, sub in [("young / recent draftees", df[df.young]), ("everyone else", df[~df.young])]:
        print(f"  {name:24s} n={len(sub):3d}  mean expert rank {sub.exp_rank.mean():5.1f}  mean realized rank {sub.real_rank.mean():5.1f}  "
              f"premium (realized worse than ranked): {sub.disp.mean():+5.1f}   share whose realized rank is worse: {(sub.disp > 0).mean():.0%}")
    return df


rows = [(i + 1, t.rsplit(" ", 1)[0], int(t.rsplit(" ", 1)[1])) for i, t in enumerate(ROTO_2025.split("|"))]
d1 = analyze(rows, [2025], "RotoWire preseason 2025-26 -> realized 2025-26", lambda r: r["age"] <= 22)
print("  biggest young-player misses (expert rank -> realized rank among the 100):")
t = d1[d1.young].sort_values("disp", ascending=False)
print("   over-ranked:", ", ".join(f"{r.player} ({int(r.exp_rank)}->{int(r.real_rank)})" for r in t.head(6).itertuples()))
print("   under-ranked:", ", ".join(f"{r.player} ({int(r.exp_rank)}->{int(r.real_rank)})" for r in t.tail(6).itertuples()))
rows2 = [(i + 1, t, np.nan) for i, t in enumerate(FANTRAX_2023.split("|"))]
d2 = analyze(rows2, [2023, 2024, 2025], "FantraxHQ preseason 2023-24 -> realized average 2023-24..2025-26",
             lambda r: (draft.get(r["norm"], 0) or 0) >= 2020)
print("  young (2020+ draftees) detail:", ", ".join(f"{r.player} ({int(r.exp_rank)}->{int(r.real_rank)})" for r in d2[d2.young].sort_values("exp_rank").itertuples()))

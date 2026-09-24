"""A premium targeted at YOUNG + ELITE-PEDIGREE players (top-5 pick, age <= 21), not all young players.
value' = value(H=10, discount .95) * (1 + beta * elite_young),  elite_young = 1 if draft pick <= 5 and age <= 21.5 (else 0)
Fit beta on one outlet, test on the other.
"""
import contextlib, re, sys, unicodedata, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
class _Q:
    def write(self,s): pass
    def flush(self): pass
    def reconfigure(self,**k): pass
src=open("horizon_test.py",encoding="utf-8").read().split('names = list(live["PLAYER_NAME"])')[0]
with contextlib.redirect_stdout(_Q()): exec(src)
names=list(live["PLAYER_NAME"])+[q["player"] for q in pros]
ages=np.concatenate([live_age_next, pr_age])
# draft slot of each player: vets from the panel, prospects from the hub
pick_v=np.exp(live["logpick"].to_numpy()) if "logpick" in live else live["draft_pick"].to_numpy()
picks=np.concatenate([np.round(pick_v), pr_pick_actual])
def nn_(n):
    n=unicodedata.normalize("NFKD",str(n)).encode("ascii","ignore").decode(); n=re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?","",n,flags=re.I); return re.sub(r"\s+"," ",re.sub(r"[^a-z ]","",n.lower())).strip()
dyn=pd.read_csv(D/"hashtag_dynasty_2026-09-24.csv"); dyn["norm"]=dyn.player.apply(nn_)
ROTO={"Victor Wembanyama":1,"Nikola Jokic":2,"Shai Gilgeous-Alexander":3,"Luka Doncic":4,"Cade Cunningham":5,"Cooper Flagg":6,"Jayson Tatum":7,"Jalen Johnson":8,"Anthony Edwards":9,"Tyrese Maxey":10,"Scottie Barnes":11,"Tyrese Haliburton":12,"Chet Holmgren":13,"Josh Giddey":14,"Cameron Boozer":15,"Amen Thompson":16,"Evan Mobley":17,"Jalen Williams":18,"Giannis Antetokounmpo":19,"Donovan Mitchell":20,"Karl-Anthony Towns":21,"Deni Avdija":22,"Devin Booker":23,"Trey Murphy":24,"Brandon Miller":25,"Jalen Brunson":26,"Alperen Sengun":27,"Bam Adebayo":28,"LaMelo Ball":29,"Trae Young":30,"Caleb Wilson":37,"Darryn Peterson":38,"Dylan Harper":40,"Kon Knueppel":41,"AJ Dybantsa":42,"VJ Edgecombe":46,"Ace Bailey":97}
rw=pd.DataFrame({"norm":[nn_(k) for k in ROTO],"rw":list(ROTO.values())})
base=pd.DataFrame({"player":names,"age":ages,"pick":picks}); base["norm"]=base.player.apply(nn_)
elite=((base.pick<=5)&(base.age<=21.5)).to_numpy()
print("elite-young group (top-5 pick, age<=21):",", ".join(sorted(base[elite].player)))
def score(k,beta):
    v=total_value_h(k,0.95,10)[:len(names)]*(1+beta*elite)
    b=base.copy(); b["v"]=v; b=b.drop_duplicates("norm").merge(dyn[["norm","rank"]].rename(columns={"rank":"hash"}),on="norm",how="left").merge(rw,on="norm",how="left")
    h=b.dropna(subset=["hash"]); r=b.dropna(subset=["rw"]); e=b[b.norm.isin(base[elite].norm)]
    return -spearmanr(h.v,h.hash)[0], -spearmanr(r.v,r.rw)[0], b
for k in (5,19):
    print(f"\nK={k}: beta | agreement Hashtag (all 173) | agreement RotoWire (37) | mean rank of the elite-young group in OUR list vs Hashtag vs RotoWire")
    for beta in (0,0.15,0.3,0.5,0.75,1.0):
        a,c,b=score(k,beta); b["our_rank"]=b.v.rank(ascending=False)
        g=b[b.norm.isin(base[elite].norm)]
        print(f"     {beta:4.2f} |   {a:.3f}   |   {c:.3f}   |   ours {g.our_rank.mean():5.1f}   Hashtag {g.hash.mean():5.1f}   RotoWire {g.rw.mean():5.1f}")
_,_,b=score(19,0.5); b["our_rank"]=b.v.rank(ascending=False); _,_,b0=score(19,0); b0["our_rank"]=b0.v.rank(ascending=False)
print("\nrank (K=19): no premium -> beta 0.5   [Hashtag | RotoWire]")
for who in ["Cooper Flagg","Cameron Boozer","AJ Dybantsa","Darryn Peterson","Caleb Wilson","Dylan Harper","Kon Knueppel","VJ Edgecombe","Ace Bailey"]:
    r0=b0[b0.player==who]; r1=b[b.player==who]
    if r0.empty: continue
    print(f"  {who:16s} {int(r0.our_rank.iloc[0]):4d} -> {int(r1.our_rank.iloc[0]):4d}   | {'' if pd.isna(r1.hash.iloc[0]) else int(r1.hash.iloc[0]):>4} | {'' if pd.isna(r1.rw.iloc[0]) else int(r1.rw.iloc[0]):>4}")

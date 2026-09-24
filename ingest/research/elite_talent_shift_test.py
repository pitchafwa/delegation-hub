"""Express the elite-young premium as a TALENT uplift in pts/g (all seasons) for top-5 picks age<=21 -- the consensus' scouting
information beyond draft slot -- and find the size that puts the group between the two independent outlets at every keeper count."""
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
names=list(live["PLAYER_NAME"])+[q["player"] for q in pros]; ages=np.concatenate([live_age_next, pr_age])
pick_v=np.round(np.exp(live["logpick"].to_numpy())); picks=np.concatenate([pick_v, pr_pick_actual])
def nn_(n):
    n=unicodedata.normalize("NFKD",str(n)).encode("ascii","ignore").decode(); n=re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?","",n,flags=re.I); return re.sub(r"\s+"," ",re.sub(r"[^a-z ]","",n.lower())).strip()
dyn=pd.read_csv(D/"hashtag_dynasty_2026-09-24.csv"); dyn["norm"]=dyn.player.apply(nn_)
ROTO={"Victor Wembanyama":1,"Nikola Jokic":2,"Shai Gilgeous-Alexander":3,"Luka Doncic":4,"Cade Cunningham":5,"Cooper Flagg":6,"Jayson Tatum":7,"Jalen Johnson":8,"Anthony Edwards":9,"Tyrese Maxey":10,"Scottie Barnes":11,"Tyrese Haliburton":12,"Chet Holmgren":13,"Josh Giddey":14,"Cameron Boozer":15,"Amen Thompson":16,"Evan Mobley":17,"Jalen Williams":18,"Giannis Antetokounmpo":19,"Donovan Mitchell":20,"Karl-Anthony Towns":21,"Deni Avdija":22,"Devin Booker":23,"Trey Murphy":24,"Brandon Miller":25,"Jalen Brunson":26,"Alperen Sengun":27,"Bam Adebayo":28,"LaMelo Ball":29,"Trae Young":30,"Caleb Wilson":37,"Darryn Peterson":38,"Dylan Harper":40,"Kon Knueppel":41,"AJ Dybantsa":42,"VJ Edgecombe":46,"Ace Bailey":97}
rw=pd.DataFrame({"norm":[nn_(k) for k in ROTO],"rw":list(ROTO.values())})
base=pd.DataFrame({"player":names,"age":ages,"pick":picks}); base["norm"]=base.player.apply(nn_)
elite=((base.pick<=5)&(base.age<=21.5)).to_numpy()
ev_e=np.where(elite[:len(live)],1.0,0.0); ep_e=elite[len(live):].astype(float)
Ev10_0,Ep10_0=Ev10.copy(),Ep10.copy()
target=(dyn[dyn.norm.isin(base[elite].norm)]["rank"].mean(), rw[rw.norm.isin(base[elite].norm)]["rw"].mean())
print(f"group of {elite.sum()}; outlets' mean rank for it: Hashtag {target[0]:.1f}, RotoWire {target[1]:.1f}  (midpoint {np.mean(target):.1f})")
rows={}
for X in (0,1,2,3,4,5,6):
    Ev10[:]=Ev10_0+X*ev_e[:,None]; Ep10[:]=Ep10_0+X*ep_e[:,None]
    for k in (1,3,5,19):
        v=total_value_h(k,0.95,10)[:len(names)]
        b=base.copy(); b["v"]=v; b=b.drop_duplicates("norm").merge(dyn[["norm","rank"]].rename(columns={"rank":"hash"}),on="norm",how="left").merge(rw,on="norm",how="left")
        b["our"]=b.v.rank(ascending=False); g=b[b.norm.isin(base[elite].norm)]
        h=b.dropna(subset=["hash"]); r=b.dropna(subset=["rw"])
        rows[(X,k)]=(g.our.mean(), -spearmanr(h.v,h.hash)[0], -spearmanr(r.v,r.rw)[0], b)
print("\nuplift (pts/g, every season) | elite-young group's mean rank in OUR list at K=1 / 3 / 5 / 19  | agreement with Hashtag / RotoWire at K=5")
for X in (0,1,2,3,4,5,6):
    print(f"   +{X}  |  "+"  ".join(f"{rows[(X,k)][0]:5.1f}" for k in (1,3,5,19))+f"   |   {rows[(X,5)][1]:.3f} / {rows[(X,5)][2]:.3f}")
for X in (0,3,4):
    b=rows[(X,5)][3]
    print(f"\n+{X} pts/g uplift, K=5 ranks  [Hashtag | RotoWire]")
    for who in ["Cooper Flagg","Cameron Boozer","AJ Dybantsa","Darryn Peterson","Caleb Wilson","Dylan Harper","Kon Knueppel","VJ Edgecombe","Ace Bailey"]:
        r=b[b.player==who]
        if len(r): print(f"   {who:16s} {int(r.our.iloc[0]):4d}   | {'' if pd.isna(r.hash.iloc[0]) else int(r.hash.iloc[0]):>4} | {'' if pd.isna(r.rw.iloc[0]) else int(r.rw.iloc[0]):>4}")

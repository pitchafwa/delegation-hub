"""Fit a youth premium on ONE expert source, test on the OTHER.
value' = value(H=10, discount 0.95) * (1 + beta * youth),  youth = max(0, 24 - age) / 4  (1.0 at 20, 0 at 24+)
Fit beta on the Hashtag crowd top-200 (205k votes); test on RotoWire's 2026-27 list (independent outlet); and the reverse.
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
src=open("horizon_test.py",encoding="utf-8").read()
# run the horizon machinery quietly, but keep its variables
tail_marker='names = list(live["PLAYER_NAME"])'
head=src.split(tail_marker)[0]
head=head.replace('sys.stdout.reconfigure(encoding="utf-8")','')
class _Q:
    def write(self,s): pass
    def flush(self): pass
    def reconfigure(self,**k): pass
with contextlib.redirect_stdout(_Q()): exec(head)
import numpy as np, pandas as pd, re, unicodedata
from scipy.stats import spearmanr
names=list(live["PLAYER_NAME"])+[q["player"] for q in pros]
ages=np.concatenate([live_age_next, pr_age])
def nn_(n):
    n=unicodedata.normalize("NFKD",str(n)).encode("ascii","ignore").decode(); n=re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?","",n,flags=re.I); return re.sub(r"\s+"," ",re.sub(r"[^a-z ]","",n.lower())).strip()
dyn=pd.read_csv(D/"hashtag_dynasty_2026-09-24.csv"); dyn["norm"]=dyn.player.apply(nn_)
exec(open("expert_premium_check.py",encoding="utf-8").read().split("sb = pd.read_csv")[0].split("ROTO_2025")[0]) if False else None
ROTO={"Victor Wembanyama":1,"Nikola Jokic":2,"Shai Gilgeous-Alexander":3,"Luka Doncic":4,"Cade Cunningham":5,"Cooper Flagg":6,"Jayson Tatum":7,"Jalen Johnson":8,"Anthony Edwards":9,"Tyrese Maxey":10,"Scottie Barnes":11,"Tyrese Haliburton":12,"Chet Holmgren":13,"Josh Giddey":14,"Cameron Boozer":15,"Amen Thompson":16,"Evan Mobley":17,"Jalen Williams":18,"Giannis Antetokounmpo":19,"Donovan Mitchell":20,"Karl-Anthony Towns":21,"Deni Avdija":22,"Devin Booker":23,"Trey Murphy":24,"Brandon Miller":25,"Jalen Brunson":26,"Alperen Sengun":27,"Bam Adebayo":28,"LaMelo Ball":29,"Trae Young":30,"Caleb Wilson":37,"Darryn Peterson":38,"Dylan Harper":40,"Kon Knueppel":41,"AJ Dybantsa":42,"VJ Edgecombe":46,"Ace Bailey":97}
rw=pd.DataFrame({"norm":[nn_(k) for k in ROTO],"rw":list(ROTO.values())})
base=pd.DataFrame({"player":names,"age":ages}); base["norm"]=base.player.apply(nn_)
def vals(k,delta,H):
    return total_value_h(k,delta,H)[:len(names)]
res={}
for k in (5,19):
    v=vals(k,0.95,10)
    for beta in (0,0.1,0.2,0.3,0.5,0.75,1.0):
        b=base.copy(); b["v"]=v*(1+beta*np.clip(24-b.age,0,None)/4)
        b=b.drop_duplicates("norm").merge(dyn[["norm","rank"]].rename(columns={"rank":"hash"}),on="norm",how="left").merge(rw,on="norm",how="left")
        h=b.dropna(subset=["hash"]); r=b.dropna(subset=["rw"])
        res[(k,beta)]=(-spearmanr(h.v,h.hash)[0], -spearmanr(r.v,r.rw)[0])
for k in (5,19):
    print(f"\nK={k}: youth premium beta  |  agreement with Hashtag crowd (n=173)  |  agreement with RotoWire top-30+prospects (n=37)")
    for beta in (0,0.1,0.2,0.3,0.5,0.75,1.0):
        a,b=res[(k,beta)]; print(f"   beta={beta:4.2f}          {a:.3f}                                   {b:.3f}")
# where do the key players land at a middle premium?
v=vals(19,0.95,10)
print("\nrank (K=19, H=10, discount 0.95) for beta = 0 / 0.3 / 0.5  [Hashtag | RotoWire]")
for who in ["Cooper Flagg","Cameron Boozer","AJ Dybantsa","Darryn Peterson","Caleb Wilson","Dylan Harper","Ace Bailey","Kon Knueppel","VJ Edgecombe"]:
    row=f"{who:16s} "
    for beta in (0,0.3,0.5):
        s=pd.Series(v*(1+beta*np.clip(24-ages,0,None)/4),index=names).groupby(level=0).first()
        row+=f"{int((s>s[who]).sum()+1):6d} "
    hh=dyn[dyn.norm==nn_(who)]["rank"]; rr=rw[rw.norm==nn_(who)]["rw"]
    row+=f" | {int(hh.iloc[0]) if len(hh) else '':>4} | {int(rr.iloc[0]) if len(rr) else '':>4}"
    print(row)

import json, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D=json.load(open(r"C:\Users\tommy\OneDrive\Documents\Claude\fantasy-basketball-hub\ingest\research\data\league_days_2026.json"))
STARTING={0,1,2,3,4,5,6,7,8,9,10,11}
rows=[]
for d,recs in D.items():
    for r in recs:
        st=[e for e in r["e"] if e[2] in STARTING and e[3]]
        rows.append((int(d),r["mp"],r["team"],len(st)))
df=pd.DataFrame(rows,columns=["day","mp","team","starts"])
df=df.sort_values(["mp","team","day"])
df["cum"]=df.groupby(["mp","team"]).starts.cumsum()
df["dayidx"]=df.groupby(["mp","team"]).cumcount()
df["ndays"]=df.groupby(["mp","team"]).day.transform("count")
df["last"]=(df.dayidx==df.ndays-1)
tw=df.groupby(["mp","team"]).agg(total=("starts","sum"),ndays=("ndays","first")).reset_index()
print("team-weeks by total starts (mp<=19, 7-day only):")
w7=tw[(tw.ndays==7)&(tw.mp<=19)]
print(w7.total.describe().round(1).to_dict(), " >40:", (w7.total>40).mean().round(2), " >45:",(w7.total>45).mean().round(2))
# before the final day
pre=df[~df["last"]].groupby(["mp","team"]).starts.sum().rename("pre_last")
lastd=df[df["last"]].set_index(["mp","team"]).starts.rename("last_day")
x=w7.set_index(["mp","team"]).join(pre).join(lastd)
print("\nstarts before final day: max", x.pre_last.max(), " 95th pct", x.pre_last.quantile(.95), " share of team-weeks with pre_last>40:", (x.pre_last>40).mean().round(2))
print("distribution of pre-final-day starts:", x.pre_last.describe().round(1).to_dict())
print("\nteam-weeks whose total>40:", (x.total>40).sum(), " of those, pre_last<=40:", ((x.total>40)&(x.pre_last<=40)).sum(), "; pre_last>40:", ((x.total>40)&(x.pre_last>40)).sum())
print("\nlast-day starts when pre_last<=40 vs >40:")
print(x.groupby(x.pre_last>40).last_day.describe().round(1))
# per-day starts pattern in a week where pre_last>40: did the count on days after crossing 40 continue?
ex=x[x.pre_last>40].head(5)
for (mp,team),_ in ex.iterrows():
    g=df[(df.mp==mp)&(df.team==team)]
    print(mp,team,g.starts.tolist(), g.cum.tolist())

print("\n=== boundary: cumulative starts at START of a day vs starts allowed that day (all matchup periods) ===")
df["cum_before"]=df.cum-df.starts
for lo,hi in [(0,29),(30,34),(35,37),(38,38),(39,39),(40,40),(41,44),(45,90)]:
    s=df[(df.cum_before>=lo)&(df.cum_before<=hi)&(df.mp<=19)]
    print(f"cum before day {lo}-{hi}: days={len(s):5d}  share with any start (excl. no-game days unknown): {(s.starts>0).mean():.2f}  mean starts {s.starts.mean():.2f}")
print("\nweeks by length -> max total starts, and implied cap (max total, 95th pct)")
tw2=tw[tw.mp<=19]
print(tw2.groupby("ndays").total.agg(["count","max",lambda s:s.quantile(.95)]).round(1))
print("\nperiod 17 (14 days) team totals:", sorted(tw[tw.mp==17].total.tolist()))
print("period 1 (6 days) team totals:", sorted(tw[tw.mp==1].total.tolist()))
print("\nplayoff periods 20-22 totals max:", tw[tw.mp>=20].groupby("mp").total.max().to_dict())

print("\n=== 7-day periods only (incl. playoffs): cum starts at start of day -> did the team get to start anyone that day? (days where team had >=1 healthy player playing unknown; use share>0) ===")
s7=df[(df.ndays==7)]
for lo,hi in [(0,29),(30,36),(37,37),(38,38),(39,39),(40,40),(41,60)]:
    s=s7[(s7.cum_before>=lo)&(s7.cum_before<=hi)]
    print(f"cum before {lo}-{hi}: team-days {len(s):5d}  any starts {(s.starts>0).mean():.2f}  mean starts {s.starts.mean():.2f}  max {s.starts.max()}")
print("\nweek-total distribution 7-day periods:", s7.groupby(['mp','team']).starts.sum().value_counts().sort_index().tail(12).to_dict())
p1=df[df.mp==1]
print("\nperiod 1 (6 days), cum before day vs starts:")
for lo,hi in [(0,28),(29,33),(34,34),(35,60)]:
    s=p1[(p1.cum_before>=lo)&(p1.cum_before<=hi)]
    print(f"  cum before {lo}-{hi}: days {len(s)} any starts {(s.starts>0).mean():.2f} mean {s.starts.mean():.2f}")

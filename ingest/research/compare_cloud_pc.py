"""Laptop-free test, part 2 (2026-11-01, Tommy): does the cloud refresh (ESPN box scores) produce the same model numbers and the same plan as the
laptop (NBA site)? Compares the dashboard files both sides build, and two weekly plans built back to back from the same ESPN league state: one on
the laptop's model files, one on the cloud's.
  models(pc_dir, cloud_dir) -> per-file summary: players on each side, how many are within a small tolerance, mean / biggest differences, worst five
  plans(pc_plan, cloud_plan) -> for Tommy's team: same starters each day?, same adds/drops, same IR moves, same tonight picks, win chance gap
Run by cloud_test.py; by hand:  python research/compare_cloud_pc.py PC_DASHBOARD_DIR CLOUD_DASHBOARD_DIR [pc_week_plan cloud_week_plan]
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _num(a, b, names, tol):
    """a, b: {key: number}. Share of shared keys within tol, mean and max abs difference, worst five."""
    ks = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    if not ks:
        return {"n": 0, "only_pc": len(a), "only_cloud": len(b)}
    d = sorted(((abs(float(a[k]) - float(b[k])), k) for k in ks), reverse=True)
    return {"n": len(ks), "only_pc": len(set(a) - set(b)), "only_cloud": len(set(b) - set(a)), "tol": tol,
            "within": round(sum(x <= tol for x, _ in d) / len(d), 4), "mean_abs": round(sum(x for x, _ in d) / len(d), 3), "max": round(d[0][0], 2),
            "worst": [f"{names.get(k, k)}: laptop {float(a[k]):.1f}, cloud {float(b[k]):.1f}" for x, k in d[:5] if x > tol]}


def models(pc, cloud):
    pc, cloud = Path(pc), Path(cloud)
    out = {}
    hp, hc = _load(pc / "hub_data.json"), _load(cloud / "hub_data.json")
    if hp and hc:
        P = {p["id"]: p for p in hp["players"]}
        C = {p["id"]: p for p in hc["players"]}
        nm = {k: v["player"] for k, v in P.items()}
        first = lambda d, k: {i: (p.get(k) or [None])[0] if isinstance(p.get(k), list) else p.get(k) for i, p in d.items()}
        out["keeper_value"] = _num(first(P, "asset_k"), first(C, "asset_k"), nm, 0.5)          # dynasty value (this year's asset value)
        out["delco_season"] = _num(first(P, "season_ppg"), first(C, "season_ppg"), nm, 0.5)    # DELCO's points per game for this season
        out["kalman_now"] = _num(first(P, "trajectory_kalman"), first(C, "trajectory_kalman"), nm, 0.5)
        out["proj_fpg"] = _num(first(P, "proj_fpg"), first(C, "proj_fpg"), nm, 0.5)
        out["injury_tier"] = _num(first(P, "injury_tier"), first(C, "injury_tier"), nm, 0)
    rp, rc = _load(pc / "redraft_data.json"), _load(cloud / "redraft_data.json")
    if rp and rc:
        P = {p["id"]: p for p in rp["players"]}
        C = {p["id"]: p for p in rc["players"]}
        nm = {k: v["name"] for k, v in P.items()}
        g = lambda d, k: {i: p.get(k) for i, p in d.items()}
        out["rest_of_season_ppg"] = _num(g(P, "ppg"), g(C, "ppg"), nm, 0.5)
        out["next_days"] = _num({i: (p.get("next_proj") or [None])[0] for i, p in P.items()}, {i: (p.get("next_proj") or [None])[0] for i, p in C.items()}, nm, 0.5)
        out["last10"] = _num(g(P, "l10"), g(C, "l10"), nm, 0.5)
        out["minutes10"] = _num(g(P, "m10"), g(C, "m10"), nm, 0.5)
        out["games_left"] = _num(g(P, "exp_gp"), g(C, "exp_gp"), nm, 0.5)
        out["rank"] = _num(g(P, "rank"), g(C, "rank"), nm, 2)
    sp, sc = _load(pc / "rotation_status.json"), _load(cloud / "rotation_status.json")
    if sp and sc:
        o = lambda d: {k for k, v in d["players"].items() if v["n"] >= 3 and v["played"] <= 1}
        out["rotation"] = {"out_pc": len(o(sp)), "out_cloud": len(o(sc)), "only_pc": sorted(o(sp) - o(sc))[:15], "only_cloud": sorted(o(sc) - o(sp))[:15]}
    fp, fc = _load(pc / "form_split.json"), _load(cloud / "form_split.json")
    if fp and fc:
        out["form"] = {"pc": len(fp["players"]), "cloud": len(fc["players"]), "only_pc": sorted(set(fp["players"]) - set(fc["players"]))[:10],
                       "only_cloud": sorted(set(fc["players"]) - set(fp["players"]))[:10]}
    tp, tc = _load(pc / "team_matchup.json"), _load(cloud / "team_matchup.json")
    if tp and tc:
        flat = lambda d: {f"{t}.{k}": v for t, x in d["teams"].items() if isinstance(x, dict) for k, v in x.items() if isinstance(v, (int, float))}
        out["team_matchup"] = _num(flat(tp), flat(tc), {}, 0.01)
    return out


def plans(pc, cloud):
    a, b = _load(pc), _load(cloud)
    if not a or not b:
        return {"error": "a plan is missing"}
    ta = next(t for t in a["teams"] if t["abbrev"] == a["my_abbrev"])
    tb = next(t for t in b["teams"] if t["abbrev"] == b["my_abbrev"])
    days = []
    for da, db in zip(ta["days"], tb["days"]):
        sa, sb = {x["name"] for x in da["start"]}, {x["name"] for x in db["start"]}
        days.append({"date": da["date"], "same": sa == sb, "only_pc": sorted(sa - sb), "only_cloud": sorted(sb - sa)})
    mv = lambda t: [f"+{(s.get('add') or {}).get('name', '?')} -{(s.get('drop') or {}).get('name', '-')}" for s in t.get("sequence") or []]
    ir = lambda t: [f"{m['action']} {m['name']}" for m in t.get("ir_moves") or []]
    tn = lambda t: [p["name"] for p in (t.get("tonight_picks") or [])[:3]]
    return {"days": days, "starters_same_days": sum(d["same"] for d in days), "n_days": len(days),
            "moves_pc": mv(ta), "moves_cloud": mv(tb), "moves_same": mv(ta) == mv(tb),
            "ir_pc": ir(ta), "ir_cloud": ir(tb), "ir_same": ir(ta) == ir(tb),
            "tonight_pc": tn(ta), "tonight_cloud": tn(tb), "tonight_same": set(tn(ta)) == set(tn(tb)),
            "win_pc": ta.get("win_prob"), "win_cloud": tb.get("win_prob"),
            "expected_pc": ta.get("expected_total"), "expected_cloud": tb.get("expected_total")}


if __name__ == "__main__":
    a = sys.argv[1:]
    res = {"models": models(a[0], a[1])}
    if len(a) >= 4:
        res["plan"] = plans(a[2], a[3])
    print(json.dumps(res, indent=1, ensure_ascii=False))

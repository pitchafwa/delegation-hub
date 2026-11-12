"""Apply the site's value-horizon setting (dashboard/site_settings.json "horizon", 1-10 seasons) to dashboard/hub_data.json without rerunning the models:
asset_k / ceil_asset_k become the values at that horizon, picked from the per-horizon tables the nightly build stores (asset_kh / ceil_asset_kh).
Run by site-setting.yml right after the setting changes (then the plan rebuilds); the nightly build_hub_data.py does the same pick itself.
  python research/apply_horizon.py [--set N]        (from ingest/; --set also writes the setting)
"""
import json
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
SET = HUB / "site_settings.json"


def main():
    s = json.loads(SET.read_text(encoding="utf-8")) if SET.exists() else {}
    if "--set" in sys.argv:
        s["horizon"] = int(sys.argv[sys.argv.index("--set") + 1])
    h = max(1, min(10, int(s.get("horizon", 10))))
    s["horizon"] = h
    SET.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    p = HUB / "hub_data.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    n = 0
    for pl in d["players"]:
        if pl.get("asset_kh"):
            pl["asset_k"] = pl["asset_kh"][h - 1]
            n += 1
        if pl.get("ceil_asset_kh"):
            pl["ceil_asset_k"] = pl["ceil_asset_kh"][h - 1]
    d["value_horizon"] = h
    p.write_text(json.dumps(d, indent=None, allow_nan=False), encoding="utf-8")         # same format as build_hub_data.py
    print(f"value horizon {h}: asset values re-picked for {n} players")


if __name__ == "__main__":
    main()

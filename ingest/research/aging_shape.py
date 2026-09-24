"""Age-dependent annual slope used for FORWARD projection (validated out-of-sample, see fit_aging_shape.py).

Finding that shaped this: for players who already have NBA data the original curve (one up-slope to a
peak age, one down-slope after) was roughly unbiased through age ~30, and a fully re-fit steeper curve was
WORSE for young players when graded honestly. What held up out-of-sample was the late-career decline: for
players 31+ the old curve was too optimistic (forecast ~2.6 pts/g too high). So: keep the original slope
below ~28, blend to the empirically fitted slope by 31, at 60% strength (best held-out error).
"""
import json
from pathlib import Path

import numpy as np

_CFG = json.loads((Path(__file__).resolve().parent / "data" / "aging_shape.json").read_text(encoding="utf-8"))
_KNOTS = np.array(_CFG["knots"], dtype=float)
_SLOPES = {s: np.array(v, dtype=float) for s, v in _CFG["slopes"].items()}
_MIX = float(_CFG["mix_strength"])
_LO, _HI = _CFG["ramp_ages"]


def annual_slope(stat, age, params):
    """params = (Q, R, peak_age, slope_up, slope_down) from the original per-stat fit. Works on scalars or arrays."""
    Q, R, peak, su, sd = params
    age = np.asarray(age, dtype=float)
    old = np.where(age - peak <= 0, su, sd)
    new = np.interp(age, _KNOTS, _SLOPES[stat])
    t = _MIX * np.clip((age - _LO) / (_HI - _LO), 0.0, 1.0)
    out = old + t * (new - old)
    return float(out) if out.ndim == 0 else out

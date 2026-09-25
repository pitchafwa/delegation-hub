"""Shared helpers for the injury research: reason-text parsing and the played/not-played outcome for each report row.

parse_reason(text) -> dict(kind, side, part, group, nature, tier)
   kind    injury | illness | gleague | rest | personal | suspension | mgmt (return to competition reconditioning) | health_safety | other
   part    canonical body part as printed (e.g. 'ankle', 'left hamstring' -> 'hamstring'); group merges related parts (leg soft tissue, foot/ankle, ...)
   nature  the phrase after ';' (Sprain, Soreness, Surgery, Injury Recovery, Injury Management ...)
   tier    minor | moderate | major | recovery | management  (see TIER rules; recovery/management are returns from earlier trouble)
"""
import re
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent / "data"
SUFFIX_WORDS = {"jr", "sr", "ii", "iii", "iv"}

GROUPS = {
    "ankle": "ankle_foot", "foot": "ankle_foot", "toe": "ankle_foot", "great toe": "ankle_foot", "heel": "ankle_foot", "plantar fascia": "ankle_foot", "achilles": "achilles",
    "achilles tendon": "achilles", "knee": "knee", "patella": "knee", "patellar tendon": "knee", "acl": "knee",
    "hamstring": "leg_soft", "calf": "leg_soft", "quad": "leg_soft", "quadricep": "leg_soft", "quadriceps": "leg_soft", "thigh": "leg_soft", "groin": "leg_soft",
    "adductor": "leg_soft", "hip flexor": "leg_soft", "shin": "leg_soft", "leg": "leg_soft", "hip": "hip_back", "back": "hip_back", "low back": "hip_back", "lower back": "hip_back",
    "glute": "hip_back", "si joint": "hip_back", "sacroiliac": "hip_back", "pelvis": "hip_back", "tailbone": "hip_back",
    "shoulder": "upper", "elbow": "upper", "wrist": "upper", "hand": "upper", "finger": "upper", "thumb": "upper", "forearm": "upper", "arm": "upper", "rib": "upper", "ribs": "upper", "chest": "upper", "neck": "upper", "biceps": "upper", "trapezius": "upper",
    "head": "head", "concussion": "head", "eye": "head", "face": "head", "nose": "head", "orbital": "head", "jaw": "head", "mouth": "head", "ear": "head",
}
MAJOR = re.compile(r"surger|tear|torn|fractur|broken|acl|repair|rupture|dislocat|stress|reconstruct|meniscus|avulsion|labrum|cartilage", re.I)
RECOVERY = re.compile(r"recovery|return to (play|competition)|reconditioning|rehab", re.I)
MGMT = re.compile(r"management", re.I)
MODERATE = re.compile(r"sprain|strain|tendin|tendon|inflam|bone bruise|impinge|subluxat|bursitis|irritation|sublux|plantar|turf|stress", re.I)
MINOR = re.compile(r"sore|tight|contus|spasm|stiff|swell|bruis|lacerat|cramp|illness|hyperext|abrasion|cut|blister|infect|flu|virus|cold|non-covid|dehydrat|migraine|headache|vertigo|dizz", re.I)


def canon_part(raw):
    p = raw.lower().strip()
    p = re.sub(r"^(left|right|both)\s+", "", p)
    p = p.replace("n/a", "").strip()
    return p


def parse_reason(text):
    t = (text or "").strip()
    if not t or t == "-":
        return dict(kind="other", side=None, part="", group="other", nature="", tier="other")
    low = t.lower()
    if low.startswith("g league"):
        return dict(kind="gleague", side=None, part="", group="other", nature="", tier="other")
    if low.startswith("health and safety"):
        return dict(kind="health_safety", side=None, part="", group="other", nature="", tier="other")
    if low.startswith("concussion protocol"):
        return dict(kind="injury", side=None, part="head", group="head", nature="concussion protocol", tier="moderate")
    for k, key in (("rest", "rest"), ("personal", "personal"), ("league suspension", "suspension"), ("not with team", "other"), ("trade pending", "other"), ("ineligible", "other"), ("return to competition", "mgmt")):
        if low.startswith(k):
            return dict(kind=key, side=None, part="", group="other", nature=t, tier="management" if key == "mgmt" else "other")
    if not low.startswith("injury/illness"):
        return dict(kind="other", side=None, part="", group="other", nature=t, tier="other")
    body = re.sub(r"^injury/illness\s*-\s*", "", t, flags=re.I)
    part_raw, _, nature = body.partition(";")
    nature = nature.strip()
    side = "left" if re.match(r"(?i)left", part_raw.strip()) else ("right" if re.match(r"(?i)right", part_raw.strip()) else None)
    part = canon_part(part_raw)
    illness = bool(re.search(r"illness|covid|flu|virus|non-covid|sick", (part_raw + " " + nature).lower())) or part in ("", "illness")
    if illness and not MAJOR.search(nature):
        return dict(kind="illness", side=None, part="illness", group="illness", nature=nature, tier="minor")
    group = GROUPS.get(part)
    if group is None:
        for k, g in GROUPS.items():
            if k in part:
                group = g
                break
    group = group or "other"
    n = nature
    if MAJOR.search(n):
        tier = "major"
    elif RECOVERY.search(n):
        tier = "recovery"
    elif MGMT.search(n):
        tier = "management"
    elif MODERATE.search(n):
        tier = "moderate"
    elif MINOR.search(n):
        tier = "minor"
    else:
        tier = "other"
    return dict(kind="injury", side=side, part=part, group=group, nature=nature.lower(), tier=tier)


def key_of_log(name):
    toks = [t for t in re.sub(r"[^A-Za-z. ]", "", str(name)).split() if t.lower().strip(".") not in SUFFIX_WORDS]
    return re.sub(r"[^a-z]", "", "".join(toks).lower())


def load_reports():
    d = pd.read_csv(D / "injury_reports_hourly.csv")
    d["reason"] = d.reason.fillna("")
    p = d.reason.map(parse_reason).apply(pd.Series)
    d = pd.concat([d, p], axis=1)
    d["gd"] = pd.to_datetime(d.game_date)
    d["season"] = d.gd.apply(lambda x: f"{x.year if x.month >= 8 else x.year - 1}-{str(x.year + 1 if x.month >= 8 else x.year)[2:]}")
    return d


def load_logs():
    g = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_NAME", "SEASON", "GAME_DATE", "TEAM", "MIN"])
    g["key"] = g.PLAYER_NAME.map(key_of_log)
    g["gd"] = pd.to_datetime(g.GAME_DATE)
    return g

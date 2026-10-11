"""Late-swap slot planner: which SLOT each starter should sit in so that late news can be fixed.

ESPN locks a player's slot when HIS OWN game tips off. Players whose games have not started can be moved freely. So over an evening:
  * a player who plays freezes whatever slot he holds when his game starts, and that slot is gone for the rest of the night;
  * a player who is scratched (news arrives before his tip) leaves an empty slot, which only an UNLOCKED player (bench or starter whose game has not started) who is eligible
    for it can fill, possibly by shuffling the other unlocked players.
The slots a night ends up with therefore depend on which slots the EARLY games freeze. Put early-tipping, reliable players in the most restrictive slots (PG, SG, SF, PF, C) and leave the
flexible ones (G, F, UT) for late-tipping or doubtful players: when a late player is scratched, the freed slot accepts any bench player. A doubtful player in a restrictive slot needs a
bench player of the same position who has not tipped yet; in UT, any bench player works. That is the rule of thumb, and this module tests it by simulation instead of assuming it,
because it costs something: putting a doubtful player in UT hands the restrictive slot to someone else, and the best match of players to slots is not always the best for cover.

best_assignment(starters, bench) simulates the evening many times (same random numbers for every candidate assignment): outcomes are revealed in tip-off order; after a scratch the
unlocked players are re-matched to the open slots by expected points; the value of an assignment is the average points of everyone who ends up playing in a starting slot.
It starts from the plan's own assignment and improves it by swapping slots; the result is kept only if it beats the original by a margin larger than simulation noise.

Inputs are dicts: {"i": any id, "slots": eligible slot names, "p": chance he plays, "pts": points if he plays, "tip": sortable tip-off (ISO string)}.
"""
import random

SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
FLEX_RANK = {"PG": 0, "SG": 0, "SF": 0, "PF": 0, "C": 0, "G": 1, "F": 1, "UT": 2}      # how many kinds of player a slot accepts (more = more flexible)
N_SIMS = 150
MIN_GAIN = 0.25            # points per night an assignment must add before the plan changes
SCRATCH_ONLY_AT_TIP = True  # news arrives before the player's own tip-off (the late-swap window); earlier news is handled by the normal daily plan


def _match(players, free):
    """greedy transversal-matroid matching: players sorted by expected points; returns {slot index: player}. `free` = list of free slot indices."""
    match = {}

    def go(pl, seen):
        for si in free:
            if SLOTS[si] in pl["slots"] and si not in seen:
                seen.add(si)
                if si not in match or go(match[si], seen):
                    match[si] = pl
                    return True
        return False
    for pl in sorted(players, key=lambda x: -x["p"] * x["pts"]):
        go(pl, set())
    return match


def _simulate(assign, starters, bench, draws):
    """assign: {starter id: slot index}. draws: list of dict id -> bool (plays). Returns the mean points and the mean number of starting slots filled by someone who played."""
    allp = {p["i"]: p for p in starters + bench}
    tips = sorted({p["tip"] for p in allp.values()})
    tot, filled = 0.0, 0
    for out in draws:
        slot_of = dict(assign)                         # id -> slot index, everyone currently placed in a starting slot (not yet frozen)
        locked = set()                                 # slot indices frozen by a player whose game started
        pts, nf = 0.0, 0
        for t in tips:
            grp = [pid for pid, p in allp.items() if p["tip"] == t]
            scratched = [pid for pid in grp if not out[pid] and pid in slot_of]
            for pid in scratched:
                slot_of.pop(pid)
            if scratched:
                # news before this tip-off: everyone whose game has not started (this group's players who play, and all later players) can be re-matched to the unfrozen slots
                pool = []
                for p in allp.values():
                    if p["tip"] == t and out[p["i"]]:
                        pool.append(dict(p, p=1.0))
                    elif p["tip"] > t:
                        pool.append(p)
                m = _match(pool, [si for si in range(len(SLOTS)) if si not in locked])
                slot_of = {pl["i"]: si for si, pl in m.items()}
            for pid in grp:
                if pid in slot_of and out[pid]:
                    locked.add(slot_of.pop(pid))
                    pts += allp[pid]["pts"]
                    nf += 1
        tot += pts
        filled += nf
    n = len(draws)
    return tot / n, filled / n


def _draws(starters, bench, n, seed):
    rng = random.Random(seed)
    ids = [p["i"] for p in starters + bench]
    ps = {p["i"]: p["p"] for p in starters + bench}
    return [{i: rng.random() < ps[i] for i in ids} for _ in range(n)]


def _from_names(starters, slots_by_id):
    out = {}
    used = set()
    for p in starters:
        want = slots_by_id[p["i"]]
        for si, s in enumerate(SLOTS):
            if s == want and si not in used:
                out[p["i"]] = si
                used.add(si)
                break
    return out


def best_assignment(starters, bench, current_slots, n=N_SIMS, seed=11):
    """returns (slot names by starter id, info). current_slots: {starter id: slot name} from the plan. info has expected points and filled slots for the original and the chosen
    assignment, and `no_cover`: expected points if nobody can be swapped in after the lineup is set."""
    base = _from_names(starters, current_slots)
    if len(base) < len(starters) or not starters:
        return current_slots, None
    risky = [p for p in starters if p["p"] < 0.985]
    tips = {p["tip"] for p in starters + bench}
    if not risky or len(tips) < 2 or not bench:
        return current_slots, {"changed": False, "reason": "no late news can matter" if not risky else "no spare player or a single tip-off time"}
    draws = _draws(starters, bench, n, seed)
    v0, f0 = _simulate(base, starters, bench, draws)
    v_nocover = sum(p["p"] * p["pts"] for p in starters)
    cur, cur_v, cur_f = dict(base), v0, f0
    byid = {p["i"]: p for p in starters}
    improved = True
    rounds = 0
    while improved and rounds < 3:
        improved, rounds = False, rounds + 1
        ids = list(cur)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                x, y = ids[a], ids[b]
                sx, sy = cur[x], cur[y]
                if SLOTS[sy] not in byid[x]["slots"] or SLOTS[sx] not in byid[y]["slots"] or SLOTS[sx] == SLOTS[sy]:
                    continue
                cand = dict(cur)
                cand[x], cand[y] = sy, sx
                v, f = _simulate(cand, starters, bench, draws)
                if v > cur_v + 1e-9:
                    cur, cur_v, cur_f, improved = cand, v, f, True
    gain = cur_v - v0
    if gain < MIN_GAIN:
        return current_slots, {"changed": False, "points": round(v0, 1), "filled": round(f0, 2), "no_cover": round(v_nocover, 1), "swap_value": round(v0 - v_nocover, 1), "gain": round(gain, 2),
                               "reason": "the plan's own slots are already as good as any alternative"}
    names = {pid: SLOTS[si] for pid, si in cur.items()}
    return names, {"changed": True, "points": round(cur_v, 1), "points_before": round(v0, 1), "filled": round(cur_f, 2), "filled_before": round(f0, 2), "no_cover": round(v_nocover, 1),
                   "swap_value": round(cur_v - v_nocover, 1), "gain": round(gain, 2)}


def cover_for(starters, bench, slots_by_id):
    """for display: for each doubtful starter, which bench players could replace him if he is scratched right before his tip (eligible for his slot, and tip at or after his)"""
    res = {}
    for p in starters:
        if p["p"] >= 0.985:
            continue
        s = slots_by_id.get(p["i"])
        opts = [b for b in bench if b["tip"] >= p["tip"] and (s in b["slots"])]
        res[p["i"]] = [b["i"] for b in sorted(opts, key=lambda b: -b["p"] * b["pts"])[:2]]
    return res

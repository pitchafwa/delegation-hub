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
# chance a scratch is NOT caught before his tip (2026-11-07, Tommy): he stays in his slot, it locks at tip with 0 points (a scratch does not use a start), and only players whose
# games start later can be moved in, to the slots still open. Assumption, not measured: the ruled-out alert and its 35-minute reminder make a miss rare. On a full lineup a miss
# costs the same in any slot; on a part-full night a miss in UT can leave a late replacement with no slot he fits, which is when a position slot is the better place for him.
MISS_P = 0.10


def _match(players, free, limit=None):
    """greedy transversal-matroid matching: players sorted by expected points; returns {slot index: player}. `free` = list of free slot indices.
    limit: at most this many players placed (a swap REPLACES a scratched starter; it does not start extra bench players the plan is resting to save starts)."""
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
        if limit is not None and len(match) >= limit:
            break
        go(pl, set())
    return match


def _simulate(assign, starters, bench, draws):
    """assign: {starter id: slot index}. draws: list of dict id -> bool (plays). Returns the mean points and the mean number of starting slots filled by someone who played."""
    allp = {p["i"]: p for p in starters + bench}
    tips = sorted({p["tip"] for p in allp.values()})
    tot, filled = 0.0, 0
    for out_ in draws:
        out, miss = out_["play"], out_["miss"]
        slot_of = dict(assign)                         # id -> slot index, everyone currently placed in a starting slot (not yet frozen)
        locked = set()                                 # slot indices frozen by a player whose game started
        pts, nf = 0.0, 0
        for t in tips:
            grp = [pid for pid, p in allp.items() if p["tip"] == t]
            scratched = [pid for pid in grp if not out[pid] and pid in slot_of]
            caught = [pid for pid in scratched if not miss[pid]]
            missed = [pid for pid in scratched if miss[pid]]
            if caught:
                # news before this tip-off: everyone whose game has not started (this group's players who play, and all later players) can be re-matched to the unfrozen slots;
                # as many players as were placed before (replacements only). A missed scratch keeps his slot.
                n_placed = len(slot_of) - len(missed)
                for pid in caught:
                    slot_of.pop(pid)
                held = {slot_of[pid] for pid in missed}
                pool = []
                for p in allp.values():
                    if p["tip"] == t and out[p["i"]]:
                        pool.append(dict(p, p=1.0))
                    elif p["tip"] > t:
                        pool.append(p)
                m = _match(pool, [si for si in range(len(SLOTS)) if si not in locked and si not in held], limit=n_placed)
                slot_of = {**{pid: slot_of[pid] for pid in missed}, **{pl["i"]: si for si, pl in m.items()}}
            for pid in grp:
                if pid in slot_of and out[pid]:
                    locked.add(slot_of.pop(pid))
                    pts += allp[pid]["pts"]
                    nf += 1
            if missed:
                # noticed only after his tip: his slot is frozen with 0; later-tipping players can still be moved, one replacement per missed scratch, into the slots still open
                for pid in missed:
                    locked.add(slot_of.pop(pid))
                later = [p for p in allp.values() if p["tip"] > t]
                n_placed = sum(1 for pid in slot_of if allp[pid]["tip"] > t) + len(missed)
                m = _match(later, [si for si in range(len(SLOTS)) if si not in locked], limit=n_placed)
                slot_of = {pl["i"]: si for si, pl in m.items()}
        tot += pts
        filled += nf
    n = len(draws)
    return tot / n, filled / n


def _draws(starters, bench, n, seed):
    rng = random.Random(seed)
    ids = [p["i"] for p in starters + bench]
    ps = {p["i"]: p["p"] for p in starters + bench}
    out = []
    for _ in range(n):
        play = {i: rng.random() < ps[i] for i in ids}
        out.append({"play": play, "miss": {i: rng.random() < MISS_P for i in ids}})
    return out


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
    # tie-break (2026-11-02, Tommy): when two placements score the same, the less certain player takes the more flexible slot (UT > G/F > a position), so news that comes
    # earlier or later than the simulation assumes can still be covered by more bench players
    ids = list(cur)
    for x in ids:
        for y in ids:
            if x == y:
                continue
            sx, sy = cur[x], cur[y]
            if (byid[x]["p"] < byid[y]["p"] - 0.02 and FLEX_RANK[SLOTS[sy]] > FLEX_RANK[SLOTS[sx]]
                    and SLOTS[sy] in byid[x]["slots"] and SLOTS[sx] in byid[y]["slots"]):
                cand = dict(cur)
                cand[x], cand[y] = sy, sx
                v, f = _simulate(cand, starters, bench, draws)
                if v >= cur_v - 1e-6:
                    cur, cur_v, cur_f = cand, max(v, cur_v), f
    gain = cur_v - v0
    moved = any(cur[k] != base[k] for k in cur)
    if gain < MIN_GAIN and not (moved and gain >= -1e-6):
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
        # a two-step swap counts too (2026-11-02, Tommy: Murray at SG "had no cover", but Quickley, unlocked in UT until his 10pm tip, can slide into SG and Wiggins take UT)
        for b in bench:
            if b in opts or b["tip"] < p["tip"]:
                continue
            if any(y is not p and y["tip"] >= p["tip"] and s in y["slots"] and slots_by_id.get(y["i"]) in b["slots"] for y in starters):
                opts.append(b)
        res[p["i"]] = [b["i"] for b in sorted(opts, key=lambda b: -b["p"] * b["pts"])[:2]]
    return res

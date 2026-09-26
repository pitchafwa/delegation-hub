"""Monte Carlo of the rest of the season: playoff odds, seeds, byes, title odds (this league: 12 teams, 6 make the playoffs, seeded by record with no reseeding; seeds 1-2 get a bye;
round 1 is 3v6 and 4v5, semi-finals 1 v winner(4/5) and 2 v winner(3/6), each round one week; 19 regular-season weeks, playoff weeks 20-22).

Team-week score = mu[team, week] - bias + u[team] * len/7 + e * sqrt(len/7)
    mu   forecast points for that matchup (the cap-aware plan blended with real scoring in season)
    u    a team-level shock shared by all remaining weeks (roster changes, management): SD tau  (the forecast error in 2025-26 was 54% team-persistent)
    e    week-level noise: SD sigma_e
Ties in the standings are broken by points for (ESPN's head-to-head tiebreaker is not modelled).  Parameters were chosen on last season's checkpoints (playoff_odds_backtest.py).
"""
import numpy as np

N_DEFAULT = 20000
BYE = 2                   # seeds 1-2
QUAL = 6


def simulate(team_ids, wins, pf, pairs, mu, lens, reg_weeks, playoff_weeks=(20, 21, 22), tau=150.0, sigma_e=175.0, bias=25.0, n=N_DEFAULT, seed=7, focus=None, focus_week=None, shift=None):
    """team_ids: list; wins, pf: dicts id -> current wins / points for; pairs: list of (week, a, b) still to play in the regular season;
       mu: dict (id, week) -> forecast points; lens: dict week -> days in the matchup; reg_weeks: last regular-season week.
       focus / focus_week: return the focus team's result in that week so odds can be conditioned on winning or losing it.
       shift: dict id -> extra points per 7 days added to mu (what-if)."""
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(team_ids)}
    T = len(team_ids)
    weeks = sorted({w for w, _, _ in pairs} | set(playoff_weeks))
    U = rng.normal(0, tau, (n, T))
    S = {}
    for w in weeks:
        L = lens.get(w, 7)
        m = np.array([mu.get((t, w), 0.0) + (shift or {}).get(t, 0.0) * L / 7.0 for t in team_ids])
        S[w] = m[None, :] - bias * L / 7.0 + U * (L / 7.0) + rng.normal(0, sigma_e, (n, T)) * np.sqrt(L / 7.0)
    W = np.tile(np.array([wins.get(t, 0) for t in team_ids], float), (n, 1))
    PF = np.tile(np.array([pf.get(t, 0.0) for t in team_ids], float), (n, 1))
    focus_won = None
    for w, a, b in pairs:
        ia, ib = idx[a], idx[b]
        sa, sb = S[w][:, ia], S[w][:, ib]
        PF[:, ia] += sa
        PF[:, ib] += sb
        W[:, ia] += (sa > sb)
        W[:, ib] += (sb > sa)
        if focus is not None and w == focus_week and focus in (a, b):
            focus_won = (sa > sb) if focus == a else (sb > sa)
    key = W + PF / 1e6
    order = np.argsort(-key, axis=1)                    # order[:, s] = team index holding seed s+1
    seed_of = np.empty_like(order)
    rows = np.arange(n)[:, None]
    seed_of[rows, order] = np.arange(T)[None, :]
    made = seed_of < QUAL
    seed_counts = np.zeros((T, T))
    for s in range(T):
        seed_counts[:, s] = (seed_of == s).mean(axis=0)
    # playoffs
    p1, p2, p3, p4, p5, p6 = (order[:, k] for k in range(6))
    w20, w21, w22 = playoff_weeks
    win = lambda x, y, wk: np.where(S[wk][rows[:, 0], x] >= S[wk][rows[:, 0], y], x, y)
    a36 = win(p3, p6, w20)
    a45 = win(p4, p5, w20)
    f1 = win(p1, a45, w21)
    f2 = win(p2, a36, w21)
    champ = win(f1, f2, w22)
    title = np.zeros(T)
    finalists = np.zeros(T)
    for t in range(T):
        title[t] = (champ == t).mean()
        finalists[t] = ((f1 == t) | (f2 == t)).mean()
    wins_final = W
    out = {"made": made.mean(axis=0), "seed": seed_counts, "bye": (seed_of < BYE).mean(axis=0), "title": title, "final": finalists, "wins": wins_final.mean(axis=0),
           "team_ids": team_ids}
    if focus is not None and focus_won is not None:
        fi = idx[focus]
        out["focus_made_win"] = float(made[focus_won, fi].mean()) if focus_won.any() else None
        out["focus_made_loss"] = float(made[~focus_won, fi].mean()) if (~focus_won).any() else None
        out["focus_bye_win"] = float((seed_of[focus_won, fi] < BYE).mean()) if focus_won.any() else None
        out["focus_bye_loss"] = float((seed_of[~focus_won, fi] < BYE).mean()) if (~focus_won).any() else None
        out["focus_title_win"] = float((champ[focus_won] == fi).mean()) if focus_won.any() else None
        out["focus_title_loss"] = float((champ[~focus_won] == fi).mean()) if (~focus_won).any() else None
        out["focus_p_win"] = float(focus_won.mean())
    out["focus_wins_dist"] = None
    if focus is not None:
        fi = idx[focus]
        dist = {}
        for k in np.unique(W[:, fi]):
            m_ = W[:, fi] == k
            dist[int(k)] = {"p": float(m_.mean()), "made": float(made[m_, fi].mean())}
        out["focus_wins_dist"] = dist
    return out

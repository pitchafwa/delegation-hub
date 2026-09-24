"""Real Kalman filter per stat, DARKO-style: the aging curve lives inside
the state TRANSITION (not bolted on after), process noise accumulates with
elapsed days (continuous time, not discrete game-steps), and observation
noise scales with that game's minutes (a rate-stat observed over more
minutes is a more reliable read on the true per-minute rate).

State: x_t = player's true per-minute rate for the stat.
Transition: x_t = x_{t-1} + age_drift(age) * days_elapsed/365
Observation: raw_count_t = minutes_t * x_t + noise(variance ~ R * minutes_t)

For MIN itself, minutes_t=1 (directly tracking minutes/game as the "stat").
"""
import numpy as np
import pandas as pd


def run_filter_all_players(
    player_ids: np.ndarray, days_since_last: np.ndarray, age_at_game: np.ndarray,
    gain: np.ndarray, observation: np.ndarray,
    Q: float, R: float, peak_age: float, slope_up: float, slope_down: float,
    x0, P0: float = 50.0, return_pre: bool = False,
):
    """Vectorized-by-segment sequential filter. Arrays must already be sorted
    by (player_id, date). Returns posterior mean x AFTER each game (so you
    can snapshot "belief as of this date"); with return_pre=True also
    returns the PRE-update prediction at each game (the honest "what would
    we have called this game before seeing its result" number -- needed for
    the DFS-style next-game validation, since using the posterior would be
    peeking at that game's own outcome).

    x0: either a single float (same starting prior for every player --
    original behavior, still used by every existing caller) OR an
    array/Series the same length as player_ids giving each ROW's player its
    own individualized starting prior (read only at that player's first
    row). Lets a real player-specific prior (e.g. an Output-B rookie
    projection) replace the population-wide average for players it's
    available for, while everyone else still gets the population fallback."""
    n = len(player_ids)
    x_post = np.empty(n)
    x_pre = np.empty(n) if return_pre else None
    # Player boundaries: where player_id changes from the previous row.
    new_player = np.empty(n, dtype=bool)
    new_player[0] = True
    new_player[1:] = player_ids[1:] != player_ids[:-1]

    x0_arr = np.broadcast_to(np.asarray(x0, dtype=float), (n,))

    x, P = x0_arr[0], P0
    for i in range(n):
        if new_player[i]:
            x, P = x0_arr[i], P0

        diff = age_at_game[i] - peak_age
        daily_slope = (slope_up if diff <= 0 else slope_down) / 365.0
        x_pred = x + daily_slope * days_since_last[i]
        P_pred = P + Q * days_since_last[i]

        if return_pre:
            x_pre[i] = x_pred

        H = gain[i]
        if H <= 1e-6:
            x, P = x_pred, P_pred
        else:
            R_eff = R * max(H, 1.0)
            S = H * H * P_pred + R_eff
            K = H * P_pred / S
            innovation = observation[i] - H * x_pred
            x = x_pred + K * innovation
            P = (1 - K * H) * P_pred

        x_post[i] = x

    if return_pre:
        return x_post, x_pre
    return x_post

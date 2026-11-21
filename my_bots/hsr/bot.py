"""hsr: hand-strength bot.

Every decision gets a strength score in [0, 1]:
  - preflop: looked up in data/preflop.json, a table of all 169 starting hands
    (equity vs. a random hand plus its percentile in the strict ordering);
  - flop / turn / river: Monte Carlo equity against random opponent hands.

We continue only when strength beats the price by a street margin:
    strength >= pot_odds + STREET_MARGIN[street] * (1 - aggression)
The margin anneals from wide preflop toward 0 on the river, so we demand a
big edge early and take more risk as the board fills in.

Heads-up preflop uses hu_preflop() instead: the button opens ~80%, the big blind
defends by pot odds and 3-bets ~12%, and at <= PUSH_FOLD_BB effective big blinds
we only shove or fold. Facing an all-in still goes through the equity rule. `aggression` is how
often the opponent has raised/shoved this match: against a maniac its range is
close to random, so equity-vs-random is accurate and we need less margin.
"""
import json
import os
import random  # noqa: F401  (seeded per match, so a seeded match replays exactly)
from time import perf_counter

import eval7

# --- Tunables -----------------------------------------------------------------
# Extra equity demanded on top of pot odds. Anneals as the board grows.
STREET_MARGIN = {"preflop": 0.12, "flop": 0.08, "turn": 0.04, "river": 0.0}

# Preflop gate: only play hands in the top (1 - PREFLOP_MIN_PCT) of the ordering.
# Scaled by (1 - aggression), so it opens up against opponents who shove a lot.
PREFLOP_MIN_PCT = 0.60            # 0.60 = top 40% of hands

# Opponent aggression estimate: (raises + prior_aggr * PRIOR_N) / (actions + PRIOR_N)
PRIOR_AGGR, PRIOR_N = 0.25, 10
# Also anneals: minimum strength to put chips in at all on each street.
STREET_FLOOR = {"preflop": 0.50, "flop": 0.45, "turn": 0.42, "river": 0.40}

# Strength needed to bet/raise for value, and the bet size as a fraction of the pot.
RAISE_STRENGTH = {"preflop": 0.64, "flop": 0.72, "turn": 0.76, "river": 0.80}
RAISE_POT_FRAC = {"preflop": 1.0, "flop": 0.75, "turn": 0.75, "river": 0.75}

# --- Heads-up preflop ranges ----------------------------------------------------
# Ranges are "top X of all 1326 combos", i.e. play when pct >= 1 - X.
HU_OPEN_FRAC = 0.80               # button: open-raise this much when it's folded to us
HU_OPEN_BB = 2.5                  # open size, in big blinds (total)
HU_ISO_FRAC = 0.30                # big blind vs a limp: raise this much, check the rest
HU_ISO_BB = 4.0
HU_3BET_FRAC = 0.12               # vs an open: re-raise this much
HU_3BET_MULT = 3.0                # re-raise to 3x their raise
HU_4BET_FRAC = 0.05               # vs a 3-bet or more: re-raise (usually a shove) this much
HU_RERAISED_DEFEND = 0.75         # vs a 3-bet+, call this share of the normal defend range
# Share of hands to continue with vs a raise, by the pot odds it gives us
# (2.5x open -> 0.30, 3x -> 0.33, 4x -> 0.375, 6x -> 0.42, 10x -> 0.45).
HU_DEFEND = [(0.25, 0.75), (0.30, 0.65), (0.333, 0.58), (0.375, 0.48),
             (0.417, 0.35), (0.45, 0.25), (0.50, 0.12)]
# Short stacks: at or below this many effective big blinds we only shove or fold.
PUSH_FOLD_BB = 12
# Button shove range by effective stack (approximate heads-up Nash).
HU_PUSH = [(3, 0.95), (5, 0.75), (8, 0.62), (10, 0.57), (12, 0.50)]
HU_SHORT_ISO = 0.60               # short BB vs a limp / small raise: shove this share of the push range

MC_TIME_S = 0.6                  # Monte Carlo budget per decision (sandbox has 0.5 CPU)
MC_MAX_ITERS = 4000

TIME_BUDGET_S = 1.95
LIMIT_MARGIN_S = 0.05


# --- Hand evaluation (eval7) --------------------------------------------------
RANK_CHARS = "23456789TJQKA"
DECK_STR = [r + s for r in RANK_CHARS for s in "cdhs"]


def to_card(c):
    return eval7.Card(c)


evaluate = eval7.evaluate                     # 5-7 cards -> int, higher is better

FULL_DECK = [to_card(c) for c in DECK_STR]


def equity(hole, board, n_opp, deadline, max_iters=MC_MAX_ITERS):
    """Monte Carlo share of the pot vs n_opp random hands. Cards are strings."""
    known = set(hole) | set(board)
    deck = [FULL_DECK[i] for i, c in enumerate(DECK_STR) if c not in known]
    h = [to_card(c) for c in hole]
    b0 = [to_card(c) for c in board]
    need = 5 - len(b0)
    draw_n = need + 2 * n_opp
    sample = random.sample

    won = 0.0
    n = 0
    while n < max_iters:
        if (n & 63) == 0 and perf_counter() >= deadline:
            break
        d = sample(deck, draw_n)
        b = b0 + d[:need]
        mine = evaluate(h + b)
        best, ties = -1, 0
        for i in range(n_opp):
            o = evaluate(d[need + 2 * i: need + 2 * i + 2] + b)
            if o > best:
                best, ties = o, 1
            elif o == best:
                ties += 1
        if mine > best:
            won += 1.0
        elif mine == best:
            won += 1.0 / (ties + 1)
        n += 1
    return won / n if n else 0.5


# --- Preflop table (strict ordering of the 169 starting hands) ----------------
def preflop_key(hole):
    """["As","Kh"] -> "AKo", ["Td","Tc"] -> "TT"."""
    (r1, s1), (r2, s2) = hole[0], hole[1]
    if RANK_CHARS.index(r1) < RANK_CHARS.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def _load_preflop():
    for d in (os.environ.get("BOT_DATA_DIR"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")):
        if d:
            p = os.path.join(d, "preflop.json")
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)        # {"AKs": {"equity": .., "pct": ..}, ...}
    return None


PREFLOP = _load_preflop()


# --- Strength + decision ------------------------------------------------------
def n_opponents(gs):
    me = gs.get("seat_to_act")
    live = [p for p in gs.get("players", [])
            if p.get("seat") != me and p.get("state") in ("active", "all_in")]
    return max(1, len(live))


def strength(gs, deadline):
    """(equity, percentile or None). Percentile is only defined preflop."""
    hole, board = gs["your_cards"], gs.get("community_cards") or []
    n_opp = n_opponents(gs)
    mc_deadline = min(deadline, perf_counter() + MC_TIME_S)
    if not board and PREFLOP and n_opp == 1:
        row = PREFLOP[preflop_key(hole)]
        return row["equity"], row["pct"]
    pct = PREFLOP[preflop_key(hole)]["pct"] if (not board and PREFLOP) else None
    return equity(hole, board, n_opp, mc_deadline), pct


def aggression(gs):
    """Share of the opponents' voluntary actions this match that were raises or shoves."""
    me = gs.get("your_bot_id")
    agg = n = 0
    for e in gs.get("match_action_log") or []:
        a = e.get("action")
        if e.get("bot_id") == me or a not in ("fold", "check", "call", "raise", "all_in"):
            continue
        n += 1
        agg += a in ("raise", "all_in")
    return (agg + PRIOR_AGGR * PRIOR_N) / (n + PRIOR_N)


def _safe_action(gs):
    return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


def _raise(gs, street):
    """Bet RAISE_POT_FRAC of the pot on top of the current bet; shove if that's most of our stack."""
    max_total = gs["your_stack"] + gs.get("your_bet_this_street", 0)
    target = gs.get("current_bet", 0) + int(RAISE_POT_FRAC[street] * (gs["pot"] + gs.get("amount_owed", 0)))
    target = max(target, gs.get("min_raise_to", 0))
    if target >= 0.8 * max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target}


def _raise_to(gs, total):
    """Raise to `total` chips on this street; shove if that's most of our stack."""
    max_total = gs["your_stack"] + gs.get("your_bet_this_street", 0)
    total = max(int(total), gs.get("min_raise_to", 0))
    if total >= 0.8 * max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": total}


def _interp(table, x):
    """Piecewise-linear lookup in [(x, y), ...] sorted by x, clamped at the ends."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def hu_preflop(gs, pct, pot_odds, aggr):
    """Heads-up preflop by position and stack depth. Returns an action, or None to
    fall through to the generic equity logic (used when we face an all-in)."""
    me = gs["seat_to_act"]
    bb = gs["blinds"]["bb"] or 1
    opp = next((p for p in gs["players"] if p["seat"] != me and p["state"] in ("active", "all_in")), None)
    if opp is None:
        return None
    my_total = gs["your_stack"] + gs.get("your_bet_this_street", 0)
    eff_bb = min(my_total, opp["stack"] + opp.get("bet_this_street", 0)) / bb
    owed = gs.get("amount_owed", 0)
    raises = sum(1 for a in gs.get("action_log") or []
                 if a.get("street") == "preflop" and a.get("action") in ("raise", "all_in"))
    on_button = me == gs.get("button_seat")
    top = lambda frac: pct >= 1.0 - min(frac, 1.0)

    # A call that commits us (opponent all-in, or most of our stack): decide on equity.
    if opp["state"] == "all_in" or owed >= 0.5 * gs["your_stack"]:
        return None

    if eff_bb <= PUSH_FOLD_BB:                            # short: shove or fold
        push = _interp(HU_PUSH, eff_bb) * (0.85 + 0.6 * aggr)
        if gs.get("can_check") or raises:                 # BB vs limp, or vs a small raise
            if top(push * HU_SHORT_ISO):
                return {"action": "all_in"}
            return _safe_action(gs)
        return {"action": "all_in"} if top(push) else {"action": "fold"}

    if raises == 0:
        if on_button and not gs.get("can_check"):         # folded to us: open or fold
            return _raise_to(gs, HU_OPEN_BB * bb) if top(HU_OPEN_FRAC) else {"action": "fold"}
        if gs.get("can_check"):                           # BB vs a limp
            return _raise_to(gs, HU_ISO_BB * bb) if top(HU_ISO_FRAC) else {"action": "check"}
        return None

    defend = _interp(HU_DEFEND, pot_odds) * min(max(0.8 + 0.8 * aggr, 0.8), 1.4)
    if raises == 1:                                       # facing an open
        if top(HU_3BET_FRAC):
            return _raise_to(gs, HU_3BET_MULT * gs.get("current_bet", 0))
        return {"action": "call"} if top(defend) else {"action": "fold"}
    if top(HU_4BET_FRAC):                                 # facing a 3-bet or more
        return {"action": "all_in"}
    return {"action": "call"} if top(defend * HU_RERAISED_DEFEND) else {"action": "fold"}


def think(gs, deadline):
    street = gs["street"]
    eq, pct = strength(gs, deadline)
    owed = gs.get("amount_owed", 0)
    pot_odds = owed / (gs["pot"] + owed) if owed else 0.0
    aggr = aggression(gs)
    loosen = 1.0 - aggr
    required = max(STREET_FLOOR[street], pot_odds + STREET_MARGIN[street] * loosen)

    info = {"equity": round(eq, 3), "pct": pct, "required": round(required, 3),
            "loosen": round(loosen, 2)}

    hu = None
    if pct is not None and n_opponents(gs) == 1:
        hu = hu_preflop(gs, pct, pot_odds, aggr)
    if hu is not None:
        act = hu
        info["hu_preflop"] = True
    elif pct is not None and owed and n_opponents(gs) == 1 and eq >= required:
        act = {"action": "call"}                          # HU, facing an all-in: equity decides
    elif pct is not None and pct < PREFLOP_MIN_PCT * loosen:  # not a strong enough starting hand
        act = _safe_action(gs)
    elif eq >= RAISE_STRENGTH[street] and eq >= required:
        act = _raise(gs, street)
    elif gs.get("can_check"):
        act = {"action": "check"}
    elif eq >= required:
        act = {"action": "call"}
    else:
        act = {"action": "fold"}
    act["debug"] = info
    return act


# --- Entry point --------------------------------------------------------------
def _budget_s(gs):
    limit_ms = gs.get("time_limit_ms")
    if limit_ms and limit_ms / 1000 - LIMIT_MARGIN_S < TIME_BUDGET_S:
        return max(limit_ms / 1000 - LIMIT_MARGIN_S, 0.0)
    return TIME_BUDGET_S


def decide(game_state: dict) -> dict:
    start = perf_counter()
    deadline = start + _budget_s(game_state) - 0.2       # leave headroom for overhead

    if game_state.get("type") == "warmup":
        return {"action": "check"}

    try:
        action = think(game_state, deadline)
    except Exception as e:
        print(f"think() failed: {e!r}")
        action = None

    if not isinstance(action, dict) or "action" not in action:
        action = _safe_action(game_state)

    action.setdefault("debug", {})["think_ms"] = round((perf_counter() - start) * 1000, 1)
    return action

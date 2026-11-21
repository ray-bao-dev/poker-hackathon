"""Bluff bots: a solid hand-strength core with a tunable bluffing layer.

Five folders share this exact file; only the PROFILE line differs:
  bluff_preflop   loose opens, light 3-bets and 4-bet shoves preflop
  bluff_flop      fires the flop with air, bluff-raises flop bets
  bluff_turn      checks back / gives up flops, then attacks the turn
  bluff_river     plays honest until the river, then over-bets air
  bluff_balanced  moderate bluffs on every street, prefers semi-bluffs (draws),
                  barrels when it was the aggressor, and bluffs less against
                  opponents who don't fold

The value side (all profiles) is "by the book": preflop ranges by position and
stack depth, postflop Monte Carlo equity vs a random hand, call by pot odds
plus a margin, and bet/raise for value with strong hands.

A "bluff" here means betting/raising postflop with < 40% equity vs a random hand,
which is how the Biggest Bluffer prize counts it.
"""
import json
import os
import random  # seeded per match, so a seeded match replays exactly
from time import perf_counter

import eval7

PROFILE = "balanced"

# --- Bluff profiles -----------------------------------------------------------
# bluff:       chance to bet air when checked to (per street)
# bluff_raise: chance to raise air when facing a bet (per street)
# barrel:      extra bluff chance if we were the last aggressor on the previous street
# size:        bluff size as a fraction of the pot (per street)
# pf_open:     button open frequency (share of hands)
# pf_3bet_bluff / pf_4bet_bluff: chance to re-raise light with a non-value hand
# pf_push_mult: widen short-stack shove ranges by this factor
# adapt:       0..1, how much to scale bluffs down vs. opponents who rarely fold
# semi_only:   on flop/turn, mostly bluff with draws (equity 0.28-0.40)
PROFILES = {
    "preflop": dict(
        bluff={"flop": 0.10, "turn": 0.05, "river": 0.05},
        bluff_raise={"flop": 0.0, "turn": 0.0, "river": 0.0},
        barrel=0.10, size={"flop": 0.6, "turn": 0.7, "river": 0.75},
        pf_open=1.00, pf_iso_bluff=0.45, pf_3bet_bluff=0.40, pf_4bet_bluff=0.20,
        pf_push_mult=1.35, adapt=0.3, semi_only=False),
    "flop": dict(
        bluff={"flop": 0.70, "turn": 0.10, "river": 0.05},
        bluff_raise={"flop": 0.30, "turn": 0.0, "river": 0.0},
        barrel=0.15, size={"flop": 0.66, "turn": 0.7, "river": 0.75},
        pf_open=0.80, pf_iso_bluff=0.05, pf_3bet_bluff=0.04, pf_4bet_bluff=0.0,
        pf_push_mult=1.0, adapt=0.3, semi_only=False),
    "turn": dict(
        bluff={"flop": 0.10, "turn": 0.70, "river": 0.10},
        bluff_raise={"flop": 0.0, "turn": 0.25, "river": 0.0},
        barrel=0.15, size={"flop": 0.5, "turn": 0.8, "river": 0.75},
        pf_open=0.80, pf_iso_bluff=0.05, pf_3bet_bluff=0.04, pf_4bet_bluff=0.0,
        pf_push_mult=1.0, adapt=0.3, semi_only=False),
    "river": dict(
        bluff={"flop": 0.05, "turn": 0.10, "river": 0.75},
        bluff_raise={"flop": 0.0, "turn": 0.0, "river": 0.25},
        barrel=0.10, size={"flop": 0.5, "turn": 0.6, "river": 1.2},
        pf_open=0.80, pf_iso_bluff=0.05, pf_3bet_bluff=0.04, pf_4bet_bluff=0.0,
        pf_push_mult=1.0, adapt=0.3, semi_only=False),
    "balanced": dict(
        bluff={"flop": 0.35, "turn": 0.25, "river": 0.20},
        bluff_raise={"flop": 0.08, "turn": 0.05, "river": 0.04},
        barrel=0.25, size={"flop": 0.6, "turn": 0.7, "river": 0.8},
        pf_open=0.85, pf_iso_bluff=0.15, pf_3bet_bluff=0.10, pf_4bet_bluff=0.02,
        pf_push_mult=1.05, adapt=1.0, semi_only=True),
}
P = PROFILES[PROFILE]

# --- Value tunables (shared) --------------------------------------------------
STREET_MARGIN = {"flop": 0.05, "turn": 0.03, "river": 0.0}   # equity over pot odds to call
VALUE_BET = {"flop": 0.68, "turn": 0.72, "river": 0.76}      # equity to bet when checked to
VALUE_RAISE = {"flop": 0.78, "turn": 0.82, "river": 0.86}    # equity to raise a bet
VALUE_SIZE = {"flop": 0.66, "turn": 0.7, "river": 0.75}
BLUFF_MAX_EQ = 0.40              # below this, a bet counts as a bluff
SEMI_MIN_EQ = 0.28               # flop/turn "draw" zone for semi-bluffs: [0.28, 0.40)

# Preflop (heads-up), ranges as "top X of hands": play when pct >= 1 - X
OPEN_BB, ISO_BB, THREEBET_MULT = 2.5, 4.0, 3.0
ISO_FRAC, THREEBET_FRAC, FOURBET_FRAC = 0.30, 0.10, 0.05
DEFEND = [(0.25, 0.75), (0.30, 0.65), (0.333, 0.58), (0.375, 0.48),
          (0.417, 0.35), (0.45, 0.25), (0.50, 0.12)]   # pot odds -> share to continue
PUSH_FOLD_BB = 12
PUSH = [(3, 0.95), (5, 0.75), (8, 0.62), (10, 0.57), (12, 0.50)]  # eff BB -> shove share

# Opponent fold-to-bet prior: (folds + PRIOR_FOLD*N) / (faced + N)
PRIOR_FOLD, PRIOR_N = 0.45, 8

MC_TIME_S, MC_MAX_ITERS = 0.5, 3000
TIME_BUDGET_S, LIMIT_MARGIN_S = 1.9, 0.05

# --- Equity -------------------------------------------------------------------
RANKS = "23456789TJQKA"
DECK_STR = [r + s for r in RANKS for s in "cdhs"]
FULL_DECK = [eval7.Card(c) for c in DECK_STR]
evaluate = eval7.evaluate


def equity(hole, board, deadline):
    """Monte Carlo equity vs one random hand."""
    known = set(hole) | set(board)
    deck = [FULL_DECK[i] for i, c in enumerate(DECK_STR) if c not in known]
    h = [eval7.Card(c) for c in hole]
    b0 = [eval7.Card(c) for c in board]
    need = 5 - len(b0)
    won = n = 0.0
    while n < MC_MAX_ITERS:
        if (int(n) & 63) == 0 and perf_counter() >= deadline:
            break
        d = random.sample(deck, need + 2)
        b = b0 + d[:need]
        mine, opp = evaluate(h + b), evaluate(d[need:] + b)
        won += 1.0 if mine > opp else 0.5 if mine == opp else 0.0
        n += 1
    return won / n if n else 0.5


def preflop_key(hole):
    (r1, s1), (r2, s2) = hole[0], hole[1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2 = r2, r1
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


def preflop_strength(hole):
    """(equity, percentile). Falls back to a crude formula if the table is missing."""
    if PREFLOP:
        row = PREFLOP[preflop_key(hole)]
        return row["equity"], row["pct"]
    a, b = sorted((RANKS.index(hole[0][0]), RANKS.index(hole[1][0])), reverse=True)
    score = (a + b) / 24 + (0.25 if a == b else 0) + (0.05 if hole[0][1] == hole[1][1] else 0)
    pct = min(max(score - 0.1, 0.0), 1.0)
    return 0.35 + 0.5 * pct, pct


# --- Helpers ------------------------------------------------------------------
def _interp(table, x):
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def _safe(gs):
    return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


def _raise_to(gs, total):
    max_total = gs["your_stack"] + gs.get("your_bet_this_street", 0)
    total = max(int(total), gs.get("min_raise_to", 0))
    if total >= 0.8 * max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": total}


def _bet_pot(gs, frac):
    """Bet/raise `frac` of the pot (after calling) on top of the current bet."""
    owed = gs.get("amount_owed", 0)
    return _raise_to(gs, gs.get("current_bet", 0) + frac * (gs["pot"] + owed))


def _opponent(gs):
    me = gs["seat_to_act"]
    return next((p for p in gs["players"]
                 if p["seat"] != me and p["state"] in ("active", "all_in")), None)


VOLUNTARY = ("fold", "check", "call", "raise", "all_in")

# Opponent stats over the whole match (match_action_log only keeps 200 entries).
_done_hands = set()
_stats = {"faced": 0, "folded": 0}


def _update_stats(gs):
    """Count how often the opponent folds when we bet or raise into them.
    Each finished hand is counted once; the oldest hand in a full log window is
    skipped because its start may have been cut off."""
    me = gs.get("your_bot_id")
    log = gs.get("match_action_log") or []
    cur = gs.get("hand_num")
    hands = {}
    for e in log:
        hands.setdefault(e.get("hand_num"), []).append(e)
    oldest = min((h for h in hands if h is not None), default=None)
    for h, entries in hands.items():
        if h is None or h == cur or h in _done_hands:
            continue
        if h == oldest and len(log) >= 200 and h != 0:
            continue
        _done_hands.add(h)
        prev = None
        for e in entries:
            if e.get("action") not in VOLUNTARY:
                continue
            if (prev is not None and prev.get("street") == e.get("street")
                    and prev.get("bot_id") == me and e.get("bot_id") != me
                    and prev.get("action") in ("raise", "all_in")):
                _stats["faced"] += 1
                _stats["folded"] += e.get("action") == "fold"
            prev = e


def fold_rate():
    return (_stats["folded"] + PRIOR_FOLD * PRIOR_N) / (_stats["faced"] + PRIOR_N)


def bluff_scale():
    """1.0 vs a normal folder; down toward ~0.2 vs a calling station; up to 1.5 vs a nit."""
    raw = min(max(fold_rate() / PRIOR_FOLD, 0.2), 1.5)
    return 1.0 + P["adapt"] * (raw - 1.0)


def we_were_aggressor(gs, street):
    """True if our action was the last bet/raise on `street` this hand."""
    me = gs.get("your_bot_id")
    last = None
    for a in gs.get("action_log") or []:
        if a.get("street") == street and a.get("action") in ("raise", "all_in"):
            last = a.get("bot_id")
    return last == me


PREV_STREET = {"flop": "preflop", "turn": "flop", "river": "turn"}


# --- Preflop ------------------------------------------------------------------
def preflop(gs):
    eq, pct = preflop_strength(gs["your_cards"])
    bb = gs["blinds"]["bb"] or 1
    owed = gs.get("amount_owed", 0)
    pot_odds = owed / (gs["pot"] + owed) if owed else 0.0
    opp = _opponent(gs)
    if opp is None:
        return _safe(gs), {}
    my_total = gs["your_stack"] + gs.get("your_bet_this_street", 0)
    eff_bb = min(my_total, opp["stack"] + opp.get("bet_this_street", 0)) / bb
    raises = sum(1 for a in gs.get("action_log") or []
                 if a.get("street") == "preflop" and a.get("action") in ("raise", "all_in"))
    on_button = gs["seat_to_act"] == gs.get("button_seat")
    top = lambda frac: pct >= 1.0 - min(frac, 1.0)
    roll = random.random()
    info = {"pct": pct, "eff_bb": round(eff_bb, 1)}

    # Committing decision (facing a shove or a huge bet): pure equity vs pot odds.
    if opp["state"] == "all_in" or owed >= 0.5 * gs["your_stack"]:
        return ({"action": "call"} if eq >= pot_odds + 0.02 else {"action": "fold"}), info

    if eff_bb <= PUSH_FOLD_BB:
        push = _interp(PUSH, eff_bb) * P["pf_push_mult"]
        if gs.get("can_check") or raises:
            push *= 0.6
        if top(push):
            return {"action": "all_in"}, info
        return _safe(gs), info

    if raises == 0:
        if on_button and not gs.get("can_check"):          # folded to us
            if top(P["pf_open"]):
                return _raise_to(gs, OPEN_BB * bb), info
            return {"action": "fold"}, info
        if gs.get("can_check"):                             # BB vs a limp
            if top(ISO_FRAC) or roll < P["pf_iso_bluff"]:
                info["bluff"] = not top(ISO_FRAC)
                return _raise_to(gs, ISO_BB * bb), info
            return {"action": "check"}, info

    defend = _interp(DEFEND, pot_odds)
    if raises <= 1:                                         # facing an open
        if top(THREEBET_FRAC):
            return _raise_to(gs, THREEBET_MULT * gs.get("current_bet", 0)), info
        if roll < P["pf_3bet_bluff"] and top(0.85):         # light 3-bet, skip pure trash
            info["bluff"] = True
            return _raise_to(gs, THREEBET_MULT * gs.get("current_bet", 0)), info
        return ({"action": "call"} if top(defend) else {"action": "fold"}), info

    # Facing a 3-bet or more
    if top(FOURBET_FRAC):
        return {"action": "all_in"}, info
    if roll < P["pf_4bet_bluff"] and top(0.6):
        info["bluff"] = True
        return {"action": "all_in"}, info
    return ({"action": "call"} if top(defend * 0.7) else {"action": "fold"}), info


# --- Postflop -----------------------------------------------------------------
def postflop(gs, deadline):
    street = gs["street"]
    eq = equity(gs["your_cards"], gs["community_cards"], min(deadline, perf_counter() + MC_TIME_S))
    owed = gs.get("amount_owed", 0)
    pot_odds = owed / (gs["pot"] + owed) if owed else 0.0
    scale = bluff_scale()
    info = {"equity": round(eq, 3), "fold_rate": round(fold_rate(), 2), "scale": round(scale, 2)}

    # How bluffable is this hand? Semi-bluff profiles prefer draws on flop/turn.
    is_air = eq < BLUFF_MAX_EQ
    weight = 1.0
    if P["semi_only"] and street != "river":
        weight = 1.0 if eq >= SEMI_MIN_EQ else 0.35
    roll = random.random()

    if gs.get("can_check"):
        if eq >= VALUE_BET[street]:
            return _bet_pot(gs, VALUE_SIZE[street]), info
        if is_air:
            p = P["bluff"][street]
            if we_were_aggressor(gs, PREV_STREET[street]):
                p += P["barrel"]
            p = min(p * weight * scale, 0.95)
            info["bluff_p"] = round(p, 2)
            if roll < p:
                info["bluff"] = True
                return _bet_pot(gs, P["size"][street]), info
        return {"action": "check"}, info

    # Facing a bet
    if eq >= VALUE_RAISE[street]:
        return _bet_pot(gs, VALUE_SIZE[street]), info
    if eq >= pot_odds + STREET_MARGIN[street]:
        return {"action": "call"}, info
    # Bluff-raise only against bets that aren't already huge relative to our stack.
    if is_air and owed <= 0.35 * gs["your_stack"]:
        p = min(P["bluff_raise"][street] * weight * scale, 0.9)
        info["bluff_p"] = round(p, 2)
        if roll < p:
            info["bluff"] = True
            return _bet_pot(gs, max(P["size"][street], 0.75)), info
    return {"action": "fold"}, info


# --- Entry point --------------------------------------------------------------
def _budget_s(gs):
    limit_ms = gs.get("time_limit_ms")
    if limit_ms and limit_ms / 1000 - LIMIT_MARGIN_S < TIME_BUDGET_S:
        return max(limit_ms / 1000 - LIMIT_MARGIN_S, 0.0)
    return TIME_BUDGET_S


def decide(game_state: dict) -> dict:
    start = perf_counter()
    if game_state.get("type") == "warmup":
        return {"action": "check"}
    deadline = start + _budget_s(game_state) - 0.2
    try:
        _update_stats(game_state)
        if game_state["street"] == "preflop":
            action, info = preflop(game_state)
        else:
            action, info = postflop(game_state, deadline)
    except Exception as e:
        print(f"decide() failed: {e!r}")
        action, info = _safe(game_state), {}
    info["profile"] = PROFILE
    info["think_ms"] = round((perf_counter() - start) * 1000, 1)
    action["debug"] = info
    return action

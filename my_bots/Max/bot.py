"""aggregate_best_v2: aggregate_best plus bluff-raising when facing a postflop bet.

New in v2 (off switch: BLUFF_RAISE = False):
  6. Facing a postflop bet with a hand below the value threshold, compare the EV
     of raising with the EV of calling (or folding):
         EV_raise = f * P + (1 - f) * (eq_c * (P + 2R - O) - R)
         EV_call  = eq * (P + O) - O
     P = pot now, O = amount owed, R = chips the raise adds, f = how often they
     fold when they bet and we raise (new stat, shrunk toward BLUFF_RAISE_PRIOR x
     break-even with BLUFF_RAISE_K), eq_c = eq * CALLED_EQ_MULT (the range that
     calls a raise is stronger). Raise when EV_raise beats max(EV_call, 0) by
     BLUFF_RAISE_MARGIN of the pot. Draws have more eq_c, so they raise first;
     against bots that fold to raises f climbs and almost everything raises,
     against calling stations f falls and it stops. At most one bluff-raise per
     street, never with BLUFF_RAISE_MAX_STACK of our stack or more.

From aggregate_best: hsr_modified_v4 plus the lessons from the v2-v4 / CFR / neural_net runs.

Kept from v4:
  - Equity is Monte Carlo against the hands the opponent can still hold. Each
    raise/shove they make this hand keeps the top NARROW_SOFT + (1 - NARROW_SOFT)
    * rate of their range, where `rate` is how often they bet/raise when nobody
    has bet yet on the street (preflop and postflop separately).
  - Call when equity >= pot odds, plus DEEP_EDGE when the call commits most of
    our stack while deep; bet/raise for value above RAISE_STRENGTH.
  - Every opponent stat is shrunk toward a safe default by sample size
    (estimate()), over the whole match (update_history() keeps every hand, since
    the engine only sends the last 200 actions).
  - Postflop sizes are drawn from BET_SIZE_RANGE for value and bluffs alike.
  - Balanced bluffing: s / (1 + s) bluffs per value bet, scaled by how much they
    fold vs break-even, converted with our own value share.

New here (each has an off switch, see the tunables):
  1. Their postflop range is ranked by equity (average rank over RANK_RUNOUTS
     shared runouts), not by current made hand, so draws and semi-bluffs stay
     in a betting range. Diagnosed leak: v4 underrated its hand vs bettors
     (est. 0.47 vs 0.60 won at showdown) and over-folded (-470 bb in 10 matches).
  2. Showdown calibration: when they show a hand they bet postflop with, compare
     its river strength with what our range model predicted and learn an
     exponent `alpha` on the range fraction (frac ** alpha), shrunk toward 1
     (= the model as is) with CALIB_K. Bets weaker than modelled -> narrow less.
  3. Small blind, unraised: raise to SB_OPEN_BB or fold; never limp (limps did
     no better than folding in the logs; standard heads-up play).
  4. Short stacks: at <= PUSH_FOLD_BB effective big blinds, shove or fold
     preflop with standard heads-up push ranges (HU_PUSH).
  5. Only hands with no pair (no showdown value) bluff.
"""
import json
import math
import os
import random  # noqa: F401  (seeded per match, so a seeded match replays exactly)
from time import perf_counter

import eval7

# --- Tunables -----------------------------------------------------------------
# Opponent stats: estimate = (count + default * K) / (chances + K).
OPEN_DEFAULT, OPEN_K = 0.4, 20    # how often they bet/raise when nobody has yet (per phase)
FOLD_K = 15                       # fold to our postflop bets; default = break-even for the size
# How hard each raise narrows their range: keep the top NARROW_SOFT + (1 - NARROW_SOFT) * rate.
NARROW_SOFT = 0.5
MIN_RANGE = 0.10                  # their range never narrows below this share of hands

# 1. Postflop range ranking by equity: shared runouts per decision (0 = rank by made hand, as v4).
RANK_RUNOUTS = 20
# 2. Showdown calibration of the narrowing: K in showdowns (0 = off), and alpha's bounds.
CALIB_K = 8
ALPHA_BOUNDS = (0.25, 2.0)

# Commit guard: extra equity needed to call off most of our stack while deep.
DEEP_EDGE = 0.10
DEEP_BB = 20                      # "deep" = each player has > this many big blinds in the pot after calling

# Strength needed to bet/raise for value.
RAISE_STRENGTH = {"preflop": 0.64, "flop": 0.72, "turn": 0.76, "river": 0.80}
PREFLOP_RAISE_FRAC = 1.0          # preflop raise (not the SB open): this fraction of the pot on top
BET_SIZE_RANGE = (0.50, 0.80)     # postflop bet/raise size, fraction of the pot, drawn uniformly

# 3. Small blind, unraised: fold below SB_FOLD_PCT, otherwise raise to SB_OPEN_BB (0 = limp, as v4).
SB_FOLD_PCT = 0.30
SB_OPEN_BB = 2.5
# 4. Short stacks (0 = off): shove ranges by effective stack, as a share of hands
#    (approximate heads-up Nash); the big blind vs a limp shoves HU_SHORT_ISO of that range.
PUSH_FOLD_BB = 12
HU_PUSH = [(3, 0.95), (5, 0.75), (8, 0.62), (10, 0.57), (12, 0.50)]
HU_SHORT_ISO = 0.60

# Bluffing (postflop, no bet yet on the street, heads-up).
BLUFF_MAX_EQ = 0.35               # only hands this weak vs their range bluff
BLUFF_NO_PAIR = True              # 5. ...and only with no pair (no showdown value)
BLUFF_SCALE = 1.0                 # multiplies the bluff probability; 0 turns bluffing off
BLUFF_MAX_STACK = 0.5             # never bluff with a bet of this share of our stack or more
VALUE_SHARE_DEFAULT, VALUE_SHARE_K = 0.5, 10

# 6. Bluff-raising (postflop, facing a bet, heads-up).
BLUFF_RAISE = True                # False = never raise without a value hand (as aggregate_best)
BLUFF_RAISE_PRIOR = 0.8           # assumed fold-to-raise rate before data, as a share of break-even
BLUFF_RAISE_K = 10                # evidence weight for that stat
CALLED_EQ_MULT = 0.75             # our equity vs the range that calls a raise, as a share of eq
BLUFF_RAISE_MARGIN = 0.05         # EV edge over calling/folding needed, as a share of the pot
BLUFF_RAISE_MAX_STACK = 0.35      # never bluff-raise with this share of our stack or more

MC_TIME_S = 0.6                   # Monte Carlo budget per decision (sandbox has 0.5 CPU)
MC_MAX_ITERS = 4000

TIME_BUDGET_S = 1.95
LIMIT_MARGIN_S = 0.05

# --- Optional overrides (used by tune_aggregate.py) ---------------------------
# data/params.json may set any of these names; RAISE_SHIFT moves every
# RAISE_STRENGTH threshold by the same amount. Without the file nothing changes.
TUNABLE = ("NARROW_SOFT", "DEEP_EDGE", "SB_FOLD_PCT", "SB_OPEN_BB", "BET_SIZE_RANGE",
           "BLUFF_SCALE", "BLUFF_MAX_EQ", "PUSH_FOLD_BB", "RAISE_SHIFT", "RANK_RUNOUTS", "CALIB_K",
           "BLUFF_RAISE", "BLUFF_RAISE_PRIOR", "BLUFF_RAISE_K", "CALLED_EQ_MULT", "BLUFF_RAISE_MARGIN")


def _apply_overrides():
    global RAISE_STRENGTH
    for d in (os.environ.get("BOT_DATA_DIR"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")):
        p = os.path.join(d, "params.json") if d else None
        if p and os.path.exists(p):
            with open(p) as f:
                params = json.load(f)
            for k, v in params.items():
                if k == "RAISE_SHIFT":
                    RAISE_STRENGTH = {s: t + float(v) for s, t in RAISE_STRENGTH.items()}
                elif k in TUNABLE:
                    globals()[k] = tuple(v) if isinstance(v, list) else v
            return params
    return {}


PARAMS = _apply_overrides()


# --- Cards --------------------------------------------------------------------
RANK_CHARS = "23456789TJQKA"
DECK_STR = [r + s for r in RANK_CHARS for s in "cdhs"]
IDX = {c: i for i, c in enumerate(DECK_STR)}
FULL_DECK = [eval7.Card(c) for c in DECK_STR]
evaluate = eval7.evaluate                     # 5-7 cards -> int, higher is better


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
ALL_COMBOS = [(i, j) for i in range(52) for j in range(i + 1, 52)]
# Preflop percentile of every one of the 1326 two-card combos (0 if the table is missing).
COMBO_PCT = {(i, j): (PREFLOP[preflop_key([DECK_STR[i], DECK_STR[j]])]["pct"] if PREFLOP else 0.0)
             for i, j in ALL_COMBOS}


# --- Match-level memory -------------------------------------------------------
VOLUNTARY = ("fold", "check", "call", "raise", "all_in")
BETS = ("raise", "all_in")

# The engine only sends the last MATCH_LOG_MAX_ENTRIES (200) actions, about 25 hands,
# so we keep every hand we have seen in module state (the bot process stays loaded
# for the whole match) and reset it when a new match starts.
_HISTORY = {}                      # hand_num -> that hand's log entries, in order
_BOARDS = {}                       # hand_num -> the longest board we saw in that hand
_CALIB = {}                        # hand_num -> calibration ratio from its showdown, or None
_HIST_KEY = {"match": None, "bot": None, "hand": -1}
_SELF = {"value": 0, "weak": 0}    # our unbet postflop spots this match, by hand class


def _match_id(gs):
    """'sim-1_h0042' -> 'sim-1'. None if the engine didn't send a hand_id."""
    hid = gs.get("hand_id")
    return hid.rsplit("_h", 1)[0] if isinstance(hid, str) and "_h" in hid else None


def update_history(gs):
    """Merge this decision's match_action_log into _HISTORY; reset on a new match.
    A hand is replaced only by a longer copy, since the oldest hand in the window
    can arrive with its start cut off."""
    me, hand, match = gs.get("your_bot_id"), gs.get("hand_num", 0), _match_id(gs)
    if match != _HIST_KEY["match"] or me != _HIST_KEY["bot"] or hand < _HIST_KEY["hand"]:
        for store in (_HISTORY, _BOARDS, _CALIB):
            store.clear()
        _SELF.update(value=0, weak=0)
        _HIST_KEY.update(match=match, bot=me)
    _HIST_KEY["hand"] = hand
    board = list(gs.get("community_cards") or [])
    if len(board) >= len(_BOARDS.get(hand, ())):
        _BOARDS[hand] = board
    by_hand = {}
    for e in gs.get("match_action_log") or []:
        by_hand.setdefault(e.get("hand_num"), []).append(e)
    for h, entries in by_hand.items():
        if len(entries) >= len(_HISTORY.get(h, ())):
            _HISTORY[h] = entries


def match_log():
    """Every stored entry, oldest hand first."""
    for h in sorted(_HISTORY, key=lambda x: (x is None, x if x is not None else 0)):
        yield from _HISTORY[h]


def estimate(count, chances, default, k):
    """Observed rate shrunk toward `default`: the data gets weight chances / (chances + k)."""
    return (count + default * k) / (chances + k)


def _phase(street):
    return "pre" if street == "preflop" else "post"


def opponent_counts(gs):
    """Raw [count, chances] over the whole match:
      pre / post: they bet/raise when nobody had bet yet on the street;
      fold:       they folded facing our postflop bet/raise;
      fold_raise: they folded when they had bet postflop and we raised over it."""
    me = gs.get("your_bot_id")
    c = {"pre": [0, 0], "post": [0, 0], "fold": [0, 0], "fold_raise": [0, 0]}
    spot, opened, ours = None, False, False   # (hand, street); someone bet; the open bet is ours
    opp_bet, over = False, False              # they bet this street; our last bet raised over theirs
    for e in match_log():
        a = e.get("action")
        if a not in VOLUNTARY:
            continue
        street = e.get("street")
        key = (e.get("hand_num"), street)
        if key != spot:
            spot, opened, ours, opp_bet, over = key, False, False, False, False
        mine = e.get("bot_id") == me
        if not mine:
            if not opened:
                s = c[_phase(street)]
                s[1] += 1
                s[0] += a in BETS
            if ours and street != "preflop":
                c["fold"][1] += 1
                c["fold"][0] += a == "fold"
                if over:
                    c["fold_raise"][1] += 1
                    c["fold_raise"][0] += a == "fold"
        if a in BETS:
            if mine:
                over = opp_bet
            else:
                opp_bet, over = True, False
            opened, ours = True, mine
    return c


def raw_range(entries, me, rates):
    """Range fraction implied by the opponent's raises in one hand's entries, before calibration."""
    frac = 1.0
    for a in entries:
        if a.get("bot_id") != me and a.get("action") in BETS:
            frac *= NARROW_SOFT + (1 - NARROW_SOFT) * rates[_phase(a.get("street"))]
    return frac


def _river_strength(hole_idx, board_idx):
    """Share of all other hands that this hand beats on a full board (ties count half)."""
    known = set(hole_idx) | set(board_idx)
    b = [FULL_DECK[i] for i in board_idx]
    mine = evaluate([FULL_DECK[i] for i in hole_idx] + b)
    beat = n = 0.0
    for i, j in ALL_COMBOS:
        if i in known or j in known:
            continue
        o = evaluate([FULL_DECK[i], FULL_DECK[j]] + b)
        beat += 1.0 if mine > o else (0.5 if mine == o else 0.0)
        n += 1
    return beat / n if n else 0.5


def calibration_alpha(gs, rates):
    """Learn how much the opponent's bets really narrow their range, from showdowns.

    For each finished hand where they bet postflop and showed down on a full board:
    our model said their range was the top f_pred; the hand they showed sits at
    river strength s, which a uniform top-f range would produce at s ~ 1 - f/2, so
    f_obs = 2 (1 - s). The ratio log(f_obs) / log(f_pred) is the exponent that
    would have made the model right. alpha is its average, shrunk toward 1."""
    me = gs.get("your_bot_id")
    current = gs.get("hand_num")
    for h, entries in _HISTORY.items():
        if h == current or h in _CALIB:
            continue
        _CALIB[h] = None
        opp = next((e for e in entries if e.get("bot_id") != me and e.get("action") == "show"), None)
        bet_post = any(e.get("bot_id") != me and e.get("action") in BETS and e.get("street") != "preflop"
                       for e in entries)
        board = _BOARDS.get(h) or []
        if not (opp and bet_post and len(board) == 5 and opp.get("cards")):
            continue
        f_pred = raw_range(entries, me, rates)
        if f_pred > 0.95:
            continue
        s = _river_strength([IDX[c] for c in opp["cards"]], [IDX[c] for c in board])
        f_obs = min(max(2.0 * (1.0 - s), 0.02), 1.0)
        _CALIB[h] = min(max(math.log(f_obs) / math.log(f_pred), 0.0), 3.0)
    obs = [r for r in _CALIB.values() if r is not None]
    if not CALIB_K:
        return 1.0, len(obs)
    alpha = estimate(sum(obs), len(obs), 1.0, CALIB_K)
    return min(max(alpha, ALPHA_BOUNDS[0]), ALPHA_BOUNDS[1]), len(obs)


def opp_range(gs, rates, alpha):
    """Share of hands the opponent can still hold (1.0 = any two cards)."""
    frac = raw_range(gs.get("action_log") or [], gs.get("your_bot_id"), rates)
    return max(MIN_RANGE, frac ** alpha) if frac < 1.0 else 1.0


# --- Equity -------------------------------------------------------------------
def _equity_scores(combos, board_cards, known):
    """Average percentile rank of each combo over RANK_RUNOUTS shared runouts to the
    river: a strength that credits draws for the times they get there."""
    rest = [i for i in range(52) if i not in known]
    need = 5 - len(board_cards)
    score = dict.fromkeys(combos, 0.0)
    cnt = dict.fromkeys(combos, 0)
    for _ in range(RANK_RUNOUTS):
        ro = random.sample(rest, need)
        ros = set(ro)
        b = board_cards + [FULL_DECK[i] for i in ro]
        vals = sorted((evaluate([FULL_DECK[c[0]], FULL_DECK[c[1]]] + b), c)
                      for c in combos if c[0] not in ros and c[1] not in ros)
        n = len(vals)
        k = 0
        while k < n:                                 # equal hands share their average rank
            j = k
            while j + 1 < n and vals[j + 1][0] == vals[k][0]:
                j += 1
            r = (k + j) / 2 / max(n - 1, 1)
            for t in range(k, j + 1):
                c = vals[t][1]
                score[c] += r
                cnt[c] += 1
            k = j + 1
    return {c: (score[c] / cnt[c] if cnt[c] else 0.0) for c in combos}


def range_combos(known, board_cards, frac):
    """Opponent combos in the top `frac` of what they could hold: by preflop
    percentile before the flop; after it by equity (flop/turn) or made hand (river)."""
    combos = [c for c in ALL_COMBOS if c[0] not in known and c[1] not in known]
    if frac >= 1.0:
        return combos
    if not board_cards:
        key = COMBO_PCT.__getitem__
    elif len(board_cards) < 5 and RANK_RUNOUTS:
        key = _equity_scores(combos, board_cards, known).__getitem__
    else:
        key = lambda c: evaluate([FULL_DECK[c[0]], FULL_DECK[c[1]]] + board_cards)
    combos.sort(key=key, reverse=True)
    return combos[:max(1, round(frac * len(combos)))]


def equity(hole, board, n_opp, frac, deadline, max_iters=MC_MAX_ITERS):
    """Monte Carlo share of the pot vs n_opp opponents holding hands from their range."""
    known = {IDX[c] for c in hole} | {IDX[c] for c in board}
    h = [FULL_DECK[IDX[c]] for c in hole]
    b0 = [FULL_DECK[IDX[c]] for c in board]
    allowed = range_combos(known, b0, frac)
    rest = [i for i in range(52) if i not in known]
    need = 5 - len(b0)
    choice, sample = random.choice, random.sample

    won = 0.0
    n = 0
    while n < max_iters:
        if (n & 63) == 0 and perf_counter() >= deadline:
            break
        used, opps = set(), []
        for _ in range(n_opp):                       # their hands, no shared cards
            c = choice(allowed)
            while c[0] in used or c[1] in used:
                c = choice(allowed)
            used.update(c)
            opps.append(c)
        runout = sample(rest, need)                  # the board, avoiding their cards
        while used.intersection(runout):
            runout = sample(rest, need)
        b = b0 + [FULL_DECK[i] for i in runout]
        mine = evaluate(h + b)
        best, ties = -1, 0
        for i, j in opps:
            o = evaluate([FULL_DECK[i], FULL_DECK[j]] + b)
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


def n_opponents(gs):
    me = gs.get("seat_to_act")
    live = [p for p in gs.get("players", [])
            if p.get("seat") != me and p.get("state") in ("active", "all_in")]
    return max(1, len(live))


def strength(gs, frac, deadline):
    """Equity vs the opponent's range. Heads-up preflop vs any two cards is a table lookup."""
    hole, board = gs["your_cards"], gs.get("community_cards") or []
    n_opp = n_opponents(gs)
    if not board and PREFLOP and n_opp == 1 and frac >= 1.0:
        return PREFLOP[preflop_key(hole)]["equity"]
    mc_deadline = min(deadline, perf_counter() + MC_TIME_S)
    return equity(hole, board, n_opp, frac, mc_deadline)


# --- Decision helpers ---------------------------------------------------------
def _safe_action(gs):
    return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


def _bet_target(gs, pot_frac):
    """Total bet this street for a bet/raise of pot_frac of the pot on top of the current bet."""
    target = gs.get("current_bet", 0) + int(pot_frac * (gs["pot"] + gs.get("amount_owed", 0)))
    return max(target, gs.get("min_raise_to", 0))


def _raise_to(gs, total):
    """Raise to `total` chips this street; shove if that's most of our stack."""
    max_total = gs["your_stack"] + gs.get("your_bet_this_street", 0)
    total = max(int(total), gs.get("min_raise_to", 0))
    if total >= 0.8 * max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": total}


def _bet(gs, pot_frac):
    """Bet/raise pot_frac of the pot; shove if that's most of our stack."""
    return _raise_to(gs, _bet_target(gs, pot_frac))


def call_threshold(gs, owed, pot_odds):
    """Pot odds, plus DEEP_EDGE when the call commits most of our stack while deep."""
    if not owed:
        return 0.0
    opp_all_in = any(p.get("state") == "all_in" for p in gs.get("players", [])
                     if p.get("seat") != gs.get("seat_to_act"))
    commits = opp_all_in or owed >= 0.5 * gs["your_stack"]
    bb = (gs.get("blinds") or {}).get("bb") or 1
    each_bb = (gs["pot"] + owed) / 2 / bb             # heads-up: each player's share after calling
    return pot_odds + (DEEP_EDGE if commits and each_bb > DEEP_BB else 0.0)


def _opp_raised(gs):
    me = gs.get("your_bot_id")
    return any(a.get("bot_id") != me and a.get("action") in BETS for a in gs.get("action_log") or [])


def _eff_bb(gs):
    """Effective stack in big blinds: the smaller of our and their total on this street."""
    me = gs.get("seat_to_act")
    opp = next((p for p in gs.get("players", []) if p.get("seat") != me
                and p.get("state") in ("active", "all_in")), None)
    bb = (gs.get("blinds") or {}).get("bb") or 1
    mine = gs["your_stack"] + gs.get("your_bet_this_street", 0)
    theirs = (opp["stack"] + opp.get("bet_this_street", 0)) if opp else mine
    return min(mine, theirs) / bb


def _interp(table, x):
    """Piecewise-linear lookup in [(x, y), ...] sorted by x, clamped at the ends."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def _no_pair(hole, board):
    """True if neither hole card pairs the board and we don't hold a pocket pair."""
    ranks = {c[0] for c in board}
    return hole[0][0] != hole[1][0] and hole[0][0] not in ranks and hole[1][0] not in ranks


def preflop_rules(gs, info):
    """Heads-up preflop spots handled before the equity logic (unraised pots only).
    Returns an action, or None to fall through."""
    if gs["street"] != "preflop" or n_opponents(gs) != 1 or not PREFLOP or _opp_raised(gs):
        return None
    pct = PREFLOP[preflop_key(gs["your_cards"])]["pct"]
    is_sb = gs.get("seat_to_act") == gs.get("sb_seat")
    eff = _eff_bb(gs)
    info["eff_bb"] = round(eff, 1)

    if PUSH_FOLD_BB and eff <= PUSH_FOLD_BB:                    # 4. short: shove or fold
        push = _interp(HU_PUSH, eff)
        if is_sb and not gs.get("can_check"):
            info["push_fold"] = True
            return {"action": "all_in"} if pct >= 1.0 - push else {"action": "fold"}
        if gs.get("can_check"):                                 # big blind vs a limp
            info["push_fold"] = True
            return {"action": "all_in"} if pct >= 1.0 - push * HU_SHORT_ISO else {"action": "check"}
        return None

    if is_sb and not gs.get("can_check"):                       # 3. SB unraised: raise or fold
        if pct < SB_FOLD_PCT:
            info["sb_fold"] = True
            return {"action": "fold"}
        if SB_OPEN_BB:
            info["sb_open"] = True
            bb = (gs.get("blinds") or {}).get("bb") or 1
            return _raise_to(gs, SB_OPEN_BB * bb)
    return None


def bluff_raise(gs, eq, required, counts, info):
    """6. Facing a postflop bet without a value hand: raise if that beats calling
    or folding in EV, counting how often they fold to raises. Returns an action or None."""
    if not BLUFF_RAISE or gs["street"] == "preflop" or n_opponents(gs) != 1:
        return None
    owed = gs.get("amount_owed", 0)
    if not owed or gs.get("can_check"):
        return None
    me, seat, street = gs.get("your_bot_id"), gs.get("seat_to_act"), gs["street"]
    if any(p.get("seat") != seat and p.get("state") == "all_in" for p in gs.get("players", [])):
        return None                                              # they can't fold any more
    if any(a.get("bot_id") == me and a.get("street") == street and a.get("action") in BETS
           for a in gs.get("action_log") or []):
        return None                                              # one raise per street
    size = random.uniform(*BET_SIZE_RANGE)
    target = _bet_target(gs, size)
    raise_chips = target - gs.get("your_bet_this_street", 0)   # R: chips this raise puts in
    if raise_chips >= BLUFF_RAISE_MAX_STACK * gs["your_stack"]:
        return None
    pot = gs["pot"]
    breakeven = raise_chips / (raise_chips + pot)
    f = estimate(*counts["fold_raise"], BLUFF_RAISE_PRIOR * breakeven, BLUFF_RAISE_K)
    eq_c = eq * CALLED_EQ_MULT
    ev_raise = f * pot + (1 - f) * (eq_c * (pot + 2 * raise_chips - owed) - raise_chips)
    ev_call = eq * (pot + owed) - owed if eq >= required else 0.0
    base = max(ev_call, 0.0)
    info.update(fold_raise=round(f, 2), fold_raise_n=counts["fold_raise"][1],
                ev_raise=round(ev_raise), ev_call=round(ev_call))
    if ev_raise > base + BLUFF_RAISE_MARGIN * pot:
        info["bluff_raise"] = True
        return _raise_to(gs, target)
    return None


# --- Decision -----------------------------------------------------------------
def think(gs, deadline):
    street = gs["street"]
    hole, board = gs["your_cards"], gs.get("community_cards") or []
    hu = n_opponents(gs) == 1
    update_history(gs)
    counts = opponent_counts(gs)
    rates = {ph: estimate(*counts[ph], OPEN_DEFAULT, OPEN_K) for ph in ("pre", "post")}
    alpha, n_calib = calibration_alpha(gs, rates)
    frac = opp_range(gs, rates, alpha)
    info = {"range": round(frac, 2), "alpha": round(alpha, 2), "calib_n": n_calib,
            "open_rate": {ph: round(r, 2) for ph, r in rates.items()}, "hist_hands": len(_HISTORY)}

    act = preflop_rules(gs, info)
    if act is not None:
        act["debug"] = info
        return act

    eq = strength(gs, frac, deadline)
    owed = gs.get("amount_owed", 0)
    pot_odds = owed / (gs["pot"] + owed) if owed else 0.0
    required = call_threshold(gs, owed, pot_odds)
    size = PREFLOP_RAISE_FRAC if street == "preflop" else random.uniform(*BET_SIZE_RANGE)
    info.update(equity=round(eq, 3), required=round(required, 3),
                pot_odds=round(pot_odds, 3), size=round(size, 2))

    # Our own unbet postflop spots: value hand vs bluff candidate (for the bluff ratio).
    unbet_post = street != "preflop" and hu and gs.get("can_check")
    candidate = eq < BLUFF_MAX_EQ and (not BLUFF_NO_PAIR or _no_pair(hole, board))
    if unbet_post:
        if eq >= RAISE_STRENGTH[street]:
            _SELF["value"] += 1
        elif candidate:
            _SELF["weak"] += 1

    if eq >= RAISE_STRENGTH[street] and eq >= required:            # value bet/raise
        act = _bet(gs, size)
    else:
        # Bluff: s / (1 + s) bluffs per value bet (balanced), scaled 0x..2x by how
        # much they fold vs break-even, converted with our value share q.
        if unbet_post and candidate:
            breakeven = size / (1 + size)
            fold_rate = estimate(*counts["fold"], breakeven, FOLD_K)
            fold_adj = min(max(fold_rate / breakeven, 0.0), 2.0)
            q = estimate(_SELF["value"], _SELF["value"] + _SELF["weak"],
                         VALUE_SHARE_DEFAULT, VALUE_SHARE_K)
            p_bluff = min(1.0, BLUFF_SCALE * breakeven * fold_adj * q / max(1.0 - q, 1e-6))
            info.update(fold_rate=round(fold_rate, 2), fold_adj=round(fold_adj, 2),
                        value_share=round(q, 2), p_bluff=round(p_bluff, 2))
            if (_bet_target(gs, size) < BLUFF_MAX_STACK * gs["your_stack"]
                    and random.random() < p_bluff):
                act = _bet(gs, size)
                info["bluff"] = True
        if act is None:                                           # 6. raise as a bluff/semi-bluff
            act = bluff_raise(gs, eq, required, counts, info)
        if act is None:                                           # check, call on price, or fold
            if gs.get("can_check"):
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

"""CFR: a heads-up bot that plays a precomputed counterfactual-regret-minimization strategy.

Based on "Deep Counterfactual Regret Minimization" (Brown, Lerer, Gross & Sandholm, ICML 2019).
The paper compares Deep CFR against tabular abstractions solved with external-sampling
Linear Monte Carlo CFR, which it calls "the leading algorithm in this setting". That
tabular approach is what this bot uses; it fits a hackathon's time and the 2 s / half-core
sandbox. The strategy is trained offline by training/train_cfr.py and loaded from
data/cfr.json.

Abstraction (shared with the trainer, which imports this file so the two never drift):
  cards    preflop: the 169 starting-hand classes (AKs, 72o, ...).
           flop/turn: equity vs a random hand, in equal-frequency buckets, split by
           whether the hand is a draw (equity well above its current strength).
           river: exact strength vs a random hand, in equal-frequency buckets.
  actions  fold, check/call, raise half pot, raise pot, all-in. Raises close to the
           whole stack become all-in, and each street caps the number of non-all-in raises.
  infoset  street, position, stack depth (preflop) or stack-to-pot ratio (postflop),
           who had the initiative on the previous street, card bucket, and this street's
           abstract betting so far. Earlier streets are summarised by the pot size (an
           imperfect-recall abstraction, standard for poker).

At play time the real betting is translated into the abstraction (each raise maps to the
nearest abstract size), the bucket is computed, and an action is sampled from the average
strategy, the part of CFR that converges to an equilibrium. Low-probability actions are
dropped first ("purification"), which removes noise left by sampling. A situation the
strategy never reached falls back to a simple equity rule.
"""
import json
import os
import random  # noqa: F401  (seeded per match, so a seeded match replays exactly)
from time import perf_counter

import eval7

# --- Abstraction (the trainer imports these, keep them here) ------------------------
STREETS = ("preflop", "flop", "turn", "river")
RANKS = "23456789TJQKA"
RAISE_FRACS = (("h", 0.5), ("p", 1.0))      # raise sizes, as a fraction of the pot after calling
MAX_RAISES = (3, 2, 2, 2)                   # non-all-in raises allowed per street
ALLIN_COMMIT = 0.8                          # a raise to >= 80% of the effective stack is an all-in
PRE_DEPTH = (6, 10, 15, 22, 32, 50, 80, 130)    # preflop classes: effective stack, in big blinds
POST_SPR = (0.6, 1.2, 2.5, 5.0, 10.0)           # postflop classes: effective stack / pot at street start
SIZE_SPLIT = 0.707                          # translate a raise of f x pot to 'h' below this, else 'p'

ALL_HANDS = eval7.HandRange("22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32")

PURIFY = 0.06            # drop actions the strategy plays less often than this
EQUITY_ITERS = 20000     # Monte Carlo samples for flop/turn equity at play time (~0.6 ms)


def classify(x, bounds):
    """Index of the interval of `bounds` that x falls in (0..len(bounds))."""
    n = 0
    for b in bounds:
        if x > b:
            n += 1
    return n


def hand_class(hole):
    """["As", "Kh"] -> "AKo", ["Td", "Tc"] -> "TT"."""
    (r1, s1), (r2, s2) = hole[0], hole[1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def raw_strength(hole, board, iters):
    """(equity, current strength) vs a random hand. hole/board are eval7 Cards.
    On the river both are the exact share of random hands we beat."""
    if len(board) == 5:
        eq = eval7.py_hand_vs_range_exact(hole, ALL_HANDS, board)
        return eq, eq
    eq = eval7.py_hand_vs_range_monte_carlo(hole, ALL_HANDS, board, iters)
    hs = eval7.py_hand_vs_range_exact(hole, ALL_HANDS, board)
    return eq, hs


def bucket_from_raw(s, eq, hs, meta):
    """Card bucket for street s (1..3) from raw strengths, using the trained thresholds."""
    q = classify(eq, meta["eq_bounds"][str(s)])
    if s < 3 and eq - hs > meta["draw_split"][str(s)]:
        return "%dd" % q
    return str(q)


def infoset_key(s, pos, depth, init, bucket, seq):
    """pos: 0 button/small blind, 1 big blind. init: 'n' none, 'm' me, 'o' opponent."""
    return "%d%d%d%s|%s|%s" % (s, pos, depth, init, bucket, seq)


def abstract_actions(s, owed, pot, current_bet, min_raise_to, my_total, opp_total, nraises):
    """Abstract actions available: [(code, raise-to total or None)]. Totals are chips on
    this street; my_total / opp_total are the most each player can have in this street."""
    acts = [("f", None)] if owed > 0 else []
    acts.append(("c", None))
    cap = min(my_total, opp_total)
    if cap > current_bet:                       # both players still have chips to raise with
        if nraises < MAX_RAISES[s]:
            last = None
            for code, frac in RAISE_FRACS:
                target = max(current_bet + frac * (pot + owed), min_raise_to)
                if target >= ALLIN_COMMIT * cap:
                    break
                if last is not None and target <= last * 1.05:
                    continue
                acts.append((code, target))
                last = target
        acts.append(("a", cap))
    return acts


def translate_raise(new_total, current_bet, pot, owed, cap, is_all_in):
    """Map a real raise to 'h', 'p' or 'a'."""
    if is_all_in or new_total >= ALLIN_COMMIT * cap:
        return "a"
    frac = (new_total - current_bet) / max(pot + owed, 1e-9)
    return "h" if frac < SIZE_SPLIT else "p"


# --- The trained strategy -------------------------------------------------------------
def _load():
    for d in (os.environ.get("BOT_DATA_DIR"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")):
        if d:
            p = os.path.join(d, "cfr.json")
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
    return None


DATA = _load()
META = DATA["meta"] if DATA else None
STRATEGY = DATA["strategy"] if DATA else {}


# --- Reading the real hand ------------------------------------------------------------
def replay(gs):
    """Walk this hand's action_log the way the engine does and translate it into the
    abstraction. Returns per-street abstract sequences, raise counts, aggressors, the pot
    and stacks at the start of the current street, and each player's starting stack."""
    log = gs.get("action_log") or []
    bets, total_in = {}, {}
    street_now = None
    added = []                                 # first pass: chips each entry put in
    for e in log:
        b, a, amt = e["bot_id"], e["action"], int(e.get("amount") or 0)
        if e["street"] != street_now:
            street_now, bets = e["street"], {}
        if a == "ante":
            add = amt
        elif a in ("small_blind", "big_blind", "call"):
            add = amt
            bets[b] = bets.get(b, 0) + amt
        elif a in ("raise", "all_in"):
            add = max(amt - bets.get(b, 0), 0)
            bets[b] = max(amt, bets.get(b, 0))
        else:
            add = 0
        added.append(add)
        total_in[b] = total_in.get(b, 0) + add

    stacks_now = {p["bot_id"]: p["stack"] for p in gs["players"]}
    start = {b: stacks_now.get(b, 0) + total_in.get(b, 0) for b in stacks_now}

    st = {"seq": ["", "", "", ""], "nraises": [0, 0, 0, 0], "aggr": [None] * 4,
          "street_pot": [0, 0, 0, 0], "street_eff": [0, 0, 0, 0]}
    left = dict(start)
    pot = 0
    s_idx = 0
    bets = {}
    cur, last_raise = 0, gs["blinds"]["bb"]
    started = [False] * 4

    def open_street(k):
        st["street_pot"][k] = pot
        st["street_eff"][k] = min(left.values()) if left else 0
        started[k] = True

    for e, add in zip(log, added):
        k = STREETS.index(e["street"]) if e["street"] in STREETS else s_idx
        if k != s_idx or not started[k]:
            if k != s_idx:
                bets, cur, last_raise = {}, 0, gs["blinds"]["bb"]
            s_idx = k
            if not started[k]:
                open_street(k)
        b, a = e["bot_id"], e["action"]
        if a == "ante":
            pot += add
            left[b] -= add
            st["street_pot"][0] = pot           # antes are in the pot before anyone acts
            continue
        if a in ("small_blind", "big_blind"):
            pot += add
            left[b] -= add
            bets[b] = bets.get(b, 0) + add
            cur = max(cur, bets[b])
            continue
        others = [o for o in left if o != b]
        mine = bets.get(b, 0)
        my_total = left[b] + mine
        opp_total = max((left[o] + bets.get(o, 0)) for o in others) if others else 0
        owed = max(cur - mine, 0)
        if a in ("check", "call"):
            code = "c"
        elif a == "fold":
            code = "f"
        else:
            new_total = int(e.get("amount") or 0)
            if new_total <= cur:
                code = "c"
            else:
                code = translate_raise(new_total, cur, pot, owed, min(my_total, opp_total), a == "all_in")
                if code in "hp":
                    st["nraises"][k] += 1
                if new_total - cur >= last_raise:
                    last_raise = new_total - cur
                cur = new_total
                st["aggr"][k] = b
        st["seq"][k] += code
        pot += add
        left[b] -= add
        bets[b] = mine + add
    s_cur = STREETS.index(gs["street"])
    if not started[s_cur]:
        st["street_pot"][s_cur] = gs["pot"]
        st["street_eff"][s_cur] = min(p["stack"] + p.get("bet_this_street", 0) for p in gs["players"]
                                      if p["state"] in ("active", "all_in"))
    st["start"] = start
    return st


def current_bucket(gs, s):
    hole = gs["your_cards"]
    if s == 0:
        return hand_class(hole)
    h = [eval7.Card(c) for c in hole]
    b = [eval7.Card(c) for c in gs["community_cards"]]
    eq, hs = raw_strength(h, b, EQUITY_ITERS)
    return bucket_from_raw(s, eq, hs, META), eq


def lookup(s, pos, depth, init, bucket, seq):
    """The strategy for this infoset, trying nearby depth classes and initiative if the
    exact one was never reached in training."""
    n = len(PRE_DEPTH) if s == 0 else len(POST_SPR)
    for dd in (0, -1, 1, -2, 2, -3, 3):
        d = depth + dd
        if not 0 <= d <= n:
            continue
        for ini in ((init,) if s == 0 else (init, "o", "m", "n")):
            row = STRATEGY.get(infoset_key(s, pos, d, ini, bucket, seq))
            if row:
                return row, d, ini
    return None, None, None


# --- Decision -------------------------------------------------------------------------
def _safe(gs):
    return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


def _to_engine(gs, code, target):
    if code == "f":
        return {"action": "check"} if gs["can_check"] else {"action": "fold"}
    if code == "c":
        return {"action": "check"} if gs["can_check"] else {"action": "call"}
    if code == "a":
        return {"action": "all_in"}
    amount = int(round(max(target, gs["min_raise_to"])))
    if amount >= gs["your_stack"] + gs["your_bet_this_street"]:
        return {"action": "all_in"}
    return {"action": "raise", "amount": amount}


def fallback(gs, s, eq):
    """Used when the strategy has no entry: plain pot-odds play on equity vs random."""
    if eq is None:
        h = [eval7.Card(c) for c in gs["your_cards"]]
        b = [eval7.Card(c) for c in gs["community_cards"]]
        eq = eval7.py_hand_vs_range_monte_carlo(h, ALL_HANDS, b, 5000)
    owed, pot = gs["amount_owed"], gs["pot"]
    odds = owed / (pot + owed) if owed else 0.0
    if eq > 0.8 and gs["your_stack"] > owed:
        target = gs["current_bet"] + 0.75 * (pot + owed)
        return _to_engine(gs, "p", target)
    if owed == 0:
        return {"action": "check"}
    return {"action": "call"} if eq >= odds + 0.08 else {"action": "fold"}


# --- Exploitation layer -------------------------------------------------------------
# CFR's strategy assumes an opponent who bluffs and folds in balance. Many bots don't,
# so during the match we keep two statistics and override the strategy only when the
# evidence clearly says a call or a bluff loses (or wins) against this opponent:
#   - how strong the opponent's hand was at showdown after it bet or raised after the
#     flop (a bot that never bluffs shows only strong hands: stop paying it off);
#   - how often it folds when we bet after the flop (don't bluff a bot that won't fold).
# The priors are what a balanced opponent would look like, so with little data the
# layer leaves the CFR strategy alone.
PRIOR_BET_STRENGTH = (0.2, 0.45, 0.65, 0.8, 0.92)   # pseudo-observations of a balanced betting range
MIN_SHOWDOWNS = 3
PRIOR_FOLD, PRIOR_FOLD_N = 0.4, 6
BLUFF_EQ = 0.5          # a bet with less equity than this is (mostly) a bluff
# Deep-stack preflop shoves (risking 40+ big blinds to win a few) turn matches into
# coin flips; the coarse abstraction overrates them. Below a 4-bet, raise instead.
DEEP_SHOVE_DEPTH = classify(40, PRE_DEPTH)
# A deep preflop all-in from a bot that rarely shoves is strong: call only with enough
# equity against this range. Bots that shove often keep the CFR strategy.
TIGHT_SHOVE_RANGE = eval7.HandRange("99+,AQs+,AKo")
PRIOR_SHOVE, PRIOR_SHOVE_N = 0.1, 10
WIDE_SHOVER = 0.3

OPP = {"done": set(), "bet_strength": [], "folds": 0, "chances": 0, "pre_acts": 0, "pre_shoves": 0}
BOARDS = {}             # hand_num -> the last board we saw in that hand


def update_model(gs, opp_id):
    hn = gs.get("hand_num")
    BOARDS[hn] = list(gs.get("community_cards") or [])
    hands = {}
    for e in gs.get("match_action_log") or []:
        hands.setdefault(e.get("hand_num"), []).append(e)
    if not hands:
        return
    first = min(hands)                          # may be cut off by the 200-entry window
    for h, entries in hands.items():
        if h == hn or h == first or h in OPP["done"]:
            continue
        OPP["done"].add(h)
        me = gs["your_bot_id"]
        opp_bet = any(e["bot_id"] == opp_id and e["action"] in ("raise", "all_in")
                      and e["street"] in ("flop", "turn", "river") for e in entries)
        shown = next((e.get("cards") for e in entries if e["bot_id"] == opp_id and e["action"] == "show"), None)
        board = BOARDS.get(h) or []
        if opp_bet and shown and len(board) == 5:
            hole = [eval7.Card(c) for c in shown]
            OPP["bet_strength"].append(eval7.py_hand_vs_range_exact(hole, ALL_HANDS, [eval7.Card(c) for c in board]))
        for e in entries:                       # how often it shoves preflop
            if e["bot_id"] == opp_id and e["street"] == "preflop" and e["action"] in ("fold", "check", "call", "raise", "all_in"):
                OPP["pre_acts"] += 1
                OPP["pre_shoves"] += e["action"] == "all_in"
        for i, e in enumerate(entries):         # opponent's answer to each of our postflop bets
            if e["bot_id"] == me and e["action"] in ("raise", "all_in") and e["street"] in ("flop", "turn", "river"):
                reply = next((x for x in entries[i + 1:] if x["bot_id"] == opp_id and x["action"] != "show"), None)
                if reply is not None:
                    OPP["chances"] += 1
                    OPP["folds"] += reply["action"] == "fold"


def p_beat_bettor(strength):
    """Chance our hand beats what this opponent bets with (observed + prior hands)."""
    obs = OPP["bet_strength"]
    below = sum(x < strength for x in obs) + sum(x < strength for x in PRIOR_BET_STRENGTH)
    return below / (len(obs) + len(PRIOR_BET_STRENGTH))


def exploit(gs, s, eq, code, info):
    """Possibly replace the CFR action `code`. Returns (code, target or None if unchanged)."""
    if s == 0 or eq is None:
        return code, None
    owed, pot = gs["amount_owed"], gs["pot"]
    if owed > 0 and code in "fc" and len(OPP["bet_strength"]) >= MIN_SHOWDOWNS:
        odds = owed / (pot + owed)
        pb = p_beat_bettor(eq)
        info["p_beat"] = round(pb, 2)
        if code == "c" and pb < odds - 0.05:
            info["exploit"] = "fold_vs_value"
            return "f", None
        if code == "f" and pb > odds + 0.15:
            info["exploit"] = "call_vs_bluffer"
            return "c", None
    if owed == 0 and code in "hp" and eq < BLUFF_EQ:
        fold_rate = (OPP["folds"] + PRIOR_FOLD * PRIOR_FOLD_N) / (OPP["chances"] + PRIOR_FOLD_N)
        frac = dict(RAISE_FRACS)[code]
        needed = frac / (1 + frac)                 # a bluff of frac x pot needs this fold rate
        info["fold_rate"] = round(fold_rate, 2)
        if fold_rate < needed - 0.05:
            info["exploit"] = "no_bluff"
            return "c", None
    return code, None


def think(gs):
    s = STREETS.index(gs["street"])
    me = gs["your_bot_id"]
    opp = next((p for p in gs["players"] if p["bot_id"] != me and p["state"] in ("active", "all_in")), None)
    if opp is None:
        return _safe(gs)
    update_model(gs, opp["bot_id"])
    bb = gs["blinds"]["bb"] or 1
    rp = replay(gs)
    pos = 0 if gs["seat_to_act"] == gs["button_seat"] else 1

    if s == 0:
        eff = min(rp["start"].values()) / bb
        depth = classify(eff, PRE_DEPTH)
        init = "n"
        bucket, eq = hand_class(gs["your_cards"]), None
    else:
        spr = rp["street_eff"][s] / max(rp["street_pot"][s], 1)
        depth = classify(spr, POST_SPR)
        a = rp["aggr"][s - 1]
        init = "n" if a is None else ("m" if a == me else "o")
        bucket, eq = current_bucket(gs, s)

    my_total = gs["your_stack"] + gs["your_bet_this_street"]
    opp_total = opp["stack"] + opp.get("bet_this_street", 0)
    legal = abstract_actions(s, gs["amount_owed"], gs["pot"], gs["current_bet"], gs["min_raise_to"],
                             my_total, opp_total, rp["nraises"][s])
    seq = rp["seq"][s]
    info = {"key": infoset_key(s, pos, depth, init, bucket, seq), "eq": None if eq is None else round(eq, 3)}

    row, d_used, i_used = lookup(s, pos, depth, init, bucket, seq) if STRATEGY else (None, None, None)
    if row is None:
        info["fallback"] = True
        act = fallback(gs, s, eq)
        act["debug"] = info
        return act

    probs = dict(zip(row[0], row[1]))
    choices = [(c, t, probs.get(c, 0.0)) for c, t in legal]
    total = sum(p for _, _, p in choices)
    if total <= 0:
        info["fallback"] = "no_overlap"
        act = fallback(gs, s, eq)
        act["debug"] = info
        return act
    choices = [(c, t, p / total) for c, t, p in choices]
    kept = [x for x in choices if x[2] >= PURIFY] or [max(choices, key=lambda x: x[2])]
    weights = [p for _, _, p in kept]
    code, target, _ = random.choices(kept, weights)[0]
    if d_used != depth or i_used != init:
        info["near"] = "%s%s" % (d_used, i_used)
    info["p"] = {c: round(p, 2) for c, _, p in choices}
    info["pick"] = code
    if s == 0 and code == "a" and depth >= DEEP_SHOVE_DEPTH and rp["nraises"][0] < 3:
        alt = [x for x in choices if x[0] in "hp"]          # deep: raise instead of shoving
        if alt:
            code, target, _ = max(alt, key=lambda x: x[2])
            info["deep_shove"] = code
    code, _ = exploit(gs, s, eq, code, info)
    if s == 0 and depth >= DEEP_SHOVE_DEPTH and gs["amount_owed"] >= 0.4 * (gs["your_stack"] + gs["your_bet_this_street"]):
        shove_rate = (OPP["pre_shoves"] + PRIOR_SHOVE * PRIOR_SHOVE_N) / (OPP["pre_acts"] + PRIOR_SHOVE_N)
        info["shove_rate"] = round(shove_rate, 2)
        if shove_rate < WIDE_SHOVER:            # facing a big preflop bet from a tight bot
            h = [eval7.Card(c) for c in gs["your_cards"]]
            eq_t = eval7.py_hand_vs_range_monte_carlo(h, TIGHT_SHOVE_RANGE, [], 4000)
            odds = gs["amount_owed"] / (gs["pot"] + gs["amount_owed"])
            code = "c" if eq_t >= odds else "f"
            info["vs_tight"] = round(eq_t, 2)
    act = _to_engine(gs, code, target)
    act["debug"] = info
    return act


def decide(game_state: dict) -> dict:
    start = perf_counter()
    if game_state.get("type") == "warmup":
        raw_strength([eval7.Card("As"), eval7.Card("Kd")],
                     [eval7.Card(c) for c in ("7h", "8h", "2c")], 1000)
        return {"action": "check"}
    try:
        action = think(game_state)
    except Exception as e:                      # never crash: a crash folds anyway
        print("think() failed: %r" % (e,))
        action = None
    if not isinstance(action, dict) or "action" not in action:
        action = _safe(game_state)
    action.setdefault("debug", {})["ms"] = round((perf_counter() - start) * 1000, 1)
    return action

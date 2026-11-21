"""switcher: rotates between neural_net, all_in, hsr_modified_v4 every 5 hands (GENERATED, do not edit).

Built by training/build_switcher.py from the sub-bots' own bot.py files. Use it as a
sparring partner that changes style mid-match, e.g. to check that an opponent-modelling
bot (CFR) notices each switch and doesn't stay stuck on an exploit for the old style.

Hand h plays SUBBOTS[(h // SEGMENT_HANDS) % len(SUBBOTS)]. Every decision's debug says
which sub-bot acted ("model") and which segment it was ("segment").
"""
import random  # noqa: F401  (seeded per match; the sub-bots share it)
from time import perf_counter

SEGMENT_HANDS = 5


# ============================================================================
# neural_net  (from my_bots/neural_net/bot.py)
# ============================================================================
def _make_neural_net():
    """Neural-net bot: a small MLP trained on the PHH hand histories (training/train_nn.py).

    Per decision: compute ~13 simple features (street, equity vs a random hand, pot
    odds, sizes), run the MLP in numpy, and pick one of 5 actions. Weights live in
    data/model.npz; without them the bot falls back to check/fold.
    """
    import math
    import os
    import random
    from time import perf_counter

    import eval7
    import numpy as np

    TIME_BUDGET_S = 1.95
    LIMIT_MARGIN_S = 0.05
    EQUITY_ITERS = 1000

    ACTIONS = ("fold", "call", "raise_half", "raise_pot", "all_in")
    STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    N_FEATURES = 13

    _ALL_HANDS = eval7.HandRange("22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32")


    # ── Features (shared with training/train_nn.py: keep the two in sync by importing) ──
    def equity(hole, board, iters=EQUITY_ITERS) -> float:
        """Chance our hand beats one random hand by the river."""
        h = [eval7.Card(c) for c in hole]
        b = [eval7.Card(c) for c in board]
        return eval7.py_hand_vs_range_monte_carlo(h, _ALL_HANDS, b, iters)


    def features(street, eq, pot, owed, eff_behind, bb, n_opp, in_pos, raises) -> list:
        x = [0.0] * N_FEATURES
        x[STREET_INDEX[street]] = 1.0
        x[4] = eq
        x[5] = owed / (pot + owed) if owed else 0.0          # pot odds
        x[6] = min(owed / pot, 5.0) if pot else 0.0          # bet faced, in pots
        x[7] = math.log1p(pot / bb) / 5                      # pot size in big blinds
        x[8] = min(eff_behind / pot, 20.0) / 20 if pot else 1.0   # stack-to-pot ratio
        x[9] = 1.0 if owed > 0 else 0.0                      # facing a bet
        x[10] = n_opp / 5                                    # opponents still in
        x[11] = 1.0 if in_pos else 0.0                       # acts last after the flop
        x[12] = min(raises, 4) / 4                           # raises so far this street
        return x


    # ── Model ─────────────────────────────────────────────────────────────────────
    _DATA_DIR = os.environ.get("BOT_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    _MODEL_PATH = os.path.join(_DATA_DIR, "model.npz")
    MODEL = None
    if os.path.exists(_MODEL_PATH):
        _m = np.load(_MODEL_PATH)
        MODEL = {
            "mean": _m["mean"], "std": _m["std"], "classes": [int(c) for c in _m["classes"]],
            "layers": [(_m[f"W{i}"], _m[f"b{i}"]) for i in range(int(_m["n_layers"]))],
        }


    def policy(x) -> dict:
        """Action name -> probability."""
        h = (np.asarray(x, dtype=float) - MODEL["mean"]) / MODEL["std"]
        last = len(MODEL["layers"]) - 1
        for i, (W, b) in enumerate(MODEL["layers"]):
            h = h @ W + b
            if i < last:
                h = np.maximum(h, 0.0)
        e = np.exp(h - h.max())
        p = e / e.sum()
        return {ACTIONS[c]: float(pi) for c, pi in zip(MODEL["classes"], p)}


    # ── Decision ──────────────────────────────────────────────────────────────────
    def _budget_s(gs):
        limit_ms = gs.get("time_limit_ms")
        if limit_ms and limit_ms / 1000 - LIMIT_MARGIN_S < TIME_BUDGET_S:
            return max(limit_ms / 1000 - LIMIT_MARGIN_S, 0.0)
        return TIME_BUDGET_S


    def _safe_action(gs):
        return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


    def think(gs, deadline):
        if MODEL is None:
            return None
        me = gs["seat_to_act"]
        opps = [p for p in gs["players"] if p["seat"] != me and p["state"] in ("active", "all_in")]
        pot, owed = gs["pot"], gs["amount_owed"]
        eff_behind = min(gs["your_stack"], max((p["stack"] for p in opps), default=0))
        raises = sum(1 for a in gs.get("action_log") or []
                     if a["street"] == gs["street"] and a["action"] in ("raise", "all_in"))

        eq = equity(gs["your_cards"], gs["community_cards"])
        x = features(gs["street"], eq, pot, owed, eff_behind, gs["blinds"]["bb"],
                     len(opps), me == gs["button_seat"], raises)
        probs = policy(x)

        if gs["can_check"]:                       # never fold when checking is free
            probs["call"] = probs.get("call", 0.0) + probs.pop("fold", 0.0)
        names, weights = zip(*probs.items())
        choice = random.choices(names, weights)[0]

        pot_after_call = pot + owed
        if choice == "fold":
            act = {"action": "fold"}
        elif choice == "call":
            act = {"action": "check"} if gs["can_check"] else {"action": "call"}
        elif choice == "raise_half":
            act = {"action": "raise", "amount": int(gs["current_bet"] + 0.5 * pot_after_call)}
        elif choice == "raise_pot":
            act = {"action": "raise", "amount": int(gs["current_bet"] + pot_after_call)}
        else:
            act = {"action": "all_in"}
        act["debug"] = {"eq": round(eq, 3), "p": {k: round(v, 3) for k, v in probs.items()}}
        return act


    def decide(game_state):
        start = perf_counter()
        deadline = start + _budget_s(game_state)
        if game_state.get("type") == "warmup":
            if MODEL is not None:                 # warm up eval7 and numpy before the clock matters
                equity(["As", "Kd"], [], 100)
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

    return decide


# ============================================================================
# all_in  (from my_bots/all_in/bot.py)
# ============================================================================
def _make_all_in():
    import random  # noqa: F401  (seeded per match, so a seeded match replays exactly)
    from time import perf_counter

    # ── Time management ───────────────────────────────────────────────────────────
    # The engine gives 2 s per decision and folds us if we go over. It cannot
    # interrupt decide(), so the strategy must watch the clock itself. `deadline`
    # is a perf_counter() timestamp; in any loop (Monte Carlo, search) check
    #     if perf_counter() >= deadline: break
    TIME_BUDGET_S = 1.95      # our own ceiling per decision
    LIMIT_MARGIN_S = 0.05     # if the engine's time_limit_ms is ever lower, stay this far under it


    def _budget_s(gs: dict) -> float:
        limit_ms = gs.get("time_limit_ms")
        if limit_ms and limit_ms / 1000 - LIMIT_MARGIN_S < TIME_BUDGET_S:
            return max(limit_ms / 1000 - LIMIT_MARGIN_S, 0.0)
        return TIME_BUDGET_S


    def _safe_action(gs: dict) -> dict:
        """What we play if thinking fails or runs out of time: never pay to continue."""
        return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


    # ── Strategy: one function per street ─────────────────────────────────────────
    def preflop(gs: dict, deadline: float) -> dict:
        return {"action": "all_in"}


    def flop(gs: dict, deadline: float) -> dict:
        return {"action": "all_in"}


    def turn(gs: dict, deadline: float) -> dict:
        return {"action": "all_in"}


    def river(gs: dict, deadline: float) -> dict:
        return {"action": "all_in"}


    STREETS = {"preflop": preflop, "flop": flop, "turn": turn, "river": river}


    def think(gs: dict, deadline: float) -> dict:
        """Pick the handler for the current street. Must return before `deadline`."""
        return STREETS[gs["street"]](gs, deadline)


    # ── Entry point ───────────────────────────────────────────────────────────────
    def decide(game_state: dict) -> dict:
        start = perf_counter()                       # start the clock first
        deadline = start + _budget_s(game_state)

        if game_state.get("type") == "warmup":       # one call before hand 1; load things here
            return {"action": "check"}

        try:
            action = think(game_state, deadline)
        except Exception as e:                       # a crash would fold anyway; fail safe and log it
            print(f"think() failed: {e!r}")
            action = None

        if not isinstance(action, dict) or "action" not in action:
            action = _safe_action(game_state)

        action["debug"] = {"think_ms": round((perf_counter() - start) * 1000, 1)}
        return action

    return decide


# ============================================================================
# hsr_modified_v4  (from my_bots/hsr_modified_v4/bot.py)
# ============================================================================
def _make_hsr_modified_v4():
    """hsr_modified_v4: hand-strength bot with range-aware equity, a preflop fold
    rule, and balanced bluffing scaled by how often the opponent folds.

    Core (unchanged from v3):
      - Equity is Monte Carlo against the hands the opponent can still hold. Each
        raise/shove they make this hand keeps the top NARROW_SOFT + (1 - NARROW_SOFT)
        * rate of their hands, where `rate` is how often they bet/raise when nobody has bet yet on
        the street (preflop and postflop tracked separately). A bot that always bets
        tells us nothing; a selective one is narrowed. Hands are ranked by preflop
        strength before the flop and by made-hand strength on the board after.
      - Call when equity >= pot odds, plus DEEP_EDGE when the call commits most of
        our stack while deep (don't flip the match on a 52% spot in a freezeout).
      - Bet/raise for value when equity clears RAISE_STRENGTH for the street.

    New in v4:
      1. Every opponent stat goes through estimate(): shrunk toward a safe default
         by sample size, so early in a match we assume nothing and late we trust data.
         Stats count over the whole match: the engine only sends the last 200 actions,
         so update_history() keeps every hand seen in module state, reset per match.
      2. Preflop fold rule: in the small blind, unraised, fold the bottom SB_FOLD_PCT.
      3. Postflop bets and raises (value and bluff alike) draw their size from
         BET_SIZE_RANGE, so size says nothing about strength.
      4. Bluffs: postflop, unbet street, hands with equity < BLUFF_MAX_EQ bet with
         probability = the opponent's estimated fold-to-bet rate, capped at twice the
         break-even rate s / (1 + s). The default fold rate is that break-even, which
         gives roughly the balanced bluff share; over-folders get bluffed more,
         calling stations less (toward 0).

    Off switches for A/B tests: SB_FOLD_PCT = 0, BET_SIZE_RANGE = (0.75, 0.75),
    BLUFF_SCALE = 0. NARROW_SOFT = 0 switches range narrowing to v2-style.
    """
    import json
    import os
    import random  # noqa: F401  (seeded per match, so a seeded match replays exactly)
    from time import perf_counter

    import eval7

    # --- Tunables -----------------------------------------------------------------
    # Opponent stats: estimate = (count + default * K) / (chances + K).
    OPEN_DEFAULT, OPEN_K = 0.4, 20    # how often they bet/raise when nobody has yet (per phase)
    FOLD_K = 15                       # fold to our postflop bets; default = break-even for the size
    # How hard each raise narrows their range: keep the top NARROW_SOFT + (1 - NARROW_SOFT) * rate.
    # 0.5 = gentle (v3-style, option A); 0.0 = narrow by the rate itself (v2-style, option C).
    NARROW_SOFT = 0.5
    # Their range never narrows below this share of hands.
    MIN_RANGE = 0.10

    # Commit guard: extra equity needed to call off most of our stack while deep.
    DEEP_EDGE = 0.10
    DEEP_BB = 20                      # "deep" = each player has > this many big blinds in the pot after calling

    # Strength needed to bet/raise for value.
    RAISE_STRENGTH = {"preflop": 0.64, "flop": 0.72, "turn": 0.76, "river": 0.80}
    PREFLOP_RAISE_FRAC = 1.0          # preflop raise: this fraction of the pot on top of the current bet
    BET_SIZE_RANGE = (0.50, 0.80)     # postflop bet/raise size, fraction of the pot, drawn uniformly

    # Preflop fold rule: small blind, unraised, fold hands below this percentile.
    SB_FOLD_PCT = 0.30

    # Bluffing (postflop, no bet yet on the street, heads-up).
    BLUFF_MAX_EQ = 0.35               # only hands this weak vs their range bluff; medium hands check
    BLUFF_SCALE = 1.0                 # multiplies the bluff probability; 0 turns bluffing off
    BLUFF_MAX_STACK = 0.5             # never bluff with a bet of this share of our stack or more

    MC_TIME_S = 0.6                   # Monte Carlo budget per decision (sandbox has 0.5 CPU)
    MC_MAX_ITERS = 4000

    TIME_BUDGET_S = 1.95
    LIMIT_MARGIN_S = 0.05


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


    # --- Opponent stats -----------------------------------------------------------
    VOLUNTARY = ("fold", "check", "call", "raise", "all_in")
    BETS = ("raise", "all_in")


    # Match-level history. The engine only sends the last MATCH_LOG_MAX_ENTRIES (200)
    # actions, about 25 hands, so we keep every hand we have seen in module state
    # (the bot process stays loaded for the whole match) and count over all of it.
    _HISTORY = {}                      # hand_num -> that hand's log entries, in order
    _HIST_KEY = {"match": None, "bot": None, "hand": -1}


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
            _HISTORY.clear()
            _HIST_KEY.update(match=match, bot=me)
        _HIST_KEY["hand"] = hand
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
        """Raw [count, chances] over the whole match (see update_history):
          pre / post: they bet/raise when nobody had bet yet on the street;
          fold:       they folded facing our postflop bet/raise."""
        me = gs.get("your_bot_id")
        c = {"pre": [0, 0], "post": [0, 0], "fold": [0, 0]}
        spot, opened, ours = None, False, False   # (hand, street); someone bet; the open bet is ours
        for e in match_log():
            a = e.get("action")
            if a not in VOLUNTARY:
                continue
            street = e.get("street")
            key = (e.get("hand_num"), street)
            if key != spot:
                spot, opened, ours = key, False, False
            if e.get("bot_id") != me:
                if not opened:
                    s = c[_phase(street)]
                    s[1] += 1
                    s[0] += a in BETS
                if ours and street != "preflop":
                    c["fold"][1] += 1
                    c["fold"][0] += a == "fold"
            if a in BETS:
                opened, ours = True, e.get("bot_id") == me
        return c


    def opp_range(gs, rates):
        """Share of hands the opponent can still hold (1.0 = any two cards): each
        raise they made this hand keeps the top NARROW_SOFT + (1 - NARROW_SOFT) * rate
        of what was left."""
        me = gs.get("your_bot_id")
        frac = 1.0
        for a in gs.get("action_log") or []:
            if a.get("bot_id") != me and a.get("action") in BETS:
                frac *= NARROW_SOFT + (1 - NARROW_SOFT) * rates[_phase(a.get("street"))]
        return max(MIN_RANGE, frac)


    # --- Equity -------------------------------------------------------------------
    def range_combos(known, board_cards, frac):
        """Opponent combos in the top `frac` of what they could hold: by preflop
        percentile before the flop, by made-hand strength on the board after."""
        combos = [c for c in ALL_COMBOS if c[0] not in known and c[1] not in known]
        if frac >= 1.0:
            return combos
        if board_cards:
            key = lambda c: evaluate([FULL_DECK[c[0]], FULL_DECK[c[1]]] + board_cards)
        else:
            key = COMBO_PCT.__getitem__
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


    # --- Decision -----------------------------------------------------------------
    def _safe_action(gs):
        return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


    def _bet_target(gs, pot_frac):
        """Total bet this street for a bet/raise of pot_frac of the pot on top of the current bet."""
        target = gs.get("current_bet", 0) + int(pot_frac * (gs["pot"] + gs.get("amount_owed", 0)))
        return max(target, gs.get("min_raise_to", 0))


    def _bet(gs, pot_frac):
        """Bet/raise pot_frac of the pot; shove if that's most of our stack."""
        max_total = gs["your_stack"] + gs.get("your_bet_this_street", 0)
        target = _bet_target(gs, pot_frac)
        if target >= 0.8 * max_total:
            return {"action": "all_in"}
        return {"action": "raise", "amount": target}


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


    def think(gs, deadline):
        street = gs["street"]
        hu = n_opponents(gs) == 1
        update_history(gs)
        counts = opponent_counts(gs)
        rates = {ph: estimate(*counts[ph], OPEN_DEFAULT, OPEN_K) for ph in ("pre", "post")}
        frac = opp_range(gs, rates)
        eq = strength(gs, frac, deadline)
        owed = gs.get("amount_owed", 0)
        pot_odds = owed / (gs["pot"] + owed) if owed else 0.0
        required = call_threshold(gs, owed, pot_odds)
        size = PREFLOP_RAISE_FRAC if street == "preflop" else random.uniform(*BET_SIZE_RANGE)

        info = {"equity": round(eq, 3), "range": round(frac, 2), "required": round(required, 3),
                "pot_odds": round(pot_odds, 3), "size": round(size, 2),
                "open_rate": {ph: round(r, 2) for ph, r in rates.items()},
                "hist_hands": len(_HISTORY)}

        # 1. Preflop fold rule: small blind, nobody has raised, bottom of the range.
        if (street == "preflop" and hu and PREFLOP and not gs.get("can_check")
                and gs.get("seat_to_act") == gs.get("sb_seat") and not _opp_raised(gs)
                and PREFLOP[preflop_key(gs["your_cards"])]["pct"] < SB_FOLD_PCT):
            act = {"action": "fold"}
            info["sb_fold"] = True
        # 2. Value bet/raise.
        elif eq >= RAISE_STRENGTH[street] and eq >= required:
            act = _bet(gs, size)
        else:
            act = None
            # 3. Bluff: postflop, nobody has bet, weak hand. P(bluff) = their fold rate,
            #    capped at twice break-even; the default fold rate is break-even.
            if street != "preflop" and hu and gs.get("can_check") and eq < BLUFF_MAX_EQ:
                breakeven = size / (1 + size)
                fold_rate = estimate(*counts["fold"], breakeven, FOLD_K)
                p_bluff = BLUFF_SCALE * min(fold_rate, 2 * breakeven)
                info.update(fold_rate=round(fold_rate, 2), p_bluff=round(p_bluff, 2))
                if (_bet_target(gs, size) < BLUFF_MAX_STACK * gs["your_stack"]
                        and random.random() < p_bluff):
                    act = _bet(gs, size)
                    info["bluff"] = True
            # 4. Otherwise: check, call on price, or fold.
            if act is None:
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

    return decide


SUBBOTS = [("neural_net", _make_neural_net()), ("all_in", _make_all_in()), ("hsr_modified_v4", _make_hsr_modified_v4())]


def current(hand_num):
    seg = (hand_num or 0) // SEGMENT_HANDS
    return seg, SUBBOTS[seg % len(SUBBOTS)]


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        for _, sub in SUBBOTS:                 # each sub-bot loads / warms up once
            try:
                sub(game_state)
            except Exception as e:
                print(f"warmup failed: {e!r}")
        return {"action": "check"}
    start = perf_counter()
    seg, (name, sub) = current(game_state.get("hand_num"))
    try:
        action = sub(game_state)
    except Exception as e:
        print(f"{name} failed: {e!r}")
        action = None
    if not isinstance(action, dict) or "action" not in action:
        action = {"action": "check"} if game_state.get("can_check") else {"action": "fold"}
    dbg = action.get("debug")
    action["debug"] = {"model": name, "segment": seg, "sub": dbg,
                       "think_ms": round((perf_counter() - start) * 1000, 1)}
    return action

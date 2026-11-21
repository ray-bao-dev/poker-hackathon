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

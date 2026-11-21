"""stabber: sparring bot that bets every postflop street it can, regardless of its cards.

Preflop: calls (limps / calls raises up to STAB_PREFLOP_CALL_BB), otherwise folds.
Postflop: when checked to, or first to act, bets STAB_FRAC of the pot. Facing a
bet or raise, calls if it costs at most CALL_FRAC of the pot, otherwise folds.

It exists to punish bots that fold too often to small bets.
"""
from time import perf_counter

STAB_FRAC = 0.5                 # bet size, fraction of the pot
CALL_FRAC = 0.35                # call raises that cost at most this share of the pot
STAB_PREFLOP_CALL_BB = 4        # preflop: call raises up to this many big blinds total


def _safe(gs):
    return {"action": "check"} if gs.get("can_check") else {"action": "fold"}


def think(gs):
    owed = gs.get("amount_owed", 0)
    pot = gs["pot"]
    if gs["street"] == "preflop":
        if gs.get("can_check"):
            return {"action": "check"}
        bb = gs["blinds"]["bb"] or 1
        return {"action": "call"} if gs.get("current_bet", 0) <= STAB_PREFLOP_CALL_BB * bb else {"action": "fold"}
    if gs.get("can_check"):
        return {"action": "raise", "amount": gs.get("current_bet", 0) + max(int(STAB_FRAC * pot), 1)}
    return {"action": "call"} if owed <= CALL_FRAC * pot else {"action": "fold"}


def decide(game_state):
    start = perf_counter()
    if game_state.get("type") == "warmup":
        return {"action": "check"}
    try:
        action = think(game_state)
    except Exception as e:
        print(f"think() failed: {e!r}")
        action = None
    if not isinstance(action, dict) or "action" not in action:
        action = _safe(game_state)
    action["debug"] = {"think_ms": round((perf_counter() - start) * 1000, 1)}
    return action

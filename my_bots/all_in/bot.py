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

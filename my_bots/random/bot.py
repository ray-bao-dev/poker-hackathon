"""Random bot: picks a uniformly random legal action each decision."""

import random  # seeded per match, so a seeded match replays exactly


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}

    stack = game_state["your_stack"]
    max_total = stack + game_state["your_bet_this_street"]   # biggest total bet we can make
    min_raise = game_state["min_raise_to"]

    if game_state["can_check"]:
        options = ["check"]
    else:
        options = ["fold", "call"]
    if stack > 0:
        options.append("all_in")
        if min_raise < max_total:                             # a non-all-in raise is possible
            options.append("raise")

    choice = random.choice(options)
    if choice == "raise":
        return {"action": "raise", "amount": random.randint(min_raise, max_total - 1)}
    return {"action": choice}

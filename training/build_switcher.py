"""Build my_bots/switcher/bot.py: one bot that rotates between other bots every few hands.

The tournament allows only bot.py in a bot folder (no sibling modules, no importlib/exec),
so each sub-bot's source is pasted into its own factory function. Its module globals
become that function's locals, so the sub-bots can't clash, and the factory returns the
sub-bot's decide(). Sub-bots must not use `global` (checked below). Their data files are
copied into switcher/data/ and must have distinct names.

    python3 training/build_switcher.py          # rebuild after changing a sub-bot
"""
import os
import shutil
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBBOTS = ["neural_net", "all_in", "hsr_modified_v4"]   # rotation order
SEGMENT_HANDS = 5
OUT_DIR = os.path.join(ROOT, "my_bots", "switcher")

HEADER = '''"""switcher: rotates between {names} every {n} hands (GENERATED, do not edit).

Built by training/build_switcher.py from the sub-bots' own bot.py files. Use it as a
sparring partner that changes style mid-match, e.g. to check that an opponent-modelling
bot (CFR) notices each switch and doesn't stay stuck on an exploit for the old style.

Hand h plays SUBBOTS[(h // SEGMENT_HANDS) % len(SUBBOTS)]. Every decision's debug says
which sub-bot acted ("model") and which segment it was ("segment").
"""
import random  # noqa: F401  (seeded per match; the sub-bots share it)
from time import perf_counter

SEGMENT_HANDS = {n}
'''

FOOTER = '''

SUBBOTS = [{pairs}]


def current(hand_num):
    seg = (hand_num or 0) // SEGMENT_HANDS
    return seg, SUBBOTS[seg % len(SUBBOTS)]


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        for _, sub in SUBBOTS:                 # each sub-bot loads / warms up once
            try:
                sub(game_state)
            except Exception as e:
                print(f"warmup failed: {{e!r}}")
        return {{"action": "check"}}
    start = perf_counter()
    seg, (name, sub) = current(game_state.get("hand_num"))
    try:
        action = sub(game_state)
    except Exception as e:
        print(f"{{name}} failed: {{e!r}}")
        action = None
    if not isinstance(action, dict) or "action" not in action:
        action = {{"action": "check"}} if game_state.get("can_check") else {{"action": "fold"}}
    dbg = action.get("debug")
    action["debug"] = {{"model": name, "segment": seg, "sub": dbg,
                       "think_ms": round((perf_counter() - start) * 1000, 1)}}
    return action
'''


def main():
    parts = [HEADER.format(names=", ".join(SUBBOTS), n=SEGMENT_HANDS)]
    data_dir = os.path.join(OUT_DIR, "data")
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    copied = {}
    for name in SUBBOTS:
        src_dir = os.path.join(ROOT, "my_bots", name)
        src = open(os.path.join(src_dir, "bot.py")).read()
        if any(line.lstrip().startswith(("global ", "from __future__")) for line in src.splitlines()):
            raise SystemExit(f"{name}: `global` / __future__ can't be wrapped in a function")
        fn = "_make_" + name
        parts.append(f"\n\n# {'=' * 76}\n# {name}  (from my_bots/{name}/bot.py)\n# {'=' * 76}\n"
                     f"def {fn}():\n{textwrap.indent(src, '    ')}\n    return decide\n")
        sub_data = os.path.join(src_dir, "data")
        for f in sorted(os.listdir(sub_data)) if os.path.isdir(sub_data) else []:
            if f in copied:
                raise SystemExit(f"data file {f} is in both {copied[f]} and {name}")
            copied[f] = name
            shutil.copy(os.path.join(sub_data, f), data_dir)
    pairs = ", ".join(f'("{n}", _make_{n}())' for n in SUBBOTS)
    parts.append(FOOTER.format(pairs=pairs))
    with open(os.path.join(OUT_DIR, "bot.py"), "w") as f:
        f.write("".join(parts))
    print(f"wrote {OUT_DIR}/bot.py with {SUBBOTS}, data: {sorted(copied)}")


if __name__ == "__main__":
    main()

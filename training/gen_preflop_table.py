"""Build my_bots/hsr/data/preflop.json: equity vs. one random hand for all 169
starting hands, plus each hand's percentile in that strict ordering
(fraction of the 1326 combos that are weaker).

    python training/gen_preflop_table.py [iters_per_hand]
"""
import importlib.util
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_DIR = os.path.join(ROOT, "my_bots", "hsr")

spec = importlib.util.spec_from_file_location("hsr_bot", os.path.join(BOT_DIR, "bot.py"))
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
random.seed(0)
R = bot.RANK_CHARS

hands = []   # (key, representative cards, combos)
for i in range(13):
    for j in range(i, 13):
        hi, lo = R[j], R[i]
        if i == j:
            hands.append((hi + lo, [hi + "s", lo + "h"], 6))
        else:
            hands.append((hi + lo + "s", [hi + "s", lo + "s"], 4))
            hands.append((hi + lo + "o", [hi + "s", lo + "h"], 12))

table = {}
for n, (key, cards, combos) in enumerate(hands, 1):
    eq = bot.equity(cards, [], 1, deadline=float("inf"), max_iters=iters)
    table[key] = {"equity": round(eq, 4), "combos": combos}
    print(f"{n:3d}/169 {key:4s} {eq:.4f}", flush=True)

ordered = sorted(table, key=lambda k: table[k]["equity"])
weaker = 0
for k in ordered:
    table[k]["pct"] = round(weaker / 1326, 4)
    weaker += table[k]["combos"]
    table[k]["rank"] = 169 - ordered.index(k)        # 1 = strongest

os.makedirs(os.path.join(BOT_DIR, "data"), exist_ok=True)
out = os.path.join(BOT_DIR, "data", "preflop.json")
with open(out, "w") as f:
    json.dump(table, f, indent=1, sort_keys=True)
print("wrote", out)

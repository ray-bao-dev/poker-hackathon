"""Train the neural_net bot from PHH hand histories.

Replays every no-limit hold'em hand in the zip, turns each decision made with
known hole cards into features (using my_bots/neural_net/bot.py's own feature
code) and a label (fold / call / raise_half / raise_pot / all_in), trains a
small MLP and saves its weights to my_bots/neural_net/data/model.npz.

Run from the repo root, in the Python 3.10 environment:
    python training/train_nn.py ../dataset/poker-hand-histories.zip
"""
import ast
import importlib.util
import os
import sys
import zipfile

import numpy as np
from sklearn.neural_network import MLPClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_DIR = os.path.join(ROOT, "my_bots", "neural_net")
OUT = os.path.join(BOT_DIR, "data", "model.npz")

_spec = importlib.util.spec_from_file_location("nnbot", os.path.join(BOT_DIR, "bot.py"))
bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bot)

STREETS = ["preflop", "flop", "turn", "river"]


def parse_phh(text):
    """Minimal PHH (TOML subset) reader: `key = value` lines."""
    d = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v in ("true", "false"):
            d[k.strip()] = v == "true"
            continue
        try:
            d[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
    return d


def decisions(hand, eq_cache):
    """Yield (features, label) for every decision with known hole cards."""
    if hand.get("variant") != "NT" or "actions" not in hand:
        return
    n = len(hand["starting_stacks"])
    stack = list(hand["starting_stacks"])
    blinds = list(hand["blinds_or_straddles"])
    antes = list(hand.get("antes") or [0] * n)
    if n == 2:                                   # PHH: heads-up blinds are reverse-assigned
        blinds, antes = blinds[::-1], antes[::-1]
    bb = max(blinds) or 1
    pot = 0
    bet = [0] * n
    for i in range(n):
        a = min(antes[i], stack[i]); stack[i] -= a; pot += a
        b = min(blinds[i], stack[i]); stack[i] -= b; bet[i] = b
    folded = [False] * n
    hole = [None] * n
    board = []
    street = 0
    raises = 0

    for raw in hand["actions"]:
        act = raw.split("#")[0].split()
        if not act:
            continue
        if act[0] == "d":
            if act[1] == "dh":
                c = act[3]
                hole[int(act[2][1:]) - 1] = None if "?" in c else [c[0:2], c[2:4]]
            elif act[1] == "db":
                c = act[2]
                board += [c[i:i + 2] for i in range(0, len(c), 2)]
                pot += sum(bet); bet = [0] * n
                street = min(street + 1, 3); raises = 0
            continue

        p = int(act[0][1:]) - 1
        kind = act[1]
        if kind not in ("f", "cc", "cbr"):
            continue
        cur = max(bet)
        owed = min(cur - bet[p], stack[p])
        live = [i for i in range(n) if not folded[i]]
        others = [i for i in live if i != p]
        total_pot = pot + sum(bet)

        if kind == "f":
            label = 0
        elif kind == "cc":
            label = 1
        else:
            to = int(act[2])
            if to >= bet[p] + stack[p]:
                label = 4
            else:
                frac = (to - cur) / max(total_pot + owed, 1)
                label = 2 if frac <= 0.75 else 3

        if hole[p] is not None and others:
            key = (tuple(sorted(hole[p])), tuple(board))
            if key not in eq_cache:
                eq_cache[key] = bot.equity(hole[p], board)
            eff_behind = min(stack[p], max(stack[i] for i in others))
            x = bot.features(STREETS[street], eq_cache[key], total_pot, owed, eff_behind, bb,
                             len(others), p == max(live), raises)
            yield x, label

        if kind == "f":
            folded[p] = True
        elif kind == "cc":
            stack[p] -= owed; bet[p] += owed
        else:
            put = min(int(act[2]) - bet[p], stack[p])
            stack[p] -= put; bet[p] += put; raises += 1


def main(zip_path):
    X, y, eq_cache, hands = [], [], {}, 0
    with zipfile.ZipFile(zip_path) as z:
        names = [i.filename for i in z.infolist() if i.filename.endswith(".phh")]
        for k, name in enumerate(names):
            hand = parse_phh(z.read(name).decode("utf-8", "replace"))
            got = list(decisions(hand, eq_cache))
            if got:
                hands += 1
                for xi, yi in got:
                    X.append(xi); y.append(yi)
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(names)} files, {len(X)} decisions")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    print(f"{hands} hands, {len(X)} decisions; label counts:",
          {bot.ACTIONS[c]: int((y == c).sum()) for c in range(len(bot.ACTIONS))})

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, early_stopping=True,
                        random_state=0)
    clf.fit((X - mean) / std, y)
    print(f"validation accuracy {clf.best_validation_score_:.3f} "
          f"(always-most-common baseline {np.bincount(y).max() / len(y):.3f})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arrays = {"mean": mean, "std": std, "classes": clf.classes_, "n_layers": len(clf.coefs_)}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    np.savez(OUT, **arrays)
    print(f"saved {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

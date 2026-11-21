"""Train the neural_netv2 bot from the v2 PHH hand histories.

Same idea as train_nn.py (features from my_bots/neural_netv2/bot.py, labels
fold / call / raise_half / raise_pot / all_in, small MLP saved to
my_bots/neural_netv2/data/model.npz), extended for the v2 zip:

  * .phhs files (many hands per file, `[1]`, `[2]`, ... sections) as well as .phh
  * decimal stacks and bet sizes (HandHQ cash games)
  * hole cards revealed at showdown (`p3 sm AhKd`) count as known hole cards.
    HandHQ deals every hand as `????`, so its only usable decisions come from
    players who reached showdown -- those never fold, so HandHQ adds no fold
    labels. All fold examples come from Pluribus/WSOP, where every hand is known.
  * parallel over files; HandHQ is huge (~16 GB), so only a random sample of
    its files is used by default.

Run from the repo root, in the Python 3.10 environment:
    python training/train_nn_v2.py my_bots/neural_net/data/poker-hand-historiesv2.zip
Options:
    --handhq-files N   HandHQ .phhs files to sample (default 1500; 0 = none, -1 = all)
    --workers N        processes (default: CPU count - 1)
    --hidden 64,64     MLP hidden layer sizes
    --seed N           sampling / training seed (default 0)
"""
import argparse
import ast
import importlib.util
import os
import random
import re
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_DIR = os.path.join(ROOT, "my_bots", "neural_netv2")
OUT = os.path.join(BOT_DIR, "data", "model.npz")

STREETS = ["preflop", "flop", "turn", "river"]
RANKS = "23456789TJQKA"
CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
HAND_SPLIT = re.compile(r"^\[\d+\]\s*$", re.M)


def load_bot():
    spec = importlib.util.spec_from_file_location("nnbot_v2", os.path.join(BOT_DIR, "bot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def split_hands(name, text):
    if name.endswith(".phhs"):
        return [t for t in HAND_SPLIT.split(text) if "actions" in t]
    return [text]


def cards_of(s):
    pair = [s[0:2], s[2:4]]
    return pair if len(s) == 4 and all(CARD_RE.match(c) for c in pair) else None


# ── Worker state ──────────────────────────────────────────────────────────────
_bot = _zip = None
_eq_cache = {}


def _init_worker(zip_path):
    global _bot, _zip
    _bot = load_bot()
    _zip = zipfile.ZipFile(zip_path)


def _equity(hole, board):
    if not board:   # preflop equity vs a random hand only depends on ranks + suitedness
        r = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
        key = (r[0], r[1], hole[0][1] == hole[1][1])
    else:
        key = (tuple(sorted(hole)), tuple(board))
    if key not in _eq_cache:
        _eq_cache[key] = _bot.equity(hole, board)
    return _eq_cache[key]


def decisions(hand):
    """Yield (features, label) for every decision with known hole cards."""
    if hand.get("variant") != "NT" or "actions" not in hand or "starting_stacks" not in hand:
        return
    n = len(hand["starting_stacks"])
    stack = [float(s) for s in hand["starting_stacks"]]
    blinds = [float(b) for b in hand.get("blinds_or_straddles") or [0] * n]
    antes = [float(a) for a in hand.get("antes") or [0] * n]
    if n == 2:                                   # PHH: heads-up blinds are reverse-assigned
        blinds, antes = blinds[::-1], antes[::-1]
    bb = max(blinds) or 1.0

    # Hole cards: dealt face up (Pluribus/WSOP) or shown at showdown (HandHQ).
    hole = [None] * n
    for raw in hand["actions"]:
        act = raw.split("#")[0].split()
        if len(act) >= 4 and act[0] == "d" and act[1] == "dh":
            hole[int(act[2][1:]) - 1] = cards_of(act[3])
        elif len(act) >= 3 and act[1] == "sm" and act[0].startswith("p"):
            c = cards_of(act[2])
            if c:
                hole[int(act[0][1:]) - 1] = c
    if not any(hole):
        return

    pot = 0.0
    bet = [0.0] * n
    for i in range(n):
        a = min(antes[i], stack[i]); stack[i] -= a; pot += a
        b = min(blinds[i], stack[i]); stack[i] -= b; bet[i] = b
    folded = [False] * n
    board = []
    street = 0
    raises = 0

    for raw in hand["actions"]:
        act = raw.split("#")[0].split()
        if len(act) < 2:
            continue
        if act[0] == "d":
            if act[1] == "db" and len(act) >= 3:
                c = act[2]
                board += [c[i:i + 2] for i in range(0, len(c), 2)]
                pot += sum(bet); bet = [0.0] * n
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
            to = float(act[2])
            if to >= bet[p] + stack[p] - 1e-6:
                label = 4
            else:
                frac = (to - cur) / max(total_pot + owed, 1e-9)
                label = 2 if frac <= 0.75 else 3

        if hole[p] is not None and others and len(board) in (0, 3, 4, 5):
            eff_behind = min(stack[p], max(stack[i] for i in others))
            x = _bot.features(STREETS[street], _equity(hole[p], board), total_pot, owed,
                              eff_behind, bb, len(others), p == max(live), raises)
            yield x, label

        if kind == "f":
            folded[p] = True
        elif kind == "cc":
            stack[p] -= owed; bet[p] += owed
        else:
            put = min(float(act[2]) - bet[p], stack[p])
            stack[p] -= put; bet[p] += put; raises += 1


def process_file(name):
    """Return (source, n_hands_used, X, y) for one zip member."""
    X, y, used = [], [], 0
    text = _zip.read(name).decode("utf-8", "replace")
    for chunk in split_hands(name, text):
        try:
            got = list(decisions(parse_phh(chunk)))
        except (ValueError, IndexError, KeyError, TypeError):
            continue                                   # malformed hand: skip it
        if got:
            used += 1
            for xi, yi in got:
                X.append(xi); y.append(yi)
    src = "handhq" if "/handhq/" in name else "pluribus" if "/pluribus/" in name else "other"
    return (src, used, np.asarray(X, dtype=np.float32).reshape(-1, 13),
            np.asarray(y, dtype=np.int8))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_path")
    ap.add_argument("--handhq-files", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bot = load_bot()
    with zipfile.ZipFile(args.zip_path) as z:
        all_names = [i.filename for i in z.infolist() if not i.is_dir()]
    small = [n for n in all_names if n.endswith(".phh")]
    handhq = sorted(n for n in all_names if n.endswith(".phhs"))
    if 0 <= args.handhq_files < len(handhq):
        handhq = random.Random(args.seed).sample(handhq, args.handhq_files)
    names = handhq + small            # big files first so workers stay busy at the end
    print(f"{len(small)} .phh files + {len(handhq)} .phhs files, {args.workers} workers")

    Xs, ys = [], []
    stats = {}                        # source -> [hands, decisions, label counts]
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.zip_path,)) as pool:
        for k, (src, used, X, y) in enumerate(pool.imap_unordered(process_file, names, chunksize=4)):
            s = stats.setdefault(src, [0, 0, np.zeros(5, dtype=np.int64)])
            s[0] += used; s[1] += len(y); s[2] += np.bincount(y, minlength=5)
            if len(y):
                Xs.append(X); ys.append(y)
            if (k + 1) % 200 == 0 or k + 1 == len(names):
                total = sum(v[1] for v in stats.values())
                print(f"  {k + 1}/{len(names)} files, {total} decisions, {time.time() - t0:.0f}s")

    for src, (hands, dec, counts) in sorted(stats.items()):
        print(f"{src:9s} {hands} hands, {dec} decisions; labels",
              {bot.ACTIONS[c]: int(counts[c]) for c in range(5)})
    X = np.concatenate(Xs).astype(float)
    y = np.concatenate(ys).astype(int)

    from sklearn.neural_network import MLPClassifier
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    hidden = tuple(int(h) for h in args.hidden.split(","))
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=200, early_stopping=True,
                        batch_size=1024, random_state=args.seed, verbose=True)
    print(f"training MLP {hidden} on {len(X)} decisions...")
    clf.fit((X - mean) / std, y)
    print(f"validation accuracy {clf.best_validation_score_:.3f} "
          f"(always-most-common baseline {np.bincount(y).max() / len(y):.3f})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arrays = {"mean": mean, "std": std, "classes": clf.classes_, "n_layers": len(clf.coefs_)}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    np.savez(OUT, **arrays)
    print(f"saved {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main()

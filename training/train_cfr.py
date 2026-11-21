"""Train my_bots/CFR: Linear external-sampling Monte Carlo CFR on an abstracted heads-up
No-Limit Hold'em game, following "Deep Counterfactual Regret Minimization" (Brown et al.,
ICML 2019), sections 2.1-2.2 and 5.3:

  - external sampling: on each iteration one player is the traverser; at the traverser's
    infosets every action is explored, while chance and the opponent sample one action
    (Algorithm 2 of the paper, with tables in place of the networks);
  - regret matching (eq. 4), playing the highest-regret action when no regret is positive
    (the paper finds this copes better with sampling error than playing uniformly);
  - Linear CFR: iteration t is weighted by t, for regrets and for the average strategy;
  - the average strategy is accumulated at the opponent's infosets during each traversal,
    like the paper's strategy memory, and is what the bot plays.

The abstraction (card buckets, bet sizes, infoset keys) is imported from my_bots/CFR/bot.py
so training and play always agree.

    python training/train_cfr.py --deals 300000 --minutes 20 --workers 9

Steps: (1) deal a pool of random hands and compute each hand's strength on every street
(cached in training/cache/), (2) set equal-frequency bucket thresholds, (3) run CFR workers
in parallel for the time given, (4) sum their average strategies (in a two-player zero-sum
game a mixture of equilibrium strategies is still an equilibrium) and write
my_bots/CFR/data/cfr.json.
"""
import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_DIR = os.path.join(ROOT, "my_bots", "CFR")
CACHE = os.path.join(ROOT, "training", "cache")

spec = importlib.util.spec_from_file_location("cfr_bot", os.path.join(BOT_DIR, "bot.py"))
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

import eval7  # noqa: E402

EQ_BUCKETS = {1: 12, 2: 12, 3: 16}      # equity buckets per street (flop/turn add a draw split)
DRAW_QUANTILE = 0.80                     # the top 20% of (equity - current strength) are draws
POOL_ITERS = 1500                        # Monte Carlo samples per flop/turn equity in the pool

DECK = [r + s for r in bot.RANKS for s in "cdhs"]
CLASSES = sorted({bot.hand_class([a, b]) for i, a in enumerate(DECK) for b in DECK[i + 1:]})


# --- (1) The deal pool --------------------------------------------------------------
def _deal_chunk(args):
    seed, n = args
    rng = random.Random(seed)
    cards = [eval7.Card(c) for c in DECK]
    out = np.zeros((n, 15), dtype=np.float32)
    # columns: classA, classB, win(A), eqA[1..3], hsA[1..3], eqB[1..3], hsB[1..3]
    for r in range(n):
        idx = rng.sample(range(52), 9)
        a, b, board = [cards[i] for i in idx[:2]], [cards[i] for i in idx[2:4]], [cards[i] for i in idx[4:]]
        out[r, 0] = CLASSES.index(bot.hand_class([DECK[idx[0]], DECK[idx[1]]]))
        out[r, 1] = CLASSES.index(bot.hand_class([DECK[idx[2]], DECK[idx[3]]]))
        va, vb = eval7.evaluate(a + board), eval7.evaluate(b + board)
        out[r, 2] = 1 if va > vb else (-1 if va < vb else 0)
        for s, nb in ((1, 3), (2, 4), (3, 5)):
            eqa, hsa = bot.raw_strength(a, board[:nb], POOL_ITERS)
            eqb, hsb = bot.raw_strength(b, board[:nb], POOL_ITERS)
            out[r, 2 + s], out[r, 5 + s] = eqa, hsa
            out[r, 8 + s], out[r, 11 + s] = eqb, hsb
    return out


def build_pool(n, workers):
    path = os.path.join(CACHE, "pool_%d.npy" % n)
    if os.path.exists(path):
        print("pool: using", path)
        return np.load(path)
    os.makedirs(CACHE, exist_ok=True)
    chunk = 5000
    jobs = [(1000 + i, min(chunk, n - i * chunk)) for i in range(math.ceil(n / chunk))]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        parts = pool.map(_deal_chunk, jobs)
    arr = np.concatenate(parts)
    np.save(path, arr)
    print("pool: %d deals in %.0fs -> %s" % (n, time.time() - t0, path))
    return arr


# --- (2) Buckets --------------------------------------------------------------------
def make_meta(pool):
    eq_bounds, draw_split = {}, {}
    for s in (1, 2, 3):
        eq = np.concatenate([pool[:, 2 + s], pool[:, 8 + s]])
        k = EQ_BUCKETS[s]
        eq_bounds[str(s)] = [round(float(x), 5) for x in np.quantile(eq, [i / k for i in range(1, k)])]
        if s < 3:
            diff = np.concatenate([pool[:, 2 + s] - pool[:, 5 + s], pool[:, 8 + s] - pool[:, 11 + s]])
            draw_split[str(s)] = round(float(np.quantile(diff, DRAW_QUANTILE)), 5)
    return {"eq_bounds": eq_bounds, "draw_split": draw_split}


def bucketize(pool, meta):
    """Each deal -> (bucketsA[4], bucketsB[4], winA) with buckets as strings."""
    m = meta
    deals = []
    for row in pool:
        ba = [CLASSES[int(row[0])]] + [bot.bucket_from_raw(s, row[2 + s], row[5 + s], m) for s in (1, 2, 3)]
        bb = [CLASSES[int(row[1])]] + [bot.bucket_from_raw(s, row[8 + s], row[11 + s], m) for s in (1, 2, 3)]
        deals.append((ba, bb, int(row[2])))
    return deals


# --- (3) CFR on the abstract game -----------------------------------------------------
# State: (s, seq, stk0, stk1, bet0, bet1, actor, nacts, nraises, last_raise, init, aggr, depth)
# Units are big blinds. Player 0 is the button / small blind, player 1 the big blind.

def stack_depth(rng):
    """Effective stacks, in big blinds, roughly as a 10,000-chip freezeout visits them."""
    return math.exp(rng.uniform(math.log(4), math.log(220)))


def root(S):
    return (0, "", S - 0.5, S - 1.0, 0.5, 1.0, 0, 0, 0, 1.0, -1, -1, bot.classify(S, bot.PRE_DEPTH))


def child(st, code, target, S):
    """Apply an action. Returns ('T', payoff to player 0) or a new state."""
    s, seq, k0, k1, b0, b1, i, nacts, nr, lr, init, aggr, depth = st
    stk = [k0, k1]
    bet = [b0, b1]
    cur = b0 if b0 > b1 else b1
    if code == "f":
        return ("T", -(S - stk[0]) if i == 0 else (S - stk[1]))
    if code == "c":
        pay = min(cur - bet[i], stk[i])
        stk[i] -= pay
        bet[i] += pay
    else:
        if code == "a":
            total = min(stk[0] + bet[0], stk[1] + bet[1])
        else:
            total = target
        pay = total - bet[i]
        stk[i] -= pay
        bet[i] = total
        if total - cur >= lr:
            lr = total - cur
        if code != "a":
            nr += 1
        aggr = i
    nacts += 1
    seq += code
    stk = [round(stk[0], 6), round(stk[1], 6)]           # no float drift: bets must match exactly
    bet = [round(bet[0], 6), round(bet[1], 6)]
    if abs(bet[0] - bet[1]) < 1e-6 and nacts >= 2:       # street closed
        if s == 3 or stk[0] <= 1e-6 or stk[1] <= 1e-6:
            return ("S", S - stk[0])                     # showdown, each player has this much in
        pot = 2 * S - stk[0] - stk[1]
        spr = min(stk) / pot
        return (s + 1, "", stk[0], stk[1], 0.0, 0.0, 1, 0, 0, 1.0, aggr, -1,
                bot.classify(spr, bot.POST_SPR))
    return (s, seq, stk[0], stk[1], bet[0], bet[1], 1 - i, nacts, nr, lr, init, aggr, depth)


def actions_at(st, S):
    s, seq, k0, k1, b0, b1, i, nacts, nr, lr, init, aggr, depth = st
    stk, bet = (k0, k1), (b0, b1)
    cur = b0 if b0 > b1 else b1
    pot = 2 * S - k0 - k1
    owed = cur - bet[i]
    if owed < 1e-6:
        owed = 0.0
    return bot.abstract_actions(s, owed, pot, cur, cur + lr,
                                stk[i] + bet[i], stk[1 - i] + bet[1 - i], nr)


def key_of(st, buckets):
    s, seq, k0, k1, b0, b1, i, nacts, nr, lr, init, aggr, depth = st
    ini = "n" if init < 0 else ("m" if init == i else "o")
    return bot.infoset_key(s, i, depth, ini, buckets[i][s], seq)


def regret_matching(reg):
    pos = [r if r > 0 else 0.0 for r in reg]
    tot = sum(pos)
    if tot > 0:
        return [x / tot for x in pos]
    if not any(reg):                                     # never updated: play uniform
        return [1.0 / len(reg)] * len(reg)
    best = max(range(len(reg)), key=reg.__getitem__)     # all regrets <= 0: the best one
    return [1.0 if j == best else 0.0 for j in range(len(reg))]


class Solver:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.table = {}      # key -> [codes list, regrets list, strategy sums list]

    def node(self, key, codes):
        nd = self.table.get(key)
        if nd is None:
            nd = self.table[key] = [list(codes), [0.0] * len(codes), [0.0] * len(codes)]
            return nd, list(range(len(codes)))
        idx = []
        for c in codes:
            try:
                idx.append(nd[0].index(c))
            except ValueError:                           # an action this infoset hasn't had
                nd[0].append(c)
                nd[1].append(0.0)
                nd[2].append(0.0)
                idx.append(len(nd[0]) - 1)
        return nd, idx

    def traverse(self, st, p, t, buckets, win, S):
        if st[0] == "T":
            return st[1] if p == 0 else -st[1]
        if st[0] == "S":
            v = win * st[1]
            return v if p == 0 else -v
        acts = actions_at(st, S)
        nd, idx = self.node(key_of(st, buckets), [c for c, _ in acts])
        reg = nd[1]
        sigma = regret_matching([reg[j] for j in idx])
        if st[6] == p:                                   # traverser: explore every action
            vals = [self.traverse(child(st, c, tg, S), p, t, buckets, win, S) for c, tg in acts]
            ev = sum(sg * v for sg, v in zip(sigma, vals))
            for j, v in zip(idx, vals):
                reg[j] += t * (v - ev)                   # Linear CFR: weight t
            return ev
        ss = nd[2]                                       # opponent: accumulate the average
        for j, sg in zip(idx, sigma):                    # strategy, then sample one action
            ss[j] += t * sg
        r, acc, k = self.rng.random(), 0.0, len(acts) - 1
        for n, sg in enumerate(sigma):
            acc += sg
            if r < acc:
                k = n
                break
        c, tg = acts[k]
        return self.traverse(child(st, c, tg, S), p, t, buckets, win, S)


def _worker(args):
    seed, seconds, deals = args
    sys.setrecursionlimit(10000)
    solver = Solver(seed)
    rng = solver.rng
    t, t_end = 0, time.time() + seconds
    while time.time() < t_end:
        for _ in range(200):
            t += 1
            for p in (0, 1):                             # alternating updates
                a, b, w = deals[rng.randrange(len(deals))]
                if rng.random() < 0.5:
                    buckets, win = (a, b), w
                else:
                    buckets, win = (b, a), -w
                S = stack_depth(rng)
                solver.traverse(root(S), p, t, buckets, win, S)
    return t, {k: (v[0], v[2]) for k, v in solver.table.items()}


# --- (4) Merge and write ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=300000)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(BOT_DIR, "data", "cfr.json"))
    args = ap.parse_args()

    pool = build_pool(args.deals, args.workers)
    meta = make_meta(pool)
    deals = bucketize(pool, meta)
    print("buckets:", json.dumps(meta)[:300], "...")

    print("training %d workers for %.1f min" % (args.workers, args.minutes))
    t0 = time.time()
    jobs = [(7 + w, args.minutes * 60, deals) for w in range(args.workers)]
    with mp.Pool(args.workers) as p:
        results = p.map(_worker, jobs)
    iters = sum(r[0] for r in results)
    print("done: %d iterations in %.0fs" % (iters, time.time() - t0))

    merged = {}
    for _, table in results:
        for key, (codes, ss) in table.items():
            m = merged.setdefault(key, {})
            for c, x in zip(codes, ss):
                m[c] = m.get(c, 0.0) + x
    strategy = {}
    for key, m in merged.items():
        tot = sum(m.values())
        if tot <= 0:
            continue
        codes = [c for c in "fchpa" if c in m]
        strategy[key] = ["".join(codes), [round(m[c] / tot, 3) for c in codes]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {"meta": meta, "strategy": strategy,
           "info": {"iterations": iters, "workers": args.workers, "deals": args.deals,
                    "minutes": args.minutes, "trained": time.strftime("%Y-%m-%d %H:%M")}}
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote %s: %d infosets, %.1f MB" % (args.out, len(strategy), os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()

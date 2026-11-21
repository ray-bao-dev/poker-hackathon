#!/usr/bin/env python3
"""Play many matches between the same bots across different seeds and write a report.

Same options as play.py, plus:
  --runs N     matches to play (default 20), seeds base, base+1, ..., base+N-1
  --swap       also replay every seed with the seat order reversed (same cards, seats
               swapped), like the tournament's duplicate pairs
  --out DIR    where reports go (default results/)
  --logs       also save every match's full event log (.jsonl) in the report folder

  python3 simulate.py my_bots/ALL_IN bots/caller --runs 50 --seed 1
  python3 simulate.py my_bots/ALL_IN my_bots/neural_net --runs 25 --swap

Each run writes results/<time>_<bots>/ with:
  matches.csv    one row per bot per match
  summary.json   totals per bot
  summary.txt    the same summary, as printed at the end
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from play import Bot, Log
from pokercore import core_hash, structures
from pokercore.seats import BotSeat
from pokercore.seeds import derive_seed, new_seed
from pokercore.table import MODES, MatchConfig, play_match

CSV_FIELDS = ["match", "seed", "swapped", "bot", "seat", "place", "final_stack", "delta",
              "hands", "bb_per_100", "busted_on_hand", "avg_think_ms", "max_think_ms",
              "timeouts", "errors", "restarts", "stopped"]


def make_bots(paths):
    """Same naming as play.py: folder name, made unique with _2, _3, ..."""
    bots, names = [], set()
    for path in paths:
        bp = Path(path)
        if bp.name == "bot.py":
            base = bp.resolve().parent.name
        elif bp.suffix in (".py", ".zip"):
            base = bp.stem
        else:
            base = bp.resolve().name
        name, k = base, 2
        while name in names:
            name, k = f"{base}_{k}", k + 1
        names.add(name)
        bots.append(Bot(name, path))
    return bots


def run_match(bots, seed, args, log_path=None):
    config = MatchConfig(mode=args.mode, hands=args.hands, starting_stack=args.stack,
                         structure=structures.parse(args.blinds, args.stack), seed=seed,
                         time_limit_ms=args.time_limit_ms)
    seats = [BotSeat(b, seed=derive_seed(seed, "bot", b.name), time_limit_ms=args.time_limit_ms,
                     use_docker=args.docker) for b in bots]
    log = Log(out=log_path)
    try:
        result = play_match(seats, config, f"sim-{seed}", log)
    finally:
        log.close()
    problems = {s.name: (s.summary().get("errors_list") or [])[:5] for s in seats}
    return result, problems


def summarise(rows, names, n_matches):
    out = {}
    for name in names:
        r = [x for x in rows if x["bot"] == name]
        if not r:
            continue
        n = len(r)
        out[name] = {
            "matches": n,
            "wins": sum(1 for x in r if x["place"] == 1),
            "win_rate": round(sum(1 for x in r if x["place"] == 1) / n, 3),
            "total_delta": sum(x["delta"] for x in r),
            "avg_delta": round(sum(x["delta"] for x in r) / n, 1),
            "avg_bb_per_100": round(sum(x["bb_per_100"] for x in r) / n, 2),
            "avg_hands": round(sum(x["hands"] for x in r) / n, 1),
            "avg_think_ms": round(sum(x["avg_think_ms"] or 0 for x in r) / n, 1),
            "max_think_ms": max((x["max_think_ms"] or 0) for x in r),
            "timeouts": sum(x["timeouts"] or 0 for x in r),
            "errors": sum(x["errors"] or 0 for x in r),
            "restarts": sum(x["restarts"] or 0 for x in r),
        }
    return {"matches_played": n_matches, "bots": out}


def summary_text(summary, header):
    lines = [header, f"{summary['matches_played']} matches", ""]
    lines.append(f"{'bot':<20} {'wins':>9} {'win%':>6} {'avg delta':>10} {'bb/100':>8} "
                 f"{'hands':>6} {'think ms':>9} {'max ms':>7} {'t/o':>4} {'err':>4}")
    for name, s in sorted(summary["bots"].items(), key=lambda kv: -kv[1]["win_rate"]):
        lines.append(f"{name:<20} {s['wins']:>4}/{s['matches']:<4} {s['win_rate'] * 100:>5.1f} "
                     f"{s['avg_delta']:>+10,.0f} {s['avg_bb_per_100']:>+8.1f} {s['avg_hands']:>6.0f} "
                     f"{s['avg_think_ms']:>9.0f} {s['max_think_ms']:>7.0f} {s['timeouts']:>4} {s['errors']:>4}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bots", nargs="+", help="2-9 bot folders, bot.py files or .zip archives")
    p.add_argument("--mode", choices=MODES, default="freezeout")
    p.add_argument("--hands", type=int, default=2000, help="hands to play (fixed/cash) or safety cap (freezeout)")
    p.add_argument("--blinds", default="standard", help="turbo, standard (default), slow, or flat like 50/100")
    p.add_argument("--stack", type=int, default=10_000)
    p.add_argument("--seed", type=int, help="first seed (default: random)")
    p.add_argument("--docker", action="store_true", help="run bots in the Docker sandbox")
    p.add_argument("--time-limit-ms", type=int, default=None)
    p.add_argument("--runs", type=int, default=20, help="number of seeds to play (default 20)")
    p.add_argument("--swap", action="store_true", help="replay each seed with the seat order reversed")
    p.add_argument("--out", default="results", help="report folder (default results/)")
    p.add_argument("--logs", action="store_true", help="save every match's event log too")
    args = p.parse_args(argv)

    if not 2 <= len(args.bots) <= 9:
        p.error("a table needs 2-9 bots")
    if args.runs < 1:
        p.error("--runs must be at least 1")

    bots = make_bots(args.bots)
    names = [b.name for b in bots]
    base = args.seed if args.seed is not None else new_seed() % 1_000_000
    report_dir = Path(args.out) / f"{time.strftime('%Y%m%d-%H%M%S')}_{'_vs_'.join(names)}"
    report_dir.mkdir(parents=True, exist_ok=True)

    header = (f"{' vs '.join(names)}: {args.mode}, blinds {args.blinds}, stack {args.stack:,}, "
              f"seeds {base}..{base + args.runs - 1}" + (", swapped seats too" if args.swap else "")
              + (", Docker sandbox" if args.docker else "") + f"  (core {core_hash()})")
    print(header)
    (report_dir / "config.json").write_text(json.dumps({**vars(args), "seed": base, "bot_names": names}, indent=2))

    rows, n = [], 0
    csv_file = open(report_dir / "matches.csv", "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    t0 = time.time()
    try:
        for i in range(args.runs):
            seed = base + i
            for swapped in ([False, True] if args.swap else [False]):
                order = list(reversed(bots)) if swapped else bots
                log_path = report_dir / f"match_{seed}{'_swap' if swapped else ''}.jsonl" if args.logs else None
                result, problems = run_match(order, seed, args, log_path)
                n += 1
                seat_of = {b.name: k for k, b in enumerate(order)}
                for r in result["results"]:
                    row = {"match": n, "seed": seed, "swapped": swapped, "bot": r["bot"],
                           "seat": seat_of[r["bot"]], "place": r["place"], "final_stack": r["final_stack"],
                           "delta": r["delta"], "hands": r["hands_played"], "bb_per_100": r["bb_per_100"],
                           "busted_on_hand": r["busted_on_hand"], "avg_think_ms": r.get("avg_think_ms"),
                           "max_think_ms": r.get("max_think_ms"), "timeouts": r.get("timeouts"),
                           "errors": r.get("errors"), "restarts": r.get("restarts"),
                           "stopped": result["stopped"]}
                    rows.append(row)
                    writer.writerow(row)
                csv_file.flush()
                winner = next((r["bot"] for r in result["results"] if r["place"] == 1), "-")
                issues = {k: v for k, v in problems.items() if v}
                print(f"  [{n}] seed {seed}{' (swapped)' if swapped else ''}: {winner} wins "
                      f"in {result['hands']} hands" + (f"  problems: {issues}" if issues else ""))
    except KeyboardInterrupt:
        print("\nstopped early; writing the report for the matches played so far")
    finally:
        csv_file.close()
        summary = summarise(rows, names, n)
        summary["duration_s"] = round(time.time() - t0, 1)
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        text = summary_text(summary, header)
        (report_dir / "summary.txt").write_text(text + "\n")
        print("\n" + text)
        print(f"\nreport: {report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

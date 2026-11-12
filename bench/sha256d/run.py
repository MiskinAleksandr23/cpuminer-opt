#!/usr/bin/env python3
"""Build and compare the four SHA256d AVX-512 configurations on a Linux CPU."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import secrets
import statistics
import subprocess
import sys

from build import build_variants, check_cpu_flags

REPO = Path(__file__).resolve().parents[2]
VARIANTS = ("base", "rounds", "ternary", "combined")
METER = re.compile(r"^SHA256D_BENCH hashes=(\d+) seconds=(\d+\.\d+)$", re.MULTILINE)
# Two-sided 95% Student t quantiles, rounded up; df >= 30 uses the df=30 value.
T_975 = (0, 12.707, 4.303, 3.183, 2.777, 2.571, 2.447, 2.365, 2.307,
         2.263, 2.229, 2.201, 2.179, 2.161, 2.145, 2.132, 2.120, 2.110,
         2.101, 2.094, 2.087, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
         2.052, 2.049, 2.046, 2.043)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def orders(rounds, rng):
    # Each cycle balances position and each ordered pair of adjacent variants.
    square = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    result = []
    while len(result) < rounds:
        labels = list(VARIANTS)
        rng.shuffle(labels)
        rows = list(square)
        rng.shuffle(rows)
        result.extend([[labels[i] for i in row] for row in rows])
    return result[:rounds]


def cpu_ticks(cpu):
    # Guest and guest_nice are already counted in user and nice.
    with open("/proc/stat") as proc:
        for line in proc:
            row = line.split()
            if row and row[0] == f"cpu{cpu}" and len(row) >= 9:
                ticks = list(map(int, row[1:9]))
                return sum(ticks), ticks[7]
    return None


def measure(binary, variant, duration, cpu, log, phase, round_no, position):
    argv = [str(binary), "-a", "sha256d", "-t", "1", "--benchmark",
            f"--time-limit={duration}", "--scantime=3", "--cpu-affinity=0", "--no-color"]
    print(f"{phase} {round_no} | {position}/4 | {variant} | {duration}s", flush=True)
    start = datetime.now(timezone.utc).isoformat()
    before = cpu_ticks(cpu)
    with log.open("x") as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=duration + 60)
    after = cpu_ticks(cpu)
    if result.returncode:
        raise RuntimeError(f"{variant} exited with code {result.returncode}; see {log}")
    matches = METER.findall(log.read_text(errors="replace"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one SHA256D_BENCH result; see {log}")
    hashes, sec = int(matches[0][0]), float(matches[0][1])
    if not hashes or hashes % 16 or not max(1, duration-10) <= sec <= duration+30:
        raise RuntimeError(f"Invalid count/duration ({hashes}, {sec}); see {log}")
    steal = None
    if before and after and after[0] > before[0]:
        steal = 100 * (after[1]-before[1]) / (after[0]-before[0])
    rate = hashes/sec
    print(f"  {rate/1e6:.4f} MH/s", flush=True)
    return dict(phase=phase, round=round_no, position=position, variant=variant,
                started_utc=start, hashes=hashes, seconds=sec, hashes_per_second=rate,
                steal_percent=steal, log=log.name)


def summarize(samples):
    blocks = {}
    for r in samples:
        if r["phase"] == "measure":
            blocks.setdefault(r["round"], {})[r["variant"]] = r["hashes_per_second"]
    by_variant = {}
    for label in VARIANTS:
        rates = [b[label] for b in blocks.values()]
        logs = [math.log(b[label]/b["base"]) for b in blocks.values()]
        center = statistics.mean(logs)
        row = {"geometric_mean_h_s": math.exp(statistics.mean(map(math.log, rates))),
               "paired_gain_percent": 100*math.expm1(center), "ci95_percent": None}
        if len(logs) >= 4 and label != "base":
            margin = T_975[min(len(logs)-1, 30)] * statistics.stdev(logs) / math.sqrt(len(logs))
            row["ci95_percent"] = [100*math.expm1(center-margin), 100*math.expm1(center+margin)]
        by_variant[label] = row
    interval = by_variant["combined"]["ci95_percent"]
    return dict(variants=by_variant, completed_rounds=len(blocks),
                combined_lower_bound_above_1_percent=(interval[0] > 1 if interval else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minutes", type=int, default=5, help="minutes per measured run (default 5)")
    p.add_argument("--rounds", type=int, default=4, help="runs per variant (default 4)")
    p.add_argument("--smoke", action="store_true", help="one 10-second run per variant, no warmup")
    p.add_argument("--cpu", type=int, help="pinned logical CPU (default first allowed)")
    p.add_argument("--jobs", type=int, default=1, help="parallel build jobs (default 1)")
    p.add_argument("--cc", default="gcc", help="GNU C compiler")
    p.add_argument("--seed", type=int, help="seed for measurement order")
    p.add_argument("--out", type=Path, help="new results directory")
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        p.error("This native benchmark requires x86-64 Linux with AVX-512.")
    if not 1 <= args.minutes <= 120 or not 1 <= args.rounds <= 100 or args.jobs < 1:
        p.error("minutes 1..120, rounds 1..100, jobs >= 1")
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        p.error(f"CPU {cpu} is unavailable; allowed CPUs: {sorted(allowed)}")
    model = check_cpu_flags(cpu)
    now = datetime.now(timezone.utc)
    out = (args.out or REPO/"bench-results"/now.strftime("%Y%m%d-%H%M%S-%f")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = random.Random(seed)
    duration, rounds = (10, 1) if args.smoke else (60*args.minutes, args.rounds)
    plan = orders(rounds, rng)
    warmup = [] if args.smoke else rng.sample(VARIANTS, len(VARIANTS))
    meta = dict(status="building", cpu=cpu, cpu_model=model, platform=platform.platform(),
                allowed_cpus=sorted(allowed), seed=seed, order=plan, warmup=warmup,
                warmup_seconds=30, measured_seconds=duration, argv=sys.argv,
                started_utc=now.isoformat())
    save_json(out/"run.json", meta)
    print(f"Results: {out}\nCPU {cpu}: {model}\n"
          f"Planned miner time: {(4*duration*rounds + 30*len(warmup))/60:g} min + build", flush=True)
    try:
        binaries = build_variants(REPO, out/"build", args.cc, args.jobs)
        os.sched_setaffinity(0, {cpu})
        meta["status"] = "running"
        save_json(out/"run.json", meta)
        samples = []
        with (out/"samples.csv").open("x", newline="") as stream:
            writer = None
            schedule = ([("warmup", 0, warmup, 30)] if warmup else [])
            schedule += [("measure", i, row, duration) for i, row in enumerate(plan, 1)]
            for phase, r, labels, seconds in schedule:
                for pos, label in enumerate(labels, 1):
                    log = out/f"{phase}-{r:02d}-{pos}-{label}.log"
                    sample = measure(binaries[label], label, seconds, cpu, log, phase, r, pos)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=sample)
                        writer.writeheader()
                    writer.writerow(sample)
                    stream.flush()
                    samples.append(sample)
        report = summarize(samples)
        save_json(out/"summary.json", report)
        for label, row in report["variants"].items():
            ci = row["ci95_percent"]
            suffix = f"  95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]%" if ci else ""
            print(f"{label:9s} {row['geometric_mean_h_s']/1e6:9.4f} MH/s"
                  f"  {row['paired_gain_percent']:+.2f}% vs base{suffix}")
        meta["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        meta.update(status="interrupted", error=repr(error))
        raise
    finally:
        save_json(out/"run.json", meta)
    print(f"Report: {out/'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))

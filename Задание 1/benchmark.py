#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess


def configurations(suite):
    if suite == "small":
        for n in (20, 40, 80):
            yield "seq", n, n, 1, 1
        for threads in (1, 4, 16):
            yield "omp", 80, 80, 1, threads
        for processes in (1, 2, 4):
            yield "mpi", 80, 80, processes, 1
        for processes in (1, 2):
            yield "hybrid", 80, 80, processes, 4
    else:
        for m, n, counts, hybrid_processes in ((400, 600, (2, 4, 8, 16), 2),
                                               (800, 1200, (4, 8, 16, 32), 4)):
            yield "seq", m, n, 1, 1
            for count in counts:
                yield "omp", m, n, 1, count
                yield "mpi", m, n, count, 1
            for threads in (1, 2, 4, 8):
                yield "hybrid", m, n, hybrid_processes, threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("small", "appendix"), default="small")
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path, default=Path("timings.csv"))
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-flags", default="")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    build = args.build.resolve()
    baseline = {}
    writer = None
    with args.output.open("x", newline="") as stream:
        for mode, m, n, processes, threads in configurations(args.suite):
            samples = []
            for _ in range(args.repeats):
                command = [str(build / ("poisson_" + mode)), "--nx", str(m), "--ny", str(n),
                           "--tol", str(args.tol)]
                if mode in ("mpi", "hybrid"):
                    command = [args.mpiexec, *shlex.split(args.mpi_flags), "-n", str(processes), *command]
                env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
                result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                samples.append(json.loads(result.stdout))
            row = dict(samples[0])
            row["solve_seconds"] = statistics.median(s["solve_seconds"] for s in samples)
            row["setup_seconds"] = statistics.median(s["setup_seconds"] for s in samples)
            row["iterations_min"] = min(s["iterations"] for s in samples)
            row["iterations_max"] = max(s["iterations"] for s in samples)
            row["repeats"] = args.repeats
            if mode == "seq":
                baseline[m, n] = row["solve_seconds"]
            row["speedup"] = baseline[m, n] / row["solve_seconds"]
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            print(f"{mode:6s} {m}x{n} processes={processes} threads={threads}: "
                  f"{row['solve_seconds']:.6g} s, speedup={row['speedup']:.3f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-flags", default="")
    args = parser.parse_args()
    build = args.build.resolve()
    launcher = [args.mpiexec, *shlex.split(args.mpi_flags)]
    env = dict(os.environ, OMP_DYNAMIC="FALSE", OMP_NUM_THREADS="4")

    def run(mode, m, n, *extra, processes=1, output=None, expected=0):
        command = [str(build / ("poisson_" + mode)), "--nx", str(m), "--ny", str(n),
                   "--tol", "1e-10", *extra]
        if output:
            command += ["--output", str(output)]
        if mode in ("mpi", "hybrid"):
            command = [*launcher, "-n", str(processes), *command]
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
        if result.returncode != expected:
            raise AssertionError(f"{shlex.join(command)}\n{result.stdout}\n{result.stderr}")
        return json.loads(result.stdout) if result.stdout.strip() else None

    with tempfile.TemporaryDirectory(prefix="checks-", dir=build) as temp:
        temp = Path(temp)
        m, n = 6, 8
        hx, hy = 8 / m, 12 / n
        points = [(i, j) for i in range(1, m) for j in range(1, n)]
        positions = {point: k for k, point in enumerate(points)}
        matrix = [[0.0] * len(points) for _ in points]
        rhs = [0.0] * len(points)
        for k, (i, j) in enumerate(points):
            x, y = -4 + i * hx, -6 + j * hy
            slices = 20000
            width_sum = 0.0
            for s in range(slices):
                yy = y - hy / 2 + (s + .5) * hy / slices
                square = 9 - (abs(yy) - 2)**2
                if square > 0:
                    width = math.sqrt(square)
                    width_sum += max(0, min(x + hx / 2, width) - max(x - hx / 2, -width))
            fraction = width_sum / (slices * hx)
            q = (1 - fraction) / max(hx, hy)**4
            matrix[k][k] = 2 / hx**2 + 2 / hy**2 + q
            rhs[k] = fraction + q * math.sin(x + y)
            for di, dj, a in ((1, 0, 1/hx**2), (-1, 0, 1/hx**2),
                              (0, 1, 1/hy**2), (0, -1, 1/hy**2)):
                neighbor = i + di, j + dj
                if neighbor in positions:
                    matrix[k][positions[neighbor]] = -a
                else:
                    rhs[k] += a * math.sin(x + di * hx + y + dj * hy)
        size = len(points)
        for k in range(size):
            for i in range(k + 1, size):
                factor = matrix[i][k] / matrix[k][k]
                for j in range(k + 1, size):
                    matrix[i][j] -= factor * matrix[k][j]
                rhs[i] -= factor * rhs[k]
        exact = [0.0] * size
        for i in range(size - 1, -1, -1):
            exact[i] = (rhs[i] - sum(matrix[i][j] * exact[j] for j in range(i + 1, size))) / matrix[i][i]
        oracle_path = temp / "oracle.csv"
        run("seq", m, n, output=oracle_path)
        with oracle_path.open() as stream:
            rows = list(csv.DictReader(stream))
        dense_error = max(abs(float(rows[i * (n+1) + j]["u"]) - exact[k])
                          for k, (i, j) in enumerate(points))
        assert dense_error < 2e-6, dense_error
        print("Difference from independent dense solution:", dense_error)

        errors = []
        for n in (20, 40, 80):
            data = run("seq", n, n, "--manufactured")
            assert data["converged"] and data["step_l2"] < data["delta"]
            assert data["residual_l2"] < 1e-6, data
            errors.append(data["exact_max_error"])
        assert errors[-1] < .005, errors
        assert all(a / b > 2.5 for a, b in zip(errors, errors[1:])), errors
        print("Manufactured-solution max errors:", errors)
        large = run("seq", 400, 600)
        assert large["converged"] and large["restarts"] == 0, large
        assert large["residual_l2"] < 1e-5, large
        print("400x600: converged without spurious energy restarts")

        for m, n in ((23, 31), (80, 80)):
            baseline_path = temp / "baseline.csv"
            baseline = run("seq", m, n, output=baseline_path)
            with baseline_path.open() as stream:
                reference = list(csv.DictReader(stream))
            assert len(reference) == (m + 1) * (n + 1)
            for row in reference:
                x, y, u = (float(row[key]) for key in ("x", "y", "u"))
                assert math.isfinite(u)
                if abs(x) == 4 or abs(y) == 6:
                    assert abs(u - math.sin(x + y)) < 1e-14
                assert int(row["inside"]) == int(x*x + (abs(y)-2)**2 < 9)
            for mode, processes in (("omp", 1), ("mpi", 1), ("mpi", 2), ("mpi", 4),
                                    ("hybrid", 1), ("hybrid", 2)):
                output = temp / f"{mode}-{processes}.csv"
                data = run(mode, m, n, processes=processes, output=output)
                assert data["converged"] and data["processes"] == processes
                assert data["threads"] == (4 if mode in ("omp", "hybrid") else 1)
                assert abs(data["iterations"] - baseline["iterations"]) <= 1
                with output.open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == len(reference)
                assert all((a["x"], a["y"], a["inside"]) == (b["x"], b["y"], b["inside"])
                           for a, b in zip(reference, rows))
                difference = max(abs(float(a["u"]) - float(b["u"])) for a, b in zip(reference, rows))
                assert difference < 1e-8, (mode, processes, difference)
                assert data["residual_l2"] < 1e-6, data
                print(f"{m}x{n} {mode} processes={processes}: max difference {difference:.3g}")

        limited = run("seq", 20, 20, "--max-iters", "1", expected=2)
        assert not limited["converged"]
        for extra in (("--tol", "nan"), ("--nx", "3"), ("--tol", "1x"), ("--unknown",)):
            run("seq", 20, 20, *extra, expected=1)
        run("seq", 4, 100, expected=1)
    print("All numerical, parallel and input checks passed")


if __name__ == "__main__":
    main()

"""Time the two implementations over the same series, and record the environment.

    python setup.py build_ext --inplace
    python benchmark.py --out results/benchmark.json

Reports the best of N repetitions rather than the mean. The minimum is the
closest thing to the cost of the work itself: noise from scheduling and other
processes can only ever make a run slower, never faster, so the mean mostly
measures how busy the machine was.

The forcing arrays are built once, outside the timing, so what is measured is
the loop and not CSV parsing or PET.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import hbv_python as py


def best_of(fn, repeats: int) -> float:
    """Seconds for the fastest of `repeats` calls."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best = elapsed
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        import hbv_cython
    except ImportError:
        print("hbv_cython is not built; run `python setup.py build_ext --inplace`",
              file=sys.stderr)
        return 2

    precipitation, temperature, pet = py.load_forcing()
    n = len(precipitation)
    p = py.Parameters()

    # Pre-converted so the timed section is the loop, not the conversion.
    pa = np.ascontiguousarray(precipitation, dtype=np.float64)
    ta = np.ascontiguousarray(temperature, dtype=np.float64)
    ea = np.ascontiguousarray(pet, dtype=np.float64)

    python_seconds = best_of(
        lambda: py.simulate(precipitation, temperature, pet, p), args.repeats
    )
    cython_seconds = best_of(
        lambda: hbv_cython.simulate(
            pa, ta, ea, p.tt, p.cfmax, p.sfcf, p.cwh, p.cfr, p.fc, p.lp,
            p.beta, p.perc, p.uzl, p.k0, p.k1, p.k2,
        ),
        args.repeats,
    )

    # Asserted here too, so a benchmark can never report a speedup for a loop
    # that computes something different.
    expected = py.simulate(precipitation, temperature, pet, p)
    actual = hbv_cython.simulate(
        pa, ta, ea, p.tt, p.cfmax, p.sfcf, p.cwh, p.cfr, p.fc, p.lp,
        p.beta, p.perc, p.uzl, p.k0, p.k1, p.k2,
    )
    identical = all(expected[i] == float(actual[i]) for i in range(n))
    if not identical:
        print("outputs differ; refusing to report a speedup", file=sys.stderr)
        return 1

    speedup = python_seconds / cython_seconds

    record = {
        "timesteps": n,
        "repeats": args.repeats,
        "statistic": "best of N, see module docstring",
        "python_seconds": round(python_seconds, 6),
        "cython_seconds": round(cython_seconds, 6),
        "speedup": round(speedup, 2),
        "python_us_per_step": round(python_seconds / n * 1e6, 4),
        "cython_us_per_step": round(cython_seconds / n * 1e6, 4),
        "outputs_bit_identical": identical,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "system": platform.system(),
            "numpy": np.__version__,
        },
    }

    print(f"timesteps          {n}")
    print(f"pure Python        {python_seconds * 1000:.2f} ms  "
          f"({record['python_us_per_step']} us/step)")
    print(f"Cython             {cython_seconds * 1000:.2f} ms  "
          f"({record['cython_us_per_step']} us/step)")
    print(f"speedup            {speedup:.2f}x")
    print(f"bit-identical      {identical}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check the numbers in this README against results/benchmark.json.

The same idea as `../scripts/check_claims.py`, applied to this subdirectory:
the README states a speedup and two timings, those come from a result file, and
the two can drift the moment the benchmark is re-run on another machine without
the prose being updated.

    python check_claims.py

Exit code 0 if every claim matches, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
BENCHMARK = HERE / "results" / "benchmark.json"
FORCING = HERE / "data" / "dwd_01262_daily.csv"


def main() -> int:
    if not BENCHMARK.is_file():
        print(f"missing {BENCHMARK}; run benchmark.py --out first", file=sys.stderr)
        return 1

    record = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    # Whitespace collapsed so a claim is not defeated by a line wrap.
    readme = re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))

    claims: list[tuple[str, str]] = [
        ("timestep count", rf"\b{record['timesteps']}\b"),
        ("speedup", rf"{record['speedup']:.2f}x"),
        ("python milliseconds", rf"{record['python_seconds'] * 1000:.2f} ms"),
        ("cython milliseconds", rf"{record['cython_seconds'] * 1000:.2f} ms"),
        ("python us per step", rf"{record['python_us_per_step']}"),
        ("cython us per step", rf"{record['cython_us_per_step']}"),
        ("python version", re.escape(record["environment"]["python"])),
        ("numpy version", re.escape(record["environment"]["numpy"])),
    ]

    failures = [(d, p) for d, p in claims if not re.search(p, readme)]

    # The parity claim is not a regex match on prose: it is read from the record,
    # because a README saying "bit-identical" while the benchmark recorded
    # otherwise is the one failure that matters most.
    if not record.get("outputs_bit_identical"):
        failures.append(("bit-identical", "benchmark recorded outputs_bit_identical=false"))

    digest = hashlib.sha256(FORCING.read_bytes()).hexdigest() if FORCING.is_file() else None
    if digest is None:
        failures.append(("forcing data present", str(FORCING)))
    elif digest not in readme:
        failures.append(("forcing sha256", digest))

    width = max(len(d) for d, _ in claims + [("forcing sha256", "")])
    for desc, pattern in claims:
        ok = (desc, pattern) not in failures
        print(f"  {'ok  ' if ok else 'FAIL'}  {desc:<{width}}")
    print(f"  {'ok  ' if digest and digest in readme else 'FAIL'}  {'forcing sha256':<{width}}")
    print(f"  {'ok  ' if record.get('outputs_bit_identical') else 'FAIL'}  "
          f"{'bit-identical':<{width}}")

    if failures:
        print(f"\n{len(failures)} claim(s) do not match README.md:", file=sys.stderr)
        for desc, pattern in failures:
            print(f"  {desc}: no match for /{pattern}/", file=sys.stderr)
        return 1

    print(f"\nAll {len(claims) + 2} claims match results/benchmark.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

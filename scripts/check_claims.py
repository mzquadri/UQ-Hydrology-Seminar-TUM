"""Check the headline numbers in README.md against the files they come from.

The README states results. Those results live in results/, and the two can drift:
a number gets rounded differently, a figure is regenerated from a later run, or a
value is carried over from a draft that no longer matches the run. This script
recomputes each headline claim from the result files and fails if the README no
longer contains it.

It checks values, not prose. A claim that is not listed here is not checked.

    python scripts/check_claims.py

Exit code 0 if every claim matches, 1 otherwise.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"

A1 = RESULTS / "assignment1_finial_gen600_atol-3"
A3 = RESULTS / "assignment3"
A4 = RESULTS / "assignment4_gen600"
A5 = RESULTS / "assignment5"
ROPE = RESULTS / "exercise3_rope"


def read_text(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing: {path.relative_to(ROOT).as_posix()}")
    return path.read_text(encoding="utf-8")


def kv(path: Path) -> dict[str, str]:
    out = {}
    for line in read_text(path).splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def labelled(path: Path, label: str) -> float:
    match = re.search(rf"^\s*{re.escape(label)}:\s*([-\d.eE+]+)", read_text(path), re.M)
    if not match:
        raise SystemExit(f"{label!r} not found in {path.name}")
    return float(match.group(1))


def column(path: Path, name: str) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [float(row[name]) for row in csv.DictReader(handle)]


def sobol(folder: str) -> dict[str, float]:
    text = read_text(A3 / folder / "sobol_summary_corrected.txt")
    grab = lambda pat: float(re.search(pat, text).group(1))  # noqa: E731
    with (A3 / folder / "sobol_indices_corrected.csv").open(newline="",
                                                            encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    top = max(rows, key=lambda r: float(r["ST"]))
    return {
        "variance": grab(r"Total variance V\(Y\):\s*([\d.eE+-]+)"),
        "sum_st": grab(r"Sum of ST:\s*([\d.eE+-]+)"),
        "sum_s1": grab(r"Sum of S1:\s*([\d.eE+-]+)"),
        "top_name": top["Parameter"],
        "top_st": float(top["ST"]),
    }


def csv_rows(path: Path) -> list[dict]:
    with need_csv(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def need_csv(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing: {path.relative_to(ROOT).as_posix()}")
    return path


def bounds_violations(folder: str, tolerance: float = 1e-6) -> int:
    """Parameters whose first-order index exceeds their total-order index.

    S1 <= ST holds for any model, so a positive count is a property of the
    estimate rather than of the hydrology.
    """
    rows = csv_rows(A3 / folder / "sobol_indices_corrected.csv")
    return sum(1 for r in rows if float(r["S1"]) > float(r["ST"]) + tolerance)


def build_claims() -> list[tuple[str, str]]:
    """Every claim as (description, a regex the README must match).

    The pattern carries context, not just the number. Checking for a bare "25"
    would pass on any stray 25 anywhere in the file, which is not a check at all.
    Patterns are matched against a whitespace-normalised copy of the README, so
    they are unaffected by where lines happen to wrap.
    """
    claims: list[tuple[str, str]] = []

    def add(desc: str, pattern: str) -> None:
        claims.append((desc, pattern))

    # --- Assignment 1
    final_nse = labelled(A1 / "final_debug_summary.txt", "final_nse")
    best_obj = labelled(A1 / "final_debug_summary.txt", "best_obj")
    evals = int(labelled(A1 / "final_debug_summary.txt", "n_evals"))
    gens = len(column(A1 / "optimization_gen_summary.csv", "generation"))
    add("best NSE", rf"an NSE of \*\*{final_nse:.4f}\*\*")
    add("best objective", rf"best objective reached is {best_obj:.4f}")
    add("generations run", rf"run for {gens} generations")
    add("model evaluations", rf"{evals:,} model evaluations")

    turn = kv(A1 / "turned_off_processes" / "NSE_values_tunroff.txt")
    add("baseline in turn-off table",
        rf"All processes on \| {float(turn['NSE without change']):.4f}")
    add("groundwater off", rf"Groundwater off \| {float(turn['NSE_GW']):.4f}")
    add("upper reservoir off", rf"Upper reservoir off \| {float(turn['NSE_urr']):.4f}")
    add("snow off", rf"Snow off \| {float(turn['NSE_snw']):.4f}")
    add("lower reservoir off",
        rf"Lower reservoir off \| {float(turn['NSE_lrr']):.4f}")

    # --- Assignment 3
    full_nse = sobol("Assignment3_Results_full_range_nse")
    narrow_nse = sobol("Assignment3_narrow_NSE")
    narrow_log = sobol("Assignment3_narrow_logNSE")
    full_log = sobol("Assignment3_Results_full_range_lognse_problematic")
    add("full-range NSE row",
        rf"Full range, NSE \| {full_nse['variance']:.3f} \| `{full_nse['top_name']}` "
        rf"\({full_nse['top_st']:.2f}\) \| {full_nse['sum_st']:.3f}")
    add("narrow NSE row",
        rf"Narrow range, NSE \| {narrow_nse['variance']:.5f} \| "
        rf"`{narrow_nse['top_name']}` \({narrow_nse['top_st']:.2f}\) \| "
        rf"{narrow_nse['sum_st']:.3f}")
    add("narrow logNSE row",
        rf"Narrow range, logNSE \| {narrow_log['variance']:.3f} \| "
        rf"`{narrow_log['top_name']}` \({narrow_log['top_st']:.2f}\) \| "
        rf"{narrow_log['sum_st']:.3f}")
    add("broken logNSE sum of S1", rf"they sum to {full_log['sum_s1']:.2f}")

    # A total-order Sobol index is bounded below by the first-order index for the
    # same parameter. The estimator here is the group's own rather than a
    # library's, so this is worth testing directly: it is a second symptom of the
    # breakdown in the discarded configuration, independent of the sum of S1 the
    # README reads it from, and it is a check on the three that are used.
    sound = sum(bounds_violations(f) for f in
                ("Assignment3_Results_full_range_nse", "Assignment3_narrow_NSE",
                 "Assignment3_narrow_logNSE"))
    broken = bounds_violations("Assignment3_Results_full_range_lognse_problematic")
    total = len(list((A3 / "Assignment3_narrow_NSE").glob("*")) and
                csv_rows(A3 / "Assignment3_narrow_NSE" / "sobol_indices_corrected.csv"))
    add("sound configurations respect ST >= S1",
        rf"it fails for \*\*{sound} of {total}\*\* parameters")
    add("the discarded configuration does not",
        rf"it fails for \*\*{broken} of {total}\*\*")
    ratio = full_nse["variance"] / narrow_nse["variance"]
    add("full/narrow variance ratio", rf"factor of about \*\*{ratio:.0f}\*\*")

    # --- Assignment 4
    a4_summary = A4 / "uncertainty_analysis_summary.txt"
    ref_nse = 1.0 - labelled(a4_summary, "Reference OFV")
    a4_ref = 1.0 - labelled(a4_summary, "OFV mean")
    text4 = read_text(a4_summary)
    mean, std = re.search(r"Perturbation: C = N\(([\d.]+), ([\d.]+)\)", text4).groups()
    lo, hi = re.search(r"clipping \[([\d.]+), ([\d.]+)\]", text4).groups()
    marc = re.search(r"Mean Absolute Relative PPT Change:\s*Mean:\s*([\d.]+)%",
                     text4).group(1)
    better = re.search(r"With reference parameters: (\d+)/(\d+)", text4).groups()
    add("precipitation multiplier", rf"N\({mean}, {std}\)")
    add("multiplier clipping", rf"\[{lo}, {hi}\]")
    add("mean absolute precipitation change",
        rf"mean absolute change in precipitation is {marc}%")
    add("precipitation-error mean NSE",
        rf"reference parameters is \*\*{a4_ref:.4f}\*\*")
    add("calibrated baseline NSE", rf"baseline of {ref_nse:.4f}")
    add("series better by chance", rf"{int(better[0])} of the {int(better[1]):,} noisy")

    # --- Assignment 5
    a5_summary = A5 / "uncertainty_analysis_summary.txt"
    a5_ref = 1.0 - labelled(a5_summary, "OFV mean")
    text5 = read_text(a5_summary)
    lo5, hi5 = re.search(r"Perturbation Interval for water level: \[-(\d+), (\d+)\]",
                         text5).groups()
    recovered = re.search(r"Compensation fraction: ([\d.]+)% of input-induced loss",
                          text5).group(1)
    calc = re.search(r"calculated NSE:\s*([\d.]+)", text5).group(1)
    fit = read_text(A5 / "results_fiting_curve.txt")
    r2_two = float(re.search(r"Final R.\s*=\s*([\d.]+)", fit).group(1))
    r2_one = float(re.search(r"R.\s*=\s*([\d.]+)\s*RMSE", fit).group(1))
    add("water level perturbation", rf"\[-{lo5}, \+{hi5}\] cm")

    # The two perturbations are not the same size, and the asymmetry the seminar
    # concludes with is only readable once that is said. Both changes are
    # recorded by the runs; the ratio below divides each loss by the perturbation
    # that produced it, so the comparison does not rest on them being equal.
    ppt_change = float(re.search(
        r"Mean Absolute Relative PPT Change:\s*Mean:\s*([\d.]+)%", text4).group(1))
    dis_change = float(re.search(
        r"Mean Absolute Relative DIS Change:\s*Mean:\s*([\d.]+)%", text5).group(1))
    ref_ofv = labelled(a4_summary, "Reference OFV")
    loss4 = labelled(a4_summary, "OFV mean") - ref_ofv
    loss5 = labelled(a5_summary, "OFV mean") - ref_ofv
    add("mean absolute discharge change",
        rf"discharge series by \*\*{dis_change}%\*\*")
    add("relative size of the two perturbations",
        rf"perturbed {dis_change / ppt_change:.2f} times harder")
    ratio = dis_change / ppt_change
    add("raw ratio of the two losses", rf"\*\*{loss5 / loss4:.0f}\*\* times larger")
    add("loss per unit of perturbation",
        rf"about \*\*{(loss5 / dis_change) / (loss4 / ppt_change):.0f}\*\* times as much")
    add("loss per squared unit of perturbation",
        rf"still \*\*{(loss5 / loss4) / ratio ** 2:.0f}\*\*")
    add("rating curve R squared", rf"R squared \*\*{r2_two:.4f}\*\*")
    add("single power law R squared", rf"against {r2_one:.4f} for a")
    add("rating-curve-error mean NSE", rf"Mean NSE falls to \*\*{a5_ref:.4f}\*\*")
    add("loss recovered by recalibration", rf"recovers {recovered}% of the loss")
    add("rating curve reconstruction NSE", rf"\"calculated NSE\" of {float(calc):.4f}")

    # --- What the perturbation code actually does
    #
    # The README used to say the multipliers were clipped to [0.75, 1.25]. They
    # are not: the clip is commented out in the script as submitted, and the
    # recorded mean absolute change matches an unclipped draw. This pins both
    # halves, so re-enabling the clip without correcting the README fails here.
    perturb = read_text(ROOT / "code" / "Ass_04_Input_Uncertainty_Group_B.py")
    clip_line = re.search(r"^(\s*)(#?)\s*multipliers = np\.clip\(multipliers",
                          perturb, re.M)
    if clip_line is None:
        raise SystemExit("the multiplier clip line is gone from Assignment 4's script")
    if clip_line.group(2) != "#":
        raise SystemExit(
            "Assignment 4's script now clips the multipliers, which the recorded "
            "run did not. Update README.md and this check together.")
    add("multipliers were not clipped", r"carries that clip commented out")

    # --- Exercise 3, only once the summaries have been extracted
    if (ROPE / "hbv_daily_1971_1980_cb" / "420_rope_selected_threshold.txt").is_file():
        first = kv(ROPE / "hbv_daily_1971_1980_cb" / "420_rope_selected_threshold.txt")
        second = kv(ROPE / "hbv_daily_1981_1990_cb" / "420_rope_selected_threshold.txt")
        stop = float(first["outside_ratio_stop"])
        add("ROPE stop criterion", rf"outside ratio is still under {stop:.0%}")
        add("ROPE 1971 to 1980",
            rf"selects \*\*{float(first['obj_thresh_ulim_selected']):g}\*\* for 1971 "
            rf"to 1980, keeping \*\*{first['n_rope']}\*\*")
        add("ROPE 1981 to 1990",
            rf"settles at \*\*{float(second['obj_thresh_ulim_selected']):g}\*\* and "
            rf"keeps \*\*{second['n_rope']}\*\*")

    return claims


def main() -> int:
    # Collapse whitespace so a pattern is not defeated by a line wrap.
    readme = re.sub(r"\s+", " ", read_text(README))

    claims = build_claims()
    failures = []
    for desc, pattern in claims:
        if not re.search(pattern, readme):
            failures.append((desc, pattern))

    width = max(len(d) for d, _ in claims)
    for desc, pattern in claims:
        ok = (desc, pattern) not in failures
        print(f"  {'ok  ' if ok else 'FAIL'}  {desc:<{width}}")

    if failures:
        print(f"\n{len(failures)} of {len(claims)} claims do not match README.md:",
              file=sys.stderr)
        for desc, pattern in failures:
            print(f"  {desc}: no match for /{pattern}/", file=sys.stderr)
        return 1

    print(f"\nAll {len(claims)} headline claims match the result files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

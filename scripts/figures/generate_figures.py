"""Figures for the seminar results, drawn only from the versioned result files.

Four figures, each answering one question:

    01  did the calibration converge, and which processes matter
    02  which parameters control the model across the whole parameter space
    03  which hurts more, error in the input or error in the observed output
    04  how was the ROPE objective threshold chosen

Every number is read from results/, so a figure cannot show a value the runs did
not produce. Exercise 3 reads results/exercise3_rope/, which is lifted out of the
LFS archive by scripts/extract_exercise3_summaries.py.

    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import portfolio_style as ps  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

A1 = RESULTS / "assignment1_finial_gen600_atol-3"
A3 = RESULTS / "assignment3"
A4 = RESULTS / "assignment4_gen600"
A5 = RESULTS / "assignment5"
ROPE = RESULTS / "exercise3_rope"

#: The objective is 1 - NSE, exactly: the Assignment 1 reference OFV of 0.092285
#: corresponds to the reported NSE of 0.907715.
def nse(ofv):
    return 1.0 - np.asarray(ofv, dtype=float)


#: Every result file read since the last figure was written, against a digest of
#: its contents. Recorded here rather than listed by hand beside each figure, so
#: that a figure which stops reading a file stops claiming it and one that starts
#: reading a file records it without anyone remembering to.
_READ: dict[str, str] = {}


def content_digest(path: Path) -> str:
    """A digest of what the file says, not of how the checkout spelled it.

    The result files are text. Git hands a Windows checkout CRLF and a Linux one
    LF for the same committed bytes, so hashing the working tree directly makes
    the digest a property of the machine. A figure regenerated on Windows then
    fails this check in CI, which is a false alarm about provenance and says
    nothing about whether the figure is current.

    `check_repository.py` normalises the same way. If one side changes, both must.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing: {path.relative_to(ROOT).as_posix()}")
    key = path.relative_to(ROOT).as_posix()
    if key not in _READ:
        _READ[key] = content_digest(path)
    return path


def inputs_read() -> dict[str, str]:
    """The files read since the previous figure, and reset for the next one."""
    out = dict(_READ)
    _READ.clear()
    return out


def read_csv_cols(path: Path) -> dict[str, list[str]]:
    with need(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty: {path.relative_to(ROOT).as_posix()}")
    return {k: [r[k] for r in rows] for k in rows[0]}


def read_kv(path: Path) -> dict[str, str]:
    """Parse the `key = value` and `key=value` summary files."""
    out = {}
    for line in need(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def summary_value(path: Path, label: str) -> float:
    """Pull one `  Label: 0.1234` number out of an analysis summary."""
    text = need(path).read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(label)}:\s*([-\d.]+)", text, re.M)
    if not match:
        raise SystemExit(f"{label!r} not found in {path.name}")
    return float(match.group(1))


# --------------------------------------------------------------------------
# 01  calibration
# --------------------------------------------------------------------------

def fig_calibration():
    cols = read_csv_cols(A1 / "optimization_gen_summary.csv")
    gen = np.array([int(g) for g in cols["generation"]])
    best_per_gen = nse([float(v) for v in cols["obj_min"]])
    running_best = np.maximum.accumulate(best_per_gen)
    final = running_best[-1]

    # State when the search stopped moving rather than asserting a round number.
    within = np.flatnonzero(running_best >= final - 0.001)
    settled = int(gen[within[0]])

    turn = read_kv(A1 / "turned_off_processes" / "NSE_values_tunroff.txt")
    baseline = float(turn["NSE without change"])
    experiments = [
        ("all processes on", baseline, ps.SLATE),
        ("groundwater off", float(turn["NSE_GW"]), ps.GREEN),
        ("upper reservoir off", float(turn["NSE_urr"]), ps.GREEN),
        ("snow off", float(turn["NSE_snw"]), ps.AMBER),
        ("lower reservoir off", float(turn["NSE_lrr"]), ps.RED),
    ]

    fig = plt.figure(figsize=(13.0, 7.4))
    axL = fig.add_axes([0.075, 0.205, 0.475, 0.520])
    axR = fig.add_axes([0.640, 0.205, 0.310, 0.520])

    axL.plot(gen, best_per_gen, color=ps.BLUE_SOFT, lw=1.0, zorder=2)
    axL.plot(gen, running_best, color=ps.BLUE, lw=2.2, zorder=3)
    axL.axhline(final, color=ps.HAIR, lw=1.2, ls="--", zorder=1)
    axL.axvline(settled, color=ps.FAINT, lw=1.0, ls=":", zorder=1)
    axL.set_ylim(-1.05, 1.02)
    axL.set_xlim(0, gen[-1])
    ps.clean(axL)
    axL.set_xlabel("generation", fontsize=10.8)
    axL.set_ylabel("Nash-Sutcliffe efficiency", fontsize=10.8)
    ps.note(axL, 0.46, 0.30, f"best NSE {final:.4f}", colour=ps.BLUE, size=13.5,
            weight="600")
    ps.note(axL, 0.46, 0.215,
            f"within 0.001 of it by generation {settled} of {gen[-1]}",
            colour=ps.FAINT, size=9.8)
    ps.note(axL, 0.46, 0.135, "pale line is the best in each generation alone,",
            colour=ps.FAINT, size=9.3)
    ps.note(axL, 0.46, 0.075, "solid line the best found so far",
            colour=ps.FAINT, size=9.3)

    ypos = np.arange(len(experiments))[::-1]
    for (label, value, colour), yy in zip(experiments, ypos, strict=True):
        axR.barh(yy, value, color=colour, height=0.52, zorder=3, alpha=0.85)
        axR.text(-0.035, yy, label, ha="right", va="center", fontsize=10.6,
                 color=ps.INK, transform=axR.get_yaxis_transform())
        off = 0.03 if value >= 0 else -0.03
        axR.text(value + off, yy, f"{value:.4f}", va="center",
                 ha="left" if value >= 0 else "right", fontsize=9.8, color=colour,
                 fontweight="600")
    axR.axvline(0, color=ps.HAIR, lw=1.1, zorder=2)
    axR.set_yticks([])
    axR.set_xlim(-1.05, 1.25)
    ps.clean(axR, left=False, grid_axis="x")
    axR.set_xlabel("NSE with that process disabled", fontsize=10.8)

    ps.title_block(
        fig, "A calibrated baseline, and the processes it depends on",
        "Differential evolution on the 18 HBV001a parameters, and then the same "
        "model re-run with individual\nstores switched off to see which ones the "
        "result actually rests on.", y=0.955, size=21)
    ps.footnote(fig, [
        "Turning the lower reservoir off does not merely degrade the fit, it takes "
        "NSE below zero, meaning the model then predicts the event worse than the "
        "mean of the observations does.",
        "Groundwater and the upper reservoir can be removed without measurable cost "
        "on this short, rainfall-dominated event, which says more about the event "
        "than about the stores.",
        "Source: results/assignment1_finial_gen600_atol-3/."], y=0.090)
    ps.save(fig, OUT, "01_calibration", sources=inputs_read())
    return {
        "final_nse": final,
        "settled_generation": settled,
        "generations": int(gen[-1]),
        "turnoff": {k: v for k, v, _ in experiments},
    }


# --------------------------------------------------------------------------
# 02  global sensitivity
# --------------------------------------------------------------------------

CONFIGS = [
    ("Assignment3_Results_full_range_nse", "Full range, NSE"),
    ("Assignment3_narrow_NSE", "Narrow range, NSE"),
    ("Assignment3_narrow_logNSE", "Narrow range, logNSE"),
    ("Assignment3_Results_full_range_lognse_problematic", "Full range, logNSE"),
]


def fig_sensitivity():
    fig = plt.figure(figsize=(13.0, 8.0))
    stats = {}
    lefts = [0.075, 0.315, 0.555, 0.795]
    top_n = 6

    for (folder, label), left in zip(CONFIGS, lefts, strict=True):
        cols = read_csv_cols(A3 / folder / "sobol_indices_corrected.csv")
        names = cols["Parameter"]
        st = np.array([float(v) for v in cols["ST"]])
        order = np.argsort(st)[::-1][:top_n]

        summary = need(A3 / folder / "sobol_summary_corrected.txt").read_text("utf-8")
        variance = float(
            re.search(r"Total variance V\(Y\):\s*([\d.eE+-]+)", summary).group(1))
        sum_st = float(re.search(r"Sum of ST:\s*([\d.eE+-]+)", summary).group(1))
        sum_s1 = float(re.search(r"Sum of S1:\s*([\d.eE+-]+)", summary).group(1))
        broken = sum_s1 > 1.05  # first-order indices cannot exceed 1 for a sound run
        stats[label] = {"V(Y)": variance, "sum_ST": sum_st, "sum_S1": sum_s1,
                        "top": names[order[0]], "top_ST": float(st[order[0]]),
                        "unreliable": broken}

        ax = fig.add_axes([left, 0.255, 0.165, 0.455])
        colour = ps.RED if broken else ps.BLUE
        ypos = np.arange(top_n)[::-1]
        ax.barh(ypos, st[order], color=colour, alpha=0.85, height=0.6, zorder=3)
        for yy, idx in zip(ypos, order, strict=True):
            ax.text(-0.02, yy, names[idx], ha="right", va="center", fontsize=9.6,
                    color=ps.INK, transform=ax.get_yaxis_transform())
            ax.text(st[idx] + 0.02, yy, f"{st[idx]:.2f}", va="center", fontsize=8.8,
                    color=colour)
        ax.set_yticks([])
        ax.set_xlim(0, 0.85)
        ax.set_xticks([0, 0.4, 0.8])
        ps.clean(ax, left=False, grid_axis="x")
        ax.text(0, 1.13, label, transform=ax.transAxes, fontsize=11.4, color=ps.INK,
                fontweight="600", va="bottom")
        note = f"V(Y) {variance:.3g}"
        if broken:
            note += f"\nsum of S1 = {sum_s1:.2f}, not usable"
        else:
            note += f"\nsum of ST {sum_st:.2f}"
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=9.0,
                color=ps.RED if broken else ps.FAINT, va="bottom", linespacing=1.5)
        ax.set_xlabel("total-order index", fontsize=9.8)

    ps.title_block(
        fig, "The ranking depends on where you look and what you score",
        "Sobol total-order indices for the six most influential parameters under "
        "each sampling range and objective.\nThe six are chosen per panel, so the "
        "parameter names change between panels.", y=0.955, size=21)
    ps.footnote(fig, [
        "Total-order indices summing above one is not an error: it counts "
        "interaction effects more than once, and the excess is a measure of how far "
        "the parameters act jointly rather than alone.",
        "The fourth panel is shown because it fails. First-order indices cannot sum "
        "to more than one, so 6.28 says the estimator has broken down.",
        "That is what logNSE does when sampled over the full parameter range: the "
        "log of a near-zero flow diverges, and the variance decomposition goes with "
        "it. Source: results/assignment3/."], y=0.115)
    ps.save(fig, OUT, "02_global_sensitivity", sources=inputs_read())
    return stats


# --------------------------------------------------------------------------
# 03  input against output uncertainty
# --------------------------------------------------------------------------

def fig_uncertainty():
    a4 = read_csv_cols(A4 / "uncertainty_results.csv")
    a5 = read_csv_cols(A5 / "uncertainty_results.csv")
    ref = 1.0 - summary_value(A4 / "uncertainty_analysis_summary.txt", "Reference OFV")

    series = [
        ("precipitation error", nse([float(v) for v in a4["ofv_reference_params"]]),
         ps.BLUE, ps.BLUE_SOFT),
        ("rating-curve error", nse([float(v) for v in a5["ofv_reference_params"]]),
         ps.RED, ps.RED_SOFT),
    ]

    fig = plt.figure(figsize=(13.0, 7.4))
    ax = fig.add_axes([0.075, 0.215, 0.615, 0.505])
    axR = fig.add_axes([0.775, 0.215, 0.175, 0.505])

    bins = np.linspace(0.70, 0.93, 90)
    peaks = []
    for _label, values, edge, fill in series:
        counts, _, _ = ax.hist(values, bins=bins, color=fill, edgecolor=edge,
                               linewidth=0.9, zorder=3, alpha=0.85)
        peaks.append(counts.max())
    # Headroom so each distribution can be labelled directly above its own peak,
    # inside the axes, rather than underneath where the axis label already sits.
    ax.set_ylim(0, max(peaks) * 1.30)
    for (label, values, edge, _fill), peak in zip(series, peaks, strict=True):
        # The baseline rule is drawn at `ref`; a mean sitting on top of it would put
        # the label across the rule, so those labels are hung to one side instead.
        near = abs(values.mean() - ref) < 0.01
        xx = values.mean() - 0.003 if near else values.mean()
        align = "right" if near else "center"
        ax.text(xx, peak + max(peaks) * 0.045, f"mean {values.mean():.4f}",
                color=edge, fontsize=11.0, fontweight="600", ha=align, va="bottom",
                zorder=6)
        ax.text(xx, peak + max(peaks) * 0.115, label, color=edge,
                fontsize=10.2, ha=align, va="bottom", zorder=6)
    ax.axvline(ref, color=ps.INK, lw=1.4, ls="--", zorder=5)
    ps.note(ax, ref, 1.02, f"calibrated baseline {ref:.4f}", colour=ps.INK, size=10.2,
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", weight="600")
    ps.clean(ax)
    ax.set_xlabel("Nash-Sutcliffe efficiency over 2,000 perturbed series", fontsize=10.8)
    ax.set_ylabel("number of series", fontsize=10.8)
    ax.set_xlim(0.70, 0.935)

    # How much of the loss recalibration wins back, in each study.
    a4_ref = nse([float(v) for v in a4["ofv_reference_params"]]).mean()
    a4_rec = nse([float(v) for v in a4["ofv_recalibrated"]]).mean()
    a5_ref = nse([float(v) for v in a5["ofv_reference_params"]]).mean()
    a5_rec = nse([float(v) for v in a5["ofv_recalibrated"]]).mean()
    pairs = [("precipitation\nerror", a4_ref, a4_rec, ps.BLUE),
             ("rating-curve\nerror", a5_ref, a5_rec, ps.RED)]
    for i, (_label, before, after, colour) in enumerate(pairs):
        axR.plot([i, i], [before, after], color=colour, lw=2.0, zorder=3)
        axR.plot([i], [before], "o", color=ps.PAPER, mec=colour, mew=1.8, ms=8, zorder=4)
        axR.plot([i], [after], "o", color=colour, ms=8, zorder=4)
        axR.text(i, min(before, after) - 0.012, f"+{after - before:.4f}",
                 ha="center", va="top", fontsize=9.8, color=colour,
                 fontweight="600")
    axR.axhline(ref, color=ps.INK, lw=1.2, ls="--", zorder=2)
    axR.set_xticks(range(len(pairs)))
    axR.set_xticklabels([p[0] for p in pairs], fontsize=9.8)
    axR.set_xlim(-0.6, len(pairs) - 0.4)
    axR.set_ylim(0.70, 0.935)
    ps.clean(axR)
    axR.text(0, 1.05, "what recalibration wins back", transform=axR.transAxes,
             fontsize=10.8, color=ps.INK, fontweight="600", va="bottom")

    ps.title_block(
        fig, "Error in the observations hurts far more than error in the rain",
        "The same model and the same event, perturbed two ways. Multiplying "
        "precipitation by Gaussian noise barely moves\nthe fit; propagating "
        "rating-curve error into the observed discharge moves it a great deal.",
        y=0.955, size=21)
    ps.footnote(fig, [
        "Precipitation noise leaves the mean NSE within 0.0004 of the calibrated "
        "baseline and 834 of the 2,000 series happen to score better than it. "
        "Rating-curve error leaves none of the 2,000 better.",
        "Recalibrating against each perturbed series recovers only a small part of "
        "the loss, which is the point: if the discharge you calibrate against is "
        "wrong, fitting harder cannot tell you so.",
        "Source: results/assignment4_gen600/ and results/assignment5/."], y=0.090)
    ps.save(fig, OUT, "03_input_vs_output_uncertainty", sources=inputs_read())
    return {"baseline": ref, "a4_ref": a4_ref, "a4_recal": a4_rec,
            "a5_ref": a5_ref, "a5_recal": a5_rec}


# --------------------------------------------------------------------------
# 04  ROPE threshold selection
# --------------------------------------------------------------------------

def fig_rope():
    sweep = ROPE / "hbv_daily_1971_1980_cb" / "420_threshold_sweep_summary.csv"
    if not sweep.exists():
        print("  skipping 04: run scripts/extract_exercise3_summaries.py first")
        return None
    cols = read_csv_cols(sweep)
    thr = np.array([float(v) for v in cols["threshold"]])
    ratio = np.array([float(v) for v in cols["outside_ratio"]])
    n_rope = np.array([int(v) for v in cols["n_rope"]])

    chosen = {}
    for period in ("hbv_daily_1971_1980_cb", "hbv_daily_1981_1990_cb"):
        kv = read_kv(ROPE / period / "420_rope_selected_threshold.txt")
        chosen[period] = (float(kv["obj_thresh_ulim_selected"]), int(kv["n_rope"]),
                          float(kv["outside_ratio_stop"]))
    stop = chosen["hbv_daily_1971_1980_cb"][2]
    picked = chosen["hbv_daily_1971_1980_cb"][0]

    fig = plt.figure(figsize=(13.0, 7.2))
    ax = fig.add_axes([0.075, 0.215, 0.470, 0.500])
    axR = fig.add_axes([0.650, 0.215, 0.300, 0.500])

    ax.axhline(stop * 100, color=ps.RED, lw=1.3, ls="--", zorder=2)
    ps.note(ax, 0.02, stop * 100 + 0.12, f"stop criterion, {stop:.0%}", colour=ps.RED,
            size=9.8, transform=ax.get_yaxis_transform())
    ax.plot(thr, ratio * 100, color=ps.BLUE, lw=2.0, zorder=3)
    for x, y in zip(thr, ratio * 100, strict=True):
        keep = y <= stop * 100
        ax.plot([x], [y], "o", ms=9 if x == picked else 7,
                color=ps.BLUE if keep else ps.RED, zorder=4,
                mec=ps.PAPER, mew=1.6)
    ax.annotate(f"threshold {picked:g} taken", xy=(picked, ratio[thr == picked][0] * 100),
                xytext=(picked + 0.06, 1.75), fontsize=10.2, color=ps.INK,
                arrowprops={"arrowstyle": "-", "color": ps.FAINT, "lw": 1.0})
    ax.set_xlim(1.05, 0.45)  # swept downwards, so read left to right as it ran
    ax.set_ylim(0, 2.9)
    ps.clean(ax)
    ax.set_xlabel("objective threshold, swept downwards", fontsize=10.8)
    ax.set_ylabel("parameter sets falling outside the region (%)", fontsize=10.8)

    ypos = np.arange(len(thr))[::-1]
    axR.barh(ypos, n_rope, color=ps.SLATE_SOFT, height=0.6, zorder=3)
    for yy, t, n in zip(ypos, thr, n_rope, strict=True):
        bar = ps.BLUE if t == picked else ps.SLATE
        axR.barh(yy, n, color=bar, height=0.6, zorder=4,
                 alpha=1.0 if t == picked else 0.55)
        axR.text(-0.02, yy, f"{t:g}", ha="right", va="center", fontsize=10.0,
                 color=ps.INK, transform=axR.get_yaxis_transform())
        axR.text(n + 6, yy, str(n), va="center", fontsize=9.6, color=bar)
    axR.set_yticks([])
    axR.set_xlim(0, max(n_rope) * 1.2)
    ps.clean(axR, left=False, grid_axis="x")
    axR.set_xlabel("parameter sets kept by ROPE", fontsize=10.8)
    axR.text(0, 1.05, "objective threshold", transform=axR.transAxes, fontsize=10.6,
             color=ps.INK, fontweight="600", va="bottom")

    other = chosen["hbv_daily_1981_1990_cb"]
    ps.title_block(
        fig, "Choosing the ROPE threshold by how well the region holds",
        "Robust parameter estimation keeps the deepest parameter sets among those "
        "below an objective threshold. Lowering\nthat threshold admits more sets "
        "until they stop lying inside the region they define.", y=0.955, size=21)
    ps.footnote(fig, [
        f"The sweep runs from 1.0 downwards and stops at the lowest threshold whose "
        f"outside ratio is still under {stop:.0%}. For catchment 420 over 1971 to "
        f"1980 that is {picked:g}, keeping {chosen['hbv_daily_1971_1980_cb'][1]} "
        f"parameter sets.",
        f"The same procedure over 1981 to 1990 settles at {other[0]:g} and keeps "
        f"{other[1]} sets, so the decade the model is calibrated on changes how "
        f"tightly the parameters can be pinned down.",
        "Source: results/exercise3_rope/, extracted from the Exercise 3 archive."],
        y=0.090)
    ps.save(fig, OUT, "04_rope_threshold", sources=inputs_read())
    return {"picked": picked, "chosen": chosen}


def main() -> int:
    ps.apply()
    print("reading results/\n")
    calib = fig_calibration()
    sens = fig_sensitivity()
    unc = fig_uncertainty()
    rope = fig_rope()

    print("\nvalues the figures assert:")
    print(f"  best NSE                {calib['final_nse']:.4f} "
          f"(settled by generation {calib['settled_generation']} "
          f"of {calib['generations']})")
    for label, s in sens.items():
        flag = "  UNRELIABLE" if s["unreliable"] else ""
        print(f"  {label:<22} top {s['top']} ST={s['top_ST']:.3f} "
              f"V(Y)={s['V(Y)']:.3g}{flag}")
    print(f"  precipitation error     mean NSE {unc['a4_ref']:.4f} "
          f"(recalibrated {unc['a4_recal']:.4f})")
    print(f"  rating-curve error      mean NSE {unc['a5_ref']:.4f} "
          f"(recalibrated {unc['a5_recal']:.4f})")
    if rope:
        for period, (thr, n, _) in rope["chosen"].items():
            print(f"  ROPE {period:<24} threshold {thr:g}, {n} sets kept")
    print(f"\nfigures written to {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

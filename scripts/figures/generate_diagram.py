"""The seminar workflow diagram, with its numbers read from results/.

The previous version of this diagram was written by hand and drifted from the runs
it described: it stated a 5% precipitation noise level where the multiplier standard
deviation is 0.083, and a 15 cm water level perturbation where the runs used 25 cm.
Generating it from the result files removes the opportunity to drift again.

    python scripts/figures/generate_diagram.py

Output: docs/diagrams/workflow.svg and docs/diagrams/workflow-dark.svg

Two files, because one was a white slab. The diagram paints its own opaque white
background, so on a dark GitHub page it stayed legible and looked like a sheet of paper
dropped onto the page. The README now serves whichever matches the reader's theme through
a <picture> element.

The light palette is unchanged to the digit, and CI asserts the light file regenerates
byte-identically, so that claim is checked rather than asserted.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DIAGRAMS = ROOT / "docs" / "diagrams"
OUTPUTS = {"light": DIAGRAMS / "workflow.svg", "dark": DIAGRAMS / "workflow-dark.svg"}

A1 = RESULTS / "assignment1_finial_gen600_atol-3"
A3 = RESULTS / "assignment3"
A4 = RESULTS / "assignment4_gen600"
A5 = RESULTS / "assignment5"

FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"

# Two palettes, same geometry. The light values are exactly what this file used before a
# dark variant existed, so the light SVG regenerates byte-identically and the CI step that
# asserts that keeps proving it.
#
# The dark values are GitHub's own dark surface and accent family, which is what the other
# repositories in this portfolio already use for their dark diagrams, so the two look like
# they came from one hand.
PALETTES = {
    "light": {
        "paper": "#FFFFFF",
        "ink": "#111827",
        "muted": "#4B5563",
        # Was #9CA3AF, which put the caption class at 2.54:1 on white. That is a WCAG AA
        # failure for normal text, and it had been shipped in the committed SVG. #6B7280
        # measures 4.83:1 on the paper and 4.63:1 on the grey panel. It also darkens the
        # arrows, which share this colour, and they were faint for the same reason.
        "faint": "#6B7280",
        "hair": "#E5E7EB",
        "blue": "#2563EB", "blue_bg": "#EFF6FF",
        "green": "#059669", "green_bg": "#ECFDF5",
        "amber": "#D97706", "amber_bg": "#FFFBEB",
        "red": "#DC2626", "red_bg": "#FEF2F2",
        "grey_bg": "#F9FAFB",
    },
    "dark": {
        "paper": "#0D1117",
        "ink": "#E6EDF3",
        # Brighter than GitHub's own #8B949E on purpose: this colour carries the box
        # sub-lines, which sit on the coloured panels rather than on the paper, and the
        # darkest of those is the blue at #0D2847. #9FB1C1 measures 6.75:1 there.
        "muted": "#9FB1C1",
        # 5.07:1 on the paper and 4.64:1 on the grey panel. #6E7681, the obvious match for
        # the light value, measured 4.12:1 and would have shipped the same failure in the
        # other theme.
        "faint": "#7D8590",
        "hair": "#30363D",
        "blue": "#58A6FF", "blue_bg": "#0D2847",
        "green": "#3FB950", "green_bg": "#0D2818",
        "amber": "#D29922", "amber_bg": "#2D2410",
        "red": "#F85149", "red_bg": "#2D1214",
        "grey_bg": "#161B22",
    },
}

# Set by apply_theme before a diagram is built.
#
# Module-level names mutated in place, rather than a palette threaded through every call.
# That is the unlovely choice on purpose: box() and arrow() are called about twenty times
# with these constants by name, and rewriting every call site to carry a palette would be
# a large diff whose only job is to prove the light output did not move. This way the
# light path is the same code it always was.
PAPER = INK = MUTED = FAINT = HAIR = ""
BLUE = BLUE_BG = GREEN = GREEN_BG = AMBER = AMBER_BG = RED = RED_BG = GREY_BG = ""


def apply_theme(theme: str) -> None:
    global PAPER, INK, MUTED, FAINT, HAIR, GREY_BG
    global BLUE, BLUE_BG, GREEN, GREEN_BG, AMBER, AMBER_BG, RED, RED_BG
    p = PALETTES[theme]
    PAPER, INK, MUTED = p["paper"], p["ink"], p["muted"]
    FAINT, HAIR = p["faint"], p["hair"]
    BLUE, BLUE_BG = p["blue"], p["blue_bg"]
    GREEN, GREEN_BG = p["green"], p["green_bg"]
    AMBER, AMBER_BG = p["amber"], p["amber_bg"]
    RED, RED_BG = p["red"], p["red_bg"]
    GREY_BG = p["grey_bg"]


def text_of(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing: {path.relative_to(ROOT).as_posix()}")
    return path.read_text(encoding="utf-8")


def labelled(path: Path, label: str) -> float:
    match = re.search(rf"^\s*{re.escape(label)}:\s*([-\d.eE+]+)", text_of(path), re.M)
    if not match:
        raise SystemExit(f"{label!r} not found in {path.name}")
    return float(match.group(1))


def top_parameter(folder: str) -> tuple[str, float]:
    with (A3 / folder / "sobol_indices_corrected.csv").open(newline="",
                                                            encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda r: float(r["ST"]))
    return best["Parameter"], float(best["ST"])


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="sub", anchor="start"):
    return (f'  <text class="{cls}" x="{x}" y="{y}" '
            f'text-anchor="{anchor}">{esc(s)}</text>\n')


def box(x, y, w, h, fill, edge, title, lines):
    out = (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
        f'stroke="{edge}" stroke-width="1.4"/>\n'
        f'  <text class="lbl" x="{x + 15}" y="{y + 25}">{esc(title)}</text>\n'
    )
    for i, line in enumerate(lines):
        out += text(x + 15, y + 46 + i * 17, line, "sub")
    return out


def arrow(x1, y1, x2, y2, colour=None):
    # Resolved here, not in the signature: a default binds once at import, which would
    # pin every arrow to whichever theme was applied first.
    if colour is None:
        colour = FAINT
    head = 8.0
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    ex, ey = x2 - ux * head, y2 - uy * head
    px, py = -uy, ux
    return (
        f'  <line x1="{x1}" y1="{y1}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{colour}" '
        f'stroke-width="1.6"/>\n'
        f'  <polygon points="{x2},{y2} {ex + px * 4.4:.1f},{ey + py * 4.4:.1f} '
        f'{ex - px * 4.4:.1f},{ey - py * 4.4:.1f}" fill="{colour}"/>\n'
    )


def main(theme: str = "light") -> int:
    # Called once per theme. The result files are re-read each time, which costs nothing
    # and keeps the drawing code below identical to what produced the light SVG before a
    # dark one existed.
    apply_theme(theme)

    final_nse = labelled(A1 / "final_debug_summary.txt", "final_nse")
    gens = sum(1 for _ in (A1 / "optimization_gen_summary.csv").open()) - 1
    global_top, global_st = top_parameter("Assignment3_Results_full_range_nse")
    narrow_top, _ = top_parameter("Assignment3_narrow_NSE")

    text4 = text_of(A4 / "uncertainty_analysis_summary.txt")
    n_series = int(re.search(r"Number of perturbed series:\s*(\d+)", text4).group(1))
    std = re.search(r"C = N\([\d.]+, ([\d.]+)\)", text4).group(1)
    marc4 = re.search(r"Relative PPT Change:\s*Mean:\s*([\d.]+)%", text4).group(1)
    ref_nse = 1.0 - labelled(A4 / "uncertainty_analysis_summary.txt", "Reference OFV")
    a4_nse = 1.0 - labelled(A4 / "uncertainty_analysis_summary.txt", "OFV mean")

    text5 = text_of(A5 / "uncertainty_analysis_summary.txt")
    bound = re.search(r"water level: \[-(\d+), (\d+)\]", text5).group(2)
    a5_nse = 1.0 - labelled(A5 / "uncertainty_analysis_summary.txt", "OFV mean")
    recovered = re.search(r"Compensation fraction: ([\d.]+)%", text5).group(1)

    W, H = 980, 700
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="{FONT}">\n'
        f'  <rect width="{W}" height="{H}" fill="{PAPER}"/>\n'
        f"  <defs><style>\n"
        f"    .h {{ fill:{INK}; font-size:19px; font-weight:600; }}\n"
        f"    .s {{ fill:{MUTED}; font-size:12.5px; }}\n"
        f"    .lbl {{ fill:{INK}; font-size:13.5px; font-weight:600; }}\n"
        f"    .sub {{ fill:{MUTED}; font-size:11px; }}\n"
        f"    .cap {{ fill:{FAINT}; font-size:11px; }}\n"
        f"  </style></defs>\n"
    )
    svg += text(30, 36, "Uncertainty quantification in hydrology: the seminar chain", "h")
    svg += text(30, 58, "One calibrated model, then four ways of asking how much of "
                        "its skill is real.", "s")

    svg += box(30, 86, 250, 92, GREY_BG, HAIR, "Forcing and model", [
        "temperature, precipitation, PET",
        "HBV001a, lumped rainfall-runoff",
        "hourly, 18 parameters",
    ])
    svg += arrow(285, 132, 335, 132)
    svg += box(340, 86, 280, 92, BLUE_BG, BLUE, "Assignment 1, calibration", [
        "differential evolution on the objective",
        f"{gens} generations",
        f"best NSE {final_nse:.4f}",
    ])
    svg += arrow(625, 132, 675, 132)
    svg += box(680, 86, 270, 92, GREY_BG, HAIR, "Process turn-off", [
        "each store disabled in turn",
        "lower reservoir is load-bearing",
        "groundwater is not, on this event",
    ])

    svg += arrow(480, 184, 480, 218)
    svg += text(30, 214, "Which parameters matter, and where you ask from", "lbl")

    svg += box(30, 232, 440, 86, AMBER_BG, AMBER, "Assignment 2, local sensitivity", [
        "one parameter at a time, plus and minus 30%",
        f"near the optimum {narrow_top} leads",
    ])
    svg += box(510, 232, 440, 86, AMBER_BG, AMBER, "Assignment 3, global sensitivity", [
        "Sobol indices, Saltelli sampling",
        f"over the full range {global_top} leads, total-order {global_st:.2f}",
    ])

    svg += arrow(480, 324, 480, 358)
    svg += text(30, 354, "How much of the skill survives error in the data", "lbl")

    svg += box(30, 372, 440, 100, GREEN_BG, GREEN, "Assignment 4, input uncertainty", [
        f"{n_series:,} precipitation series, multiplier N(1.0, {std})",
        f"mean absolute change {marc4}%",
        f"NSE {ref_nse:.4f} to {a4_nse:.4f}, essentially unchanged",
    ])
    svg += box(510, 372, 440, 100, RED_BG, RED, "Assignment 5, output uncertainty", [
        f"{n_series:,} discharge series, water level plus or minus {bound} cm",
        "rating curve, two power laws blended by a sigmoid",
        f"NSE {ref_nse:.4f} to {a5_nse:.4f}",
    ])

    svg += arrow(480, 478, 480, 512)
    svg += box(30, 526, 920, 76, GREY_BG, HAIR, "What the sequence shows", [
        f"Error in the observed discharge costs roughly "
        f"{ref_nse - a5_nse:.2f} of NSE; error in the rainfall costs "
        f"{ref_nse - a4_nse:.4f}.",
        f"Recalibrating against the corrupted discharge recovers {recovered}% of the "
        f"loss, so fitting harder cannot reveal that the target is wrong.",
    ])
    svg += text(30, 632, "Exercise 2 calibrates a MODFLOW groundwater model with "
                         "parallel DREAM; Exercise 3 applies ROPE over two decades "
                         "of daily HBV runs.", "cap")
    svg += text(30, 652, "Generated by scripts/figures/generate_diagram.py from the "
                         "files in results/.", "cap")
    svg += "</svg>\n"

    # A newline inside a text element renders as a space and silently ruins layout.
    for chunk in svg.split("<text")[1:]:
        body = chunk.split(">", 1)[1].split("</text>")[0]
        assert "\n" not in body, "newline inside a text element"

    out = OUTPUTS[theme]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8", newline="\n")
    print(f"  wrote {out.relative_to(ROOT).as_posix()}")

    # The numbers are the same in both files, so report them once rather than twice.
    if theme == "light":
        print(f"    calibration      NSE {final_nse:.4f} over {gens} generations")
        print(f"    precipitation    NSE {a4_nse:.4f}, multiplier sd {std}")
        print(f"    rating curve     NSE {a5_nse:.4f}, water level bound {bound} cm")
    return 0


if __name__ == "__main__":
    # Both themes, light first so its summary prints above the dark file's line.
    raise SystemExit(max(main(theme) for theme in ("light", "dark")))

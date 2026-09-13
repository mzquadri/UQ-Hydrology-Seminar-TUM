"""Check that the versioned deliverables are present, parse, and match results/.

Three things are checked. The files the README points at exist, the assignment
scripts still compile, and each committed figure was drawn from the version of
results/ that is committed beside it.

The last one is what the README promises when it says a figure cannot show a
value the runs did not produce. Continuous integration regenerated the figures
into the runner's working tree and never looked at what it had replaced, so a
figure left behind by a change to results/ would have gone unnoticed.

Comparing the images byte for byte is not available here: requirements.txt pins
no versions and matplotlib renders the same figure differently between releases.
What is compared instead is the digest of each result file the figure recorded
reading when it was written.
"""

import hashlib
import json
import py_compile
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"

#: Written by scripts/figures/portfolio_style.py into every figure it saves.
INPUTS_KEY = "ResultInputs"

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
NUL = bytes([0x00])
REQUIRED = (
    "README.md",
    "requirements.txt",
    "code/Ass_01_Model_Parameter_Optimisation_Group_B.py",
    "code/Ass_02_local_SA_Group_B.py",
    "code/Ass_03_Global_SA_Group_B.py",
    "code/Ass_04_Input_Uncertainty_Group_B.py",
    "code/Ass_05_Output_Uncertain_Group_B.py",
    "code/Ass_05_fittingCurve_Group_B.py",
    "results/assignment1_finial_gen600_atol-3/optimization_gen_summary.csv",
    "results/assignment3/Assignment3_narrow_NSE/sobol_indices_corrected.csv",
    "Overleaf_Projects/Mathematical methods for uncertainty quantification"
    " in hydrology/main.tex",
)



def png_text(path: Path) -> dict[str, str]:
    """The tEXt entries of a PNG, read without a third-party imaging library."""
    raw = path.read_bytes()
    if raw[:len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise SystemExit(f"{path.name} is not a PNG")
    entries, offset = {}, len(PNG_SIGNATURE)
    while offset + 8 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        if kind == b"tEXt":
            key, _, value = raw[offset + 8:offset + 8 + length].partition(NUL)
            entries[key.decode("latin-1")] = value.decode("latin-1")
        elif kind == b"IEND":
            break
        offset += 12 + length
    return entries


def check_figures() -> list[str]:
    """Each figure against the result files it recorded reading."""
    failures: list[str] = []
    figures = sorted(FIGURES.glob("*.png"))
    if not figures:
        return ["No figures in docs/figures; run scripts/figures/generate_figures.py"]

    renderers, checked = set(), 0
    for figure in figures:
        text = png_text(figure)
        renderers.add(text.get("Software", "unrecorded"))
        if INPUTS_KEY not in text:
            failures.append(
                f"{figure.name} records no inputs; regenerate it with "
                f"scripts/figures/generate_figures.py")
            continue
        for name, digest in json.loads(text[INPUTS_KEY]).items():
            source = ROOT / name
            if not source.is_file():
                failures.append(f"{figure.name} was drawn from {name}, which is gone")
                continue
            now = hashlib.sha256(source.read_bytes()).hexdigest()[:len(digest)]
            if now != digest:
                failures.append(
                    f"{figure.name} was drawn from an older {name}; "
                    f"regenerate the figures")
            checked += 1

    # A figure regenerated on its own carries a different matplotlib version from
    # the rest as soon as the installed version moves, which is what a
    # half-finished regeneration looks like.
    if len(renderers) > 1:
        failures.append(f"the figures were not rendered together: {sorted(renderers)}")

    if not failures:
        print(f"Figure check passed: {checked} result files behind "
              f"{len(figures)} figures are the committed ones.")
    return failures


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        print("Missing required artifacts:", *missing, sep="\n- ", file=sys.stderr)
        return 1

    for source in (ROOT / "code").glob("*.py"):
        py_compile.compile(source, doraise=True)

    print(f"Repository check passed: {len(REQUIRED)} required artifacts available.")

    failures = check_figures()
    if failures:
        print("", *failures, sep="\n- ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check that the versioned hydrology deliverables are present and parseable."""

from pathlib import Path
import py_compile
import sys


ROOT = Path(__file__).resolve().parents[1]
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
    "Overleaf_Projects/Mathematical methods for uncertainty quantification in hydrology/main.tex",
)


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        print("Missing required artifacts:", *missing, sep="\n- ", file=sys.stderr)
        return 1

    for source in (ROOT / "code").glob("*.py"):
        py_compile.compile(source, doraise=True)

    print(f"Repository check passed: {len(REQUIRED)} required artifacts available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

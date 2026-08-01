"""Check that the documented reference-run artifacts are available."""

from pathlib import Path
import py_compile
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "src/generate_data.py",
    "src/pipeline.py",
    "results/model_results.json",
    "results/classification_report.txt",
    "results/figures/03_model_comparison.png",
)


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        print("Missing required artifacts:", *missing, sep="\n- ")
        return 1

    for path in ("src/generate_data.py", "src/pipeline.py"):
        py_compile.compile(str(ROOT / path), doraise=True)

    print(f"Repository check passed: {len(REQUIRED_PATHS)} required artifacts available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Rebuild the paper figures from the bundled processed plotting snapshot."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["02_batch_size_scaling", "03_scaling_and_BIOscore",
        "04_learning_rate_scaling", "05_depth_width_scaling",
        "06_geneformer_surface_contours"]


def main():
    if not shutil.which("rsvg-convert"):
        raise SystemExit("Install librsvg (rsvg-convert) before building figures.")
    for index in range(1, 7):
        subprocess.run([sys.executable, "-m", f"tools.paper_plots.figure_{index:02d}"],
                       cwd=ROOT, check=True)
        if index > 1:
            stem = MAIN[index - 2]
            subprocess.run(["rsvg-convert", "--format=pdf", "--output",
                            str(ROOT / "manuscript/figures" / f"{stem}.pdf"),
                            str(ROOT / "plots/paper" / f"{stem}.svg")], check=True)
    subprocess.run([sys.executable, "-m", "tools.paper_plots.supplementary"],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

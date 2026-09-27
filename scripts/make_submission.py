"""Build <team>_submission.zip in the layout the challenge asks for.

    python scripts/make_submission.py --team MyTeam

Layout:
    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/, README.md, requirements.txt, pyproject.toml, tests/, scripts/}
    Documentation_template.md   (docs/Documentation_template.md if filled in, else the blank template)

Runs the official validator first and refuses to package a failing submission.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ITEMS = ["src", "tests", "scripts", "README.md", "requirements.txt", "pyproject.toml"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    matching = args.output_dir / "matching_results.tsv"
    candidate = args.output_dir / "candidate_pairs.tsv"
    if not args.skip_validate:
        res = subprocess.run([
            sys.executable, "student_resource/utils/validate_submission.py",
            "--matching", str(matching), "--candidate", str(candidate),
            "--test-dir", "student_resource/dataset/test",
        ], cwd=ROOT)
        if res.returncode != 0:
            print("validator failed; not packaging")
            return 1

    doc = ROOT / "docs" / "Documentation_template.md"
    if not doc.exists():
        doc = ROOT / "student_resource" / "Documentation_template.md"

    out = ROOT / f"{args.team}_submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(matching, "output/matching_results.tsv")
        z.write(candidate, "output/candidate_pairs.tsv")
        base = "code/business_entity_resolution"
        for item in CODE_ITEMS:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts and ".egg-info" not in str(f)]
            for f in files:
                z.write(f, f"{base}/{f.relative_to(ROOT).as_posix()}")
        z.write(doc, "Documentation_template.md")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

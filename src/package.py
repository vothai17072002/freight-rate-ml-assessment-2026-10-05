"""Create a portable submission ZIP, excluding environments and render scratch files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [".gitignore", ".gitattributes", "README.md", "requirements.txt", "requirements-lock.txt", "score.py",
         "assessment.pdf", "validation_predictions.csv"]
DIRECTORIES = ["src", "tests", "data", "experiments", "artifacts", "reports", "submission"]


def candidates():
    paths = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        for path in (ROOT / name).rglob("*"):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in {"__pycache__", "rendered", "video_assets"} for part in relative.parts):
                continue
            if path.suffix in {".zip", ".pyc"}:
                continue
            if relative.as_posix() == "submission/manifest.json":
                continue
            paths.append(path)
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "freight-rate-assessment-submission.zip")
    args = parser.parse_args()
    required = FILES + ["reports/freight_rate_report.pdf", "submission/december_chart_predictions.csv",
                        "reports/loom_script_en.md", "reports/candidate_notes_vi.md",
                        "artifacts/full_model.joblib", "artifacts/reduced_model.joblib"]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing submission files: {missing}")
    paths = candidates()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "accuracy_scope": "Recorded September-October retrospective holdout only; organizer's November-December labels are unavailable.",
                "loom_status": "Local script and optional narrated draft; candidate must obtain a real Loom link.",
                "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}}
    manifest_path = ROOT / "submission/manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(args.output) as archive:
        error = archive.testzip()
        if error:
            raise SystemExit(f"ZIP integrity failed: {error}")
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes, {len(paths)} files)")
    print("ZIP SHA256", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

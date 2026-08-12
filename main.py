from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_script(repo_root: Path, rel_path: str) -> None:
    script_path = repo_root / rel_path
    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    run_script(repo_root, "data/raw_script.py")
    run_script(repo_root, "data/handling.py")
    run_script(repo_root, "src/preprocessing.py")
    run_script(repo_root, "src/validate_ml_ready.py")
    run_script(repo_root, "src/train.py")
    run_script(repo_root, "src/evaluate.py")
    run_script(repo_root, "src/export_for_apk.py")


if __name__ == "__main__":
    main()

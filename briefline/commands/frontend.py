"""Start the Briefline Streamlit frontend with the active Python environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "frontend" / "streamlit_app.py"


def main(argv: Optional[Sequence[str]] = None) -> int:
    extra_args = list(argv or ())
    if any(argument in {"-h", "--help"} for argument in extra_args):
        print(
            "Usage: python -m briefline frontend [STREAMLIT OPTIONS]\n\n"
            "Starts frontend/streamlit_app.py with the active Python environment."
        )
        return 0
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *extra_args],
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

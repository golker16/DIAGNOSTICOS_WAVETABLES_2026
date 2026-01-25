# wtdiag.py
# Wrapper importable (sin guiones) para que wtgui.py pueda hacer `import wtdiag`.
# Carga wtdiag-2.py, que carga wtdiag-1.py.

import importlib.util
import sys
from pathlib import Path


def _resource_path(rel_name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel_name


def _load_front():
    front_path = _resource_path("wtdiag-2.py")
    if not front_path.exists():
        front_path = Path(__file__).resolve().parent / "wtdiag-2.py"

    spec = importlib.util.spec_from_file_location("wtdiag_front", str(front_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude crear spec para {front_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_front = _load_front()

cmd_index = _front.cmd_index
cmd_diag = _front.cmd_diag
cmd_match = _front.cmd_match

main = _front.main
FEATURE_SR = getattr(_front, "FEATURE_SR", 48000)

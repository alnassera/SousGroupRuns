from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_PACKAGE_ROOT = Path(__file__).resolve().parent
_BUNDLE_ROOT = _PACKAGE_ROOT.parent
_bundle_root_text = str(_BUNDLE_ROOT)
if _bundle_root_text not in sys.path:
    sys.path.insert(0, _bundle_root_text)


def _load_bundle_module(module_name: str) -> ModuleType:
    existing = sys.modules.get(module_name)
    module_path = _BUNDLE_ROOT / f"{module_name}.py"
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == module_path.resolve():
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bundle module '{module_name}' from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def export_bundle_module(namespace: dict, module_name: str) -> None:
    module = _load_bundle_module(module_name)
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in module.__dict__ if not name.startswith("_")]
    for name in public_names:
        namespace[name] = getattr(module, name)
    namespace["__all__"] = list(public_names)
    namespace["__doc__"] = module.__doc__

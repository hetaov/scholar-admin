# Auto-generated backward-compatibility shim (READ+WRITE transparent proxy).
# Original file moved to: services.metrics.gamification
# This shim:
#   - READS (access, import *) transparently come from the target module.
#   - WRITES (monkeypatch.setattr, direct assignment) go through to the
#     target module, so runtime code inside services.metrics.gamification that reads module-level
#     globals (e.g. AUTH_MODE, LLM_JUDGE_MODEL, _call_judge, ...) always sees
#     the monkey-patched value, even when tests patch via the old shim path.
#   - Underscore-prefixed symbols like _repair_json are fully exported.
import sys as _sys
import importlib as _importlib

_target = _importlib.import_module("services.metrics.gamification")
_target_name = "services.metrics.gamification"
_shim_name = __name__

class _ShimModule(type(_sys)):
    """Custom module class that forwards attribute READ/WRITE to the target.
    
    This lets monkeypatch.setattr("services.X", attr, val) actually
    mutate the real services.<group>.<mod> namespace where code runs.
    """
    _target_mod = _target
    _dct = _sys.modules[_shim_name].__dict__  # shim's original dict

    def __getattr__(cls, name):
        try:
            return getattr(_target, name)
        except AttributeError:
            raise AttributeError(
                f"module '{_shim_name}' (shim for {_target_name}) "
                f"has no attribute '{name}'"
            )

    def __setattr__(cls, name, value):
        # Rout ALL attribute writes to the REAL target module.
        # Exception: Python-internal dunder names (used by import machinery)
        # go to the shim's own dict to avoid breaking import system.
        if name.startswith("__") and name.endswith("__"):
            _ShimModule._dct[name] = value
        else:
            setattr(_target, name, value)

    def __dir__(cls):
        return sorted(set(list(_ShimModule._dct.keys()) + list(vars(_target).keys())))

# Replace the shim module's class with our proxying class
_sys.modules[_shim_name].__class__ = _ShimModule

# Also populate shim's __dict__ once so `from services.X import Y` /
# `from services.X import *` immediately resolve via normal Python lookup
# (Python skips __getattr__ if name already in module dict). We intentionally
# do NOT pre-populate non-dunder names so __getattr__ is always invoked
# (forces read delegation, keeping patched value in sync).
# Only ensure __all__ points to target's public-ish names.
try:
    __all__
except NameError:
    __all__ = [n for n in vars(_target).keys() if not n.startswith("__")]

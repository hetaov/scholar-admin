# Auto-generated backward-compatibility shim (READ+WRITE transparent proxy).
# Original file moved to: services.routes.ai_v3
# This shim:
#   - READS (access, import *) transparently come from the target module.
#   - WRITES (monkeypatch.setattr, direct assignment) go through to the
#     target module, so runtime code inside services.routes.ai_v3 that reads module-level
#     globals (e.g. _background_tasks, _SUPPORTED_PREFERRED_TYPES, ...) always sees
#     the monkey-patched value, even when tests patch via the old shim path.
#   - Underscore-prefixed symbols like _fail are fully exported.
import sys as _sys
import importlib as _importlib

_target = _importlib.import_module("services.routes.ai_v3")
_target_name = "services.routes.ai_v3"
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
        if name.startswith("__") and name.endswith("__"):
            _ShimModule._dct[name] = value
        else:
            setattr(_target, name, value)

    def __dir__(cls):
        return sorted(set(list(_ShimModule._dct.keys()) + list(vars(_target).keys())))

# Replace the shim module's class with our proxying class
_sys.modules[_shim_name].__class__ = _ShimModule

try:
    __all__
except NameError:
    __all__ = [n for n in vars(_target).keys() if not n.startswith("__")]

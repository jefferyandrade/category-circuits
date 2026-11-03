"""Deferred imports for optional numerical and model helpers."""
from __future__ import annotations

import importlib

class LazyAttribute:
    """Resolve an optional helper once; callers receive the original object."""
    def __init__(self, module: str, attribute: str | None = None):
        self.module, self.attribute = module, attribute
        self._value = None

    def resolve(self):
        if self._value is None:
            module = importlib.import_module(self.module)
            self._value = module if self.attribute is None else getattr(module, self.attribute)
        return self._value

    def __call__(self, *args, **kwargs):
        return self.resolve()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.resolve(), name)


def prepare_decision_imports() -> None:
    """Resolve first-decision numerical helpers during setup."""
    from mechanistic_systems.inference import sequential_bce
    for name in ("norm", "student_t", "chi2"):
        value = getattr(sequential_bce, name)
        if isinstance(value, LazyAttribute):
            setattr(sequential_bce, name, value.resolve())

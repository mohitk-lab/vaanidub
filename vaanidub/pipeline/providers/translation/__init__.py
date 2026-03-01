"""Translation providers — lazy imports to avoid loading ML deps at import time."""

_PROVIDERS = {
    "IndicTrans2Provider": ".indictrans2",
    "GoogleTranslateProvider": ".google_translate",
}

__all__ = list(_PROVIDERS)


def __getattr__(name: str):
    if name in _PROVIDERS:
        import importlib
        module = importlib.import_module(_PROVIDERS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

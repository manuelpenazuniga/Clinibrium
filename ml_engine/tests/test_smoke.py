"""TB1.0 — smoke: el paquete y sus deps ML importan (kill-gate del track)."""


def test_package_imports() -> None:
    import ml_engine  # noqa: F401
    import ml_engine.core  # noqa: F401

    assert ml_engine.__version__ == "0.1.0"


def test_ml_deps_import() -> None:
    """Confirma que las wheels pesadas resolvieron en py3.12/arm64."""
    import catboost  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import shap  # noqa: F401
    import sklearn  # noqa: F401

    assert True


def test_core_never_imports_clinibrium_or_domains() -> None:
    """AD-15/AD-16: el core es agnóstico — no importa A ni dominios."""
    import inspect
    import pkgutil

    import ml_engine.core as core_pkg

    offenders = []
    for mod in pkgutil.iter_modules(core_pkg.__path__):
        m = __import__(f"ml_engine.core.{mod.name}", fromlist=["_"])
        src = inspect.getsource(m)
        if "clinibrium" in src or "ml_engine.domains" in src:
            offenders.append(mod.name)
    assert not offenders, f"core importa A/dominios: {offenders}"

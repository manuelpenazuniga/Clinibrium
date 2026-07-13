"""TB1.0 — smoke: the package and its ML deps import (track kill-gate)."""


def test_package_imports() -> None:
    import ml_engine  # noqa: F401
    import ml_engine.core  # noqa: F401

    assert ml_engine.__version__ == "0.1.0"


def test_ml_deps_import() -> None:
    """Confirms the heavy wheels resolved on py3.12/arm64."""
    import catboost  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import shap  # noqa: F401
    import sklearn  # noqa: F401

    assert True


def test_core_never_imports_clinibrium_or_domains() -> None:
    """AD-15/AD-16: the core is agnostic — it does not IMPORT A nor domains.

    Checks real imports via AST (not mentions in docstrings/comments).
    """
    import ast
    import inspect
    import pkgutil

    import ml_engine.core as core_pkg

    def _imported_modules(source: str) -> set[str]:
        mods: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        return mods

    offenders = []
    for mod in pkgutil.iter_modules(core_pkg.__path__):
        m = __import__(f"ml_engine.core.{mod.name}", fromlist=["_"])
        imported = _imported_modules(inspect.getsource(m))
        if any(x.startswith("clinibrium") or x.startswith("ml_engine.domains") for x in imported):
            offenders.append(mod.name)
    assert not offenders, f"core imports A/domains: {offenders}"

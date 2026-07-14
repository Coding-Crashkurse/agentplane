"""Every published package ships a PEP 561 marker (FEEDBACK 1.1).

Without ``py.typed`` the packages are typed but *invisibly* so: a consumer
running ``mypy --strict`` gets ``import-untyped`` and has to work around it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
PACKAGES = ("core", "sdk", "registry", "runtime")


@pytest.mark.parametrize("package", PACKAGES)
def test_package_ships_py_typed(package: str) -> None:
    marker = REPO_ROOT / "packages" / package / "src" / f"agentplane_{package}" / "py.typed"
    assert marker.is_file(), f"agentplane-{package} is missing its py.typed marker"

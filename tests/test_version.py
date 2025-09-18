import sys
import tomllib
from pathlib import Path

import gwrefpy


def test_version_consistency() -> None:
    """Test that all version declarations in the project are consistent."""
    project_root = Path(__file__).parent.parent
    versions = {}

    # Read version from pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
        versions["pyproject.toml"] = pyproject["project"]["version"]

    # Get version from gwrefpy module
    versions["gwrefpy.__version__"] = gwrefpy.__version__

    # Import and read version from docs/conf.py
    docs_path = str(project_root / "docs")
    sys.path.insert(0, docs_path)
    try:
        import conf

        versions["docs/conf.py"] = conf.release
    finally:
        sys.path.remove(docs_path)

    # Check all versions are the same
    unique_versions = set(versions.values())
    assert len(unique_versions) == 1, f"Version mismatch detected:\n" + "\n".join(
        [f"  {source}: {version}" for source, version in versions.items()]
    )

"""The package version has exactly one source of truth: the installed
distribution metadata (which comes from pyproject.toml). A hard-coded
``__version__`` drifted to 0.1.0 after the 0.1.1 bump and was published in a
report's provenance line; this pins the two together."""
from importlib import metadata

import robustrep


def test_dunder_version_matches_distribution_metadata():
    assert robustrep.__version__ == metadata.version("robustrep")


def test_version_is_not_the_stale_literal():
    assert robustrep.__version__ != "0.1.0"

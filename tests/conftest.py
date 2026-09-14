import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_records(rows):
    """rows: list of dicts with at least rater, ratee, value. Fills defaults."""
    defaults = dict(scale="d0", tag="quality", ts=0, evidence_uri=None, source="test")
    return pd.DataFrame([{**defaults, **r} for r in rows])


@pytest.fixture
def records_factory():
    return make_records

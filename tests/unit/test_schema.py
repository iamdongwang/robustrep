import pandas as pd
import pytest

from robustrep.schema import (
    RECORD_COLUMNS, RESULT_COLUMNS, make_scale, scale_decimals, validate_records,
)


def test_record_columns_are_eight():
    assert RECORD_COLUMNS == [
        "rater", "ratee", "value", "scale", "tag", "ts", "evidence_uri", "source"]


def test_scale_roundtrip():
    assert make_scale(2) == "d2"
    assert scale_decimals("d2") == 2
    with pytest.raises(ValueError):
        scale_decimals("x2")


def test_validate_fills_optional_columns(records_factory):
    df = validate_records(records_factory([dict(rater="a", ratee="1", value=87)]))
    assert list(df["evidence_level"]) == [0]
    assert list(df["cluster"]) == ["a"]
    assert list(df["revoked"]) == [0]


def test_validate_rejects_missing_column():
    with pytest.raises(ValueError, match="missing columns"):
        validate_records(pd.DataFrame({"rater": ["a"]}))


def test_validate_keeps_optional_if_present(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=1, evidence_level=3, cluster="c")])
    out = validate_records(df)
    assert out.loc[0, "evidence_level"] == 3 and out.loc[0, "cluster"] == "c"


def test_result_columns():
    assert "robust_score" in RESULT_COLUMNS and "naive_mean" in RESULT_COLUMNS

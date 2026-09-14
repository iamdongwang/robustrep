import pandas as pd
import pytest

from robustrep.schema import (
    RECORD_COLUMNS, RESULT_COLUMNS, make_scale, scale_decimals, validate_records,
)


def test_record_columns_are_eight():
    assert RECORD_COLUMNS == [
        "rater", "ratee", "value", "scale", "tag", "ts", "evidence_uri", "source"]
    assert len(RECORD_COLUMNS) == 8


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
    assert out.loc[0, "evidence_level"] == 3
    assert out.loc[0, "cluster"] == "c"


def test_result_columns():
    assert RESULT_COLUMNS == [
        "ratee", "robust_score", "ci_low", "ci_high", "n_clusters", "n_raw",
        "zero_evidence_ratio", "sybil_flag", "naive_mean", "insufficient",
    ]


def test_null_rater_rejected(records_factory):
    df = records_factory([dict(rater=None, ratee="1", value=1)])
    with pytest.raises(ValueError, match="column 'rater'"):
        validate_records(df)


def test_non_numeric_value_rejected(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value="abc")])
    with pytest.raises(ValueError, match="column 'value'"):
        validate_records(df)


def test_ts_coerced_to_int(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=1, ts="1700")])
    out = validate_records(df)
    assert out["ts"].dtype == "int64"
    assert out.loc[0, "ts"] == 1700


def test_evidence_level_domain(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=1, evidence_level=9)])
    with pytest.raises(ValueError, match="evidence_level"):
        validate_records(df)
    df2 = records_factory([dict(rater="a", ratee="1", value=1, evidence_level=2.5)])
    with pytest.raises(ValueError, match="evidence_level"):
        validate_records(df2)


def test_cluster_inherits_string_identity(records_factory):
    df = records_factory([dict(rater=123, ratee="1", value=1)])
    out = validate_records(df)
    assert out.loc[0, "cluster"] == "123"
    assert out.loc[0, "rater"] == "123"


def test_input_not_mutated(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=1)])
    copy = df.copy()
    validate_records(df)
    pd.testing.assert_frame_equal(df, copy)


def test_empty_frame_ok():
    df = pd.DataFrame({c: [] for c in RECORD_COLUMNS})
    out = validate_records(df)
    assert len(out) == 0
    for col in ("evidence_level", "cluster", "revoked"):
        assert col in out.columns


def test_make_scale_rejects_negative():
    with pytest.raises(ValueError):
        make_scale(-1)


def test_config_post_init():
    from robustrep.config import Config

    with pytest.raises(ValueError):
        Config(ci_level=1.5)
    with pytest.raises(ValueError):
        Config(evidence_weights=(1, 2, 3))


def test_non_finite_value_rejected(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=float("inf"))])
    with pytest.raises(ValueError, match="non-finite"):
        validate_records(df)


def test_duplicated_columns_rejected(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=1)])
    df.columns = ["rater" if c == "scale" else c for c in df.columns]
    with pytest.raises(ValueError, match="duplicated"):
        validate_records(df)


def test_evidence_level_non_numeric_rejected(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=1, evidence_level="x")])
    with pytest.raises(ValueError, match="evidence_level"):
        validate_records(df)


def test_bad_scale_rejected_at_boundary(records_factory):
    df = records_factory([dict(rater="a", ratee="1", value=1, scale="percent")])
    with pytest.raises(ValueError, match="column 'scale'"):
        validate_records(df)


def test_scale_decimals_over_255_rejected():
    with pytest.raises(ValueError):
        scale_decimals("d256")
    with pytest.raises(ValueError):
        make_scale(256)
    assert scale_decimals("d255") == 255
    assert make_scale(255) == "d255"
    assert scale_decimals("d19") == 19
    assert make_scale(19) == "d19"


def test_config_more_checks():
    from robustrep.config import Config

    with pytest.raises(ValueError):
        Config(sybil_jaccard=5.0)
    with pytest.raises(ValueError):
        Config(rpc_urls=())
    with pytest.raises(ValueError):
        Config(bootstrap_n=-1)
    with pytest.raises(ValueError):
        Config(min_clusters=0)


def test_config_rejects_zero_evidence_weight():
    from robustrep.config import Config

    with pytest.raises(ValueError):
        Config(evidence_weights=(0, 0.3, 0.7, 1))


def test_config_sybil_pair_and_jaccard_bounds():
    from robustrep.config import Config

    with pytest.raises(ValueError):
        Config(sybil_jaccard=0)
    with pytest.raises(ValueError):
        Config(sybil_max_pairs=0)


def test_config_rejects_invalid_timing_and_count_params():
    from robustrep.config import Config

    with pytest.raises(ValueError):
        Config(bootstrap_n=-1)
    with pytest.raises(ValueError):
        Config(min_clusters=0)
    with pytest.raises(ValueError):
        Config(rpc_urls=())


def test_config_rejects_invalid_window_share_and_chunk_params():
    from robustrep.config import Config

    with pytest.raises(ValueError):
        Config(sybil_window_s=-1)
    with pytest.raises(ValueError):
        Config(chunk_blocks=0)
    with pytest.raises(ValueError):
        Config(sybil_flag_share=-0.1)
    with pytest.raises(ValueError):
        Config(sybil_flag_share=1.1)

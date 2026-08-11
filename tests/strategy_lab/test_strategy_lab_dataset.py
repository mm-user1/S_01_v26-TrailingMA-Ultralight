from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from tools.strategy_lab.config import canonical_json_bytes, load_run_spec
from tools.strategy_lab.dataset import (
    CANDIDATE_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    METRIC_AXIS,
    SEGMENT_AXIS,
    DatasetError,
    atomic_write,
    atomic_write_group,
    atomic_write_json,
    group_matrix,
    project_candidates,
    result_row_values,
    segment_matrix,
    validate_artifact_record,
    validate_candidate_projection,
)

from tests.strategy_lab.phase1a_helpers import CURRENT_RUN_SPEC, fake_rows


@pytest.fixture(scope="module")
def plan():
    value = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False).plan
    assert value is not None
    return value


def test_frozen_schema_axes_are_exact():
    assert DATASET_SCHEMA_VERSION == "strategy_lab_dataset_v1"
    assert CANDIDATE_SCHEMA_VERSION == "strategy_lab_candidates_v1"
    assert SEGMENT_AXIS == ("is", "oos")
    assert METRIC_AXIS == (
        "net_profit_pct",
        "max_drawdown_pct",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "gross_profit",
        "gross_loss",
        "profit_factor",
        "win_rate_pct",
        "romad",
        "max_consecutive_losses",
        "sharpe_daily",
        "sharpe_daily_observations",
        "sharpe_daily_active_days",
        "sqn",
        "rejected_fill_count",
        "zero_size_entry_count",
        "invalid_stop_distance_count",
        "max_required_leverage",
        "flags",
    )


def test_candidate_projection_round_trips_all_480_rows_and_geometry(plan):
    versions = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False).generation["strategy"]
    projected = project_candidates(plan, versions)
    reparsed = json.loads(canonical_json_bytes(projected))

    validate_candidate_projection(reparsed, plan)
    assert projected["candidate_count"] == 480
    assert len(projected["candidates"]) == 480
    assert [row["candidate_id"] for row in projected["candidates"]] == list(range(1, 481))
    assert len({row["semantic_key"] for row in projected["candidates"]}) == 480
    for row in projected["candidates"]:
        assert len(row["axis_value_codes"]) == len(projected["global_axis_names"])
        assert len(row["active_axis_mask"]) == len(projected["global_axis_names"])
        assert all(
            active == (code != -1)
            for active, code in zip(row["active_axis_mask"], row["axis_value_codes"])
        )
        assert "start" not in row["params"] and "end" not in row["params"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version="wrong"),
        lambda value: value.update(candidate_count=479),
        lambda value: value["candidates"][0].update(row_index=1),
        lambda value: value["candidates"][0].update(candidate_id=2),
        lambda value: value["candidates"][0].update(semantic_key="wrong"),
    ],
)
def test_candidate_projection_schema_order_and_identity_are_strict(plan, mutate):
    versions = load_run_spec(CURRENT_RUN_SPEC, validate_inventory=False).generation["strategy"]
    projected = project_candidates(plan, versions)
    mutate(projected)
    with pytest.raises(DatasetError):
        validate_candidate_projection(projected, plan)


def test_direct_metric_mapping_preserves_unavailable_and_zero(plan):
    none_row = fake_rows(plan, daily_none=True)[0]
    zero_row = fake_rows(plan, daily_none=False)[0]
    none_values = result_row_values(none_row)
    zero_values = result_row_values(zero_row)

    assert np.isnan(none_values[METRIC_AXIS.index("sharpe_daily_observations")])
    assert np.isnan(none_values[METRIC_AXIS.index("sharpe_daily_active_days")])
    assert zero_values[METRIC_AXIS.index("sharpe_daily_observations")] == 0.0
    assert zero_values[METRIC_AXIS.index("sharpe_daily_active_days")] == 0.0
    assert zero_values[METRIC_AXIS.index("total_trades")] == 0.0
    assert np.isnan(zero_values[METRIC_AXIS.index("sqn")])
    assert zero_values[METRIC_AXIS.index("max_required_leverage")] == 0.0


def test_group_matrix_has_exact_alignment_shape_dtype_and_keeps_all_rows(plan):
    rows = fake_rows(plan)
    matrix = group_matrix(rows, rows, plan)

    assert matrix.shape == (480, 2, 20)
    assert matrix.dtype == np.float64
    assert matrix[0, 0, METRIC_AXIS.index("total_trades")] == 0.0
    assert matrix[479, 1, METRIC_AXIS.index("net_profit_pct")] == pytest.approx(47.9)
    assert np.isnan(matrix[:, :, METRIC_AXIS.index("sqn")]).all()


@pytest.mark.parametrize(
    "row_change",
    [
        {"status": "error", "error": "kernel failed"},
        {"total_trades": -1},
        {"total_trades": 1.5},
        {"total_trades": float("inf")},
        {"sharpe_daily_observations": -1},
    ],
)
def test_error_rows_and_malformed_counters_are_hard_failures(plan, row_change):
    row = replace(fake_rows(plan)[0], **row_change)
    with pytest.raises(DatasetError):
        result_row_values(row)


def test_missing_guardrail_and_wrong_execution_order_are_rejected(plan):
    row = fake_rows(plan)[0]
    guardrails = dict(row.guardrail_summary)
    guardrails.pop("flags")
    with pytest.raises(DatasetError, match="missing guardrail"):
        result_row_values(replace(row, guardrail_summary=guardrails))

    rows = list(fake_rows(plan))
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(DatasetError, match="candidate-ID order"):
        segment_matrix(rows, plan)
    with pytest.raises(DatasetError, match="row count"):
        segment_matrix(rows[:-1], plan)


def test_atomic_writer_uses_exact_float64_3d_contract_and_validates_artifact(tmp_path):
    target = tmp_path / "groups" / "AAAUSDT" / "window_01.npy"
    matrix = np.zeros((480, 2, 20), dtype=np.float64)
    atomic_write_group(target, matrix)

    assert np.array_equal(np.load(target, allow_pickle=False), matrix)
    assert not list(tmp_path.rglob("*.tmp"))
    record = {
        "path": target.relative_to(tmp_path).as_posix(),
        "size": target.stat().st_size,
        "sha256": __import__("hashlib").sha256(target.read_bytes()).hexdigest(),
        "shape": [480, 2, 20],
        "dtype": "float64",
    }
    assert validate_artifact_record(
        tmp_path, record, expected_shape=[480, 2, 20], expected_dtype="float64"
    )
    with pytest.raises(DatasetError, match="three-dimensional float64"):
        atomic_write_group(tmp_path / "bad.npy", matrix.astype(np.float32))
    with pytest.raises(DatasetError, match="three-dimensional float64"):
        atomic_write_group(tmp_path / "bad2.npy", matrix[:, 0])


def test_atomic_text_is_canonical_utf8_lf_and_temp_cleanup_is_task_owned(tmp_path):
    target = tmp_path / "value.json"
    unrelated = tmp_path / ".unrelated.tmp"
    unrelated.write_text("keep", encoding="utf-8")
    atomic_write_json(target, {"z": 1, "a": 2})

    assert target.read_bytes() == b'{"a":2,"z":1}\n'
    assert unrelated.read_text(encoding="utf-8") == "keep"

    def fail(temporary):
        temporary.write_bytes(b"partial")
        raise RuntimeError("controlled writer failure")

    with pytest.raises(RuntimeError, match="controlled writer failure"):
        atomic_write(tmp_path / "failed.json", fail)
    assert not (tmp_path / "failed.json").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not [path for path in tmp_path.iterdir() if "strategy-lab-" in path.name]

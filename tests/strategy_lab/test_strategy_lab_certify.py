from __future__ import annotations

import copy
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.grid_engine import rank_grid_results
from core.optuna_engine import OptimizationResult
from tools.strategy_lab import certify as certify_module
from tools.strategy_lab.certify import (
    MISMATCH_SAMPLE_LIMIT,
    PROTECTED_CANONICAL_OUTPUTS,
    SELECTED_TRIAL_NET_PROFIT_BASIS,
    WINDOW_NET_PROFIT_BASIS,
    _compare_grid_runs,
    _grid_run_parity_facts,
    _selected_slow_parity,
    _selected_slow_row_mismatches,
    allowed_roots_with_data_root,
    assert_certification_work_dir_allowed,
    assert_path_contained,
    candidate_identity_mappings,
    changed_snapshot_paths,
    geometry_candidate_ids,
    finite_mtm_group_facts,
    legacy_column_preservation_facts,
    select_primary_candidate,
    selected_trial_net_profit_from_lab,
    semantic_float_equal,
    utc_timestamps_equal,
    wfa_window_net_profit_from_lab,
)
from tools.strategy_lab.config import load_run_spec
from tools.strategy_lab.dataset import (
    METRIC_AXIS,
    SEGMENT_AXIS,
    DatasetError,
    project_candidates,
)
from ui import server_services

from tests.strategy_lab.phase1a_helpers import REPO_ROOT, fake_execute, fake_rows


def test_geometry_candidate_selection_is_fixed_before_metrics_are_available():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    projection = project_candidates(spec.plan, spec.generation["strategy"])

    selected = geometry_candidate_ids(projection)

    assert selected == (1, 3, 10, 13, 73, 240, 241, 385, 480)
    assert set(projection["global_axis_names"]) == {
        "stopX",
        "stopRR",
        "stopLP",
        "stopMaxPct",
        "stopMaxDays",
    }


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (None, float("nan"), True),
        (float("nan"), float("nan"), True),
        (None, 0.0, False),
        (float("inf"), None, True),
        (1.25, 1.25, True),
        (1.0, 1.0 + 5e-10, True),
        (1.0, 1.0 + 2e-9, False),
    ],
)
def test_semantic_float_comparison_preserves_unavailable_and_zero(left, right, expected):
    assert semantic_float_equal(left, right) is expected


def test_exact_primary_tie_matches_production_grid_ranking_contract():
    candidates = [
        {"candidate_id": 9, "semantic_key": "semantic-a"},
        {"candidate_id": 7, "semantic_key": "semantic-a"},
        {"candidate_id": 1, "semantic_key": "semantic-b"},
        {"candidate_id": 3, "semantic_key": "semantic-c"},
    ]
    profits = np.array([4.0, 4.0, 4.0, 3.0], dtype=np.float64)
    selected_index, tie_count = select_primary_candidate(profits, candidates)

    production_rows = []
    for candidate, profit in zip(candidates, profits):
        row = OptimizationResult(
            params={},
            net_profit_pct=float(profit),
            max_drawdown_pct=0.0,
            total_trades=1,
            optuna_trial_number=int(candidate["candidate_id"]),
        )
        row.semantic_key = candidate["semantic_key"]
        row.candidate_id = candidate["candidate_id"]
        production_rows.append(row)
    assert all(
        row.optuna_trial_number == row.candidate_id
        for row in production_rows
    )
    ranked = rank_grid_results(
        production_rows,
        objectives=["net_profit_pct"],
        primary_objective="net_profit_pct",
        constraints=[],
    )

    assert tie_count == 3
    assert candidates[selected_index] == {
        "candidate_id": 7,
        "semantic_key": "semantic-a",
    }
    assert candidates[selected_index]["semantic_key"] == ranked[0].semantic_key
    assert candidates[selected_index]["candidate_id"] == ranked[0].candidate_id


def test_net_profit_storage_bases_are_explicit_and_fixed():
    lab_value = 12.5
    assert WINDOW_NET_PROFIT_BASIS == "legacy_wfa_100"
    assert wfa_window_net_profit_from_lab(lab_value) == 1025.0
    assert SELECTED_TRIAL_NET_PROFIT_BASIS == "initial_capital_1000"
    assert selected_trial_net_profit_from_lab(lab_value) == lab_value


def test_allowed_csv_root_is_unique_and_monkeypatch_restores_after_failure(tmp_path):
    original = server_services.CSV_ALLOWED_ROOTS
    data_root = tmp_path / "market"
    data_root.mkdir()

    with pytest.raises(RuntimeError, match="controlled"):
        with pytest.MonkeyPatch.context() as scoped:
            updated = allowed_roots_with_data_root(
                server_services.CSV_ALLOWED_ROOTS,
                data_root,
            )
            scoped.setattr(server_services, "CSV_ALLOWED_ROOTS", updated)
            scoped.setattr(
                server_services,
                "CSV_ALLOWED_ROOTS",
                allowed_roots_with_data_root(
                    server_services.CSV_ALLOWED_ROOTS,
                    data_root,
                ),
            )
            assert server_services.CSV_ALLOWED_ROOTS.count(data_root.resolve()) == 1
            raise RuntimeError("controlled")

    assert server_services.CSV_ALLOWED_ROOTS is original


def test_normal_certification_tests_do_not_import_opt_in_wfa_module():
    assert not any(
        name.endswith("phase1b_real_wfa_certification")
        for name in sys.modules
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "unavailable"),
        ("2026-01-01T00:00:00", "timezone-aware"),
        ("not-a-timestamp", "invalid timestamp"),
    ],
)
def test_timestamp_comparison_rejects_invalid_authority(value, message):
    with pytest.raises(DatasetError, match=message):
        utc_timestamps_equal(value, "2026-01-01T00:00:00Z", field="boundary")


def test_timestamp_comparison_normalizes_aware_values_to_utc():
    assert utc_timestamps_equal(
        "2026-01-01T08:00:00+08:00",
        "2026-01-01T00:00:00Z",
        field="boundary",
    )
    assert not utc_timestamps_equal(
        "2026-01-01T08:00:01+08:00",
        "2026-01-01T00:00:00Z",
        field="boundary",
    )


def test_storage_containment_is_explicit_and_actionable(tmp_path):
    contained = tmp_path / "isolated" / "study.db"
    contained.parent.mkdir()
    contained.touch()
    assert assert_path_contained(contained, tmp_path, field="storage") == (
        contained.relative_to(tmp_path)
    )
    with pytest.raises(DatasetError, match="not contained"):
        assert_path_contained(REPO_ROOT, tmp_path, field="storage")


@pytest.mark.parametrize("protected", PROTECTED_CANONICAL_OUTPUTS)
@pytest.mark.parametrize("location", ["exact", "descendant"])
def test_certification_work_dir_rejects_both_canonical_outputs(protected, location):
    work_dir = protected if location == "exact" else protected / "certification" / "run"
    with pytest.raises(DatasetError, match="protected canonical output paths"):
        assert_certification_work_dir_allowed(work_dir)


def test_certification_work_dir_allows_unrelated_output_and_tmp_paths():
    allowed = (
        REPO_ROOT / "tools" / "strategy_lab" / "output" / "unrelated-smoke",
        REPO_ROOT / "tools" / "strategy_lab" / "tmp" / "tz04-follow-up",
    )
    assert [assert_certification_work_dir_allowed(path) for path in allowed] == [
        path.resolve() for path in allowed
    ]


def test_finite_mtm_group_gate_accepts_partial_availability_and_rejects_empty_column():
    group = np.full((480, 2, len(METRIC_AXIS)), np.nan, dtype=np.float64)
    group[7, 1, METRIC_AXIS.index("max_drawdown_mtm_pct")] = 2.5
    assert finite_mtm_group_facts(group, group_label="CRVUSDT/window_01.npy") == {
        "finite_mtm_count": 1,
        "mtm_value_count": 960,
    }
    group[7, 1, METRIC_AXIS.index("max_drawdown_mtm_pct")] = np.nan
    with pytest.raises(
        DatasetError,
        match=r"CRVUSDT/window_01.npy: generated MTM column has no finite values",
    ):
        finite_mtm_group_facts(group, group_label="CRVUSDT/window_01.npy")


def test_finite_mtm_group_gate_uses_the_loaded_plan_candidate_count():
    group = np.full((2400, 2, len(METRIC_AXIS)), np.nan, dtype=np.float64)
    group[2399, 0, METRIC_AXIS.index("max_drawdown_mtm_pct")] = 1.0
    assert finite_mtm_group_facts(
        group, group_label="SAR/window_01.npy", candidate_count=2400
    ) == {"finite_mtm_count": 1, "mtm_value_count": 4800}


def test_legacy_column_preservation_accepts_exact_equal_nan_values():
    rng = np.random.default_rng(41)
    schema_v1 = rng.normal(size=(480, 2, 20)).astype(np.float64)
    schema_v1[3, 1, 6] = np.nan
    schema_v2 = np.concatenate(
        [schema_v1.copy(), np.zeros((480, 2, 1), dtype=np.float64)], axis=2
    )
    facts = legacy_column_preservation_facts(
        schema_v2,
        schema_v1,
        schema_v2_label="v2/CRVUSDT/window_01.npy",
        schema_v1_label="v1/CRVUSDT/window_01.npy",
    )
    assert facts["mismatch_count"] == 0
    assert facts["compared_value_count"] == 480 * 2 * 20
    assert facts["bitwise_equal_with_equal_nan"] is True


@pytest.mark.parametrize(
    ("target", "replacement", "diagnostic"),
    [
        ("v2", np.zeros((479, 2, 21), dtype=np.float64), "v2-label"),
        ("v1", np.zeros((480, 2, 20), dtype=np.float32), "v1-label"),
    ],
)
def test_legacy_column_preservation_rejects_shape_or_dtype(
    target, replacement, diagnostic
):
    schema_v2 = np.zeros((480, 2, 21), dtype=np.float64)
    schema_v1 = np.zeros((480, 2, 20), dtype=np.float64)
    if target == "v2":
        schema_v2 = replacement
    else:
        schema_v1 = replacement
    with pytest.raises(DatasetError, match=diagnostic):
        legacy_column_preservation_facts(
            schema_v2,
            schema_v1,
            schema_v2_label="v2-label",
            schema_v1_label="v1-label",
        )


def test_legacy_column_preservation_reports_bounded_mismatches():
    schema_v1 = np.zeros((480, 2, 20), dtype=np.float64)
    schema_v2 = np.zeros((480, 2, 21), dtype=np.float64)
    schema_v2[: MISMATCH_SAMPLE_LIMIT + 2, 0, 0] = 1.0
    with pytest.raises(DatasetError) as raised:
        legacy_column_preservation_facts(
            schema_v2,
            schema_v1,
            schema_v2_label="v2/CRVUSDT/window_01.npy",
            schema_v1_label="v1/CRVUSDT/window_01.npy",
        )
    message = str(raised.value)
    assert "mismatch_count=7" in message
    assert "additional_mismatches_omitted=2" in message
    assert "candidate_id=1" in message
    assert "field=net_profit_pct" in message


def test_candidate_identity_maps_do_not_depend_on_projection_list_position():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    projection = project_candidates(spec.plan, spec.generation["strategy"])
    reordered = copy.deepcopy(projection)
    reordered["candidates"] = list(reversed(reordered["candidates"]))

    projected_by_id, plan_index_by_id = candidate_identity_mappings(
        spec.plan,
        reordered,
    )

    assert plan_index_by_id[1] == 0
    assert projected_by_id[1]["semantic_key"] == (
        spec.plan.candidate_table.semantic_key_for_index(0)
    )


def test_geometry_selection_reports_duplicate_ids_and_empty_axis():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    projection = project_candidates(spec.plan, spec.generation["strategy"])
    duplicate = copy.deepcopy(projection)
    duplicate["candidates"][1]["candidate_id"] = duplicate["candidates"][0][
        "candidate_id"
    ]
    with pytest.raises(DatasetError, match="duplicate candidate ID"):
        geometry_candidate_ids(duplicate)

    empty_axis = copy.deepcopy(projection)
    axis_name = empty_axis["global_axis_names"][0]
    for row in empty_axis["candidates"]:
        row["active_axis_mask"][0] = False
    with pytest.raises(DatasetError, match=rf"{axis_name!r}.*no active"):
        geometry_candidate_ids(empty_axis)


def test_compiled_reference_evidence_counters_are_derived():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    compiled = fake_execute(spec.plan)
    reference = SimpleNamespace(
        rows=tuple(
            replace(row, backend_kind="reference")
            for row in fake_rows(spec.plan)
        ),
        selected=(),
        metadata={"backend_kind": "reference"},
    )

    evidence = _compare_grid_runs(
        compiled,
        reference,
        spec.plan,
        expected_candidate_count=spec.plan.deduped_candidate_count,
    )

    assert evidence["identity_mismatch_count"] == 0
    assert evidence["availability_pattern_mismatch_count"] == 0
    assert evidence["exact_field_mismatch_count"] == 0
    assert evidence["floating_mismatch_count"] == 0
    assert evidence["mismatch_count"] == 0


def _parity_surfaces(plan, compiled_rows):
    compiled = fake_execute(plan)
    compiled.rows = tuple(compiled_rows)
    reference = SimpleNamespace(
        rows=tuple(
            replace(row, backend_kind="reference")
            for row in fake_rows(plan)
        ),
        selected=(),
        metadata={"backend_kind": "reference"},
    )
    return compiled, reference


@pytest.mark.parametrize(
    ("category", "counter_name", "field_name"),
    [
        ("identity", "identity_mismatch_count", "semantic_key"),
        (
            "availability_pattern",
            "availability_pattern_mismatch_count",
            "romad",
        ),
        ("exact_field", "exact_field_mismatch_count", "total_trades"),
        ("floating", "floating_mismatch_count", "net_profit_pct"),
    ],
)
def test_compiled_reference_negative_counters_and_diagnostics_are_derived(
    category,
    counter_name,
    field_name,
):
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    rows = list(fake_rows(spec.plan))
    if category == "identity":
        rows[0] = replace(rows[0], semantic_key=rows[0].semantic_key + "-changed")
    elif category == "availability_pattern":
        rows[0] = replace(rows[0], romad=0.5)
    elif category == "exact_field":
        rows[0] = replace(rows[0], total_trades=1)
    else:
        rows[0] = replace(rows[0], net_profit_pct=1.0)
    compiled, reference = _parity_surfaces(spec.plan, rows)

    evidence, samples = _grid_run_parity_facts(
        compiled,
        reference,
        spec.plan,
        expected_candidate_count=spec.plan.deduped_candidate_count,
    )

    assert evidence[counter_name] == 1
    assert evidence["mismatch_count"] == 1
    assert samples[0]["category"] == category
    assert samples[0]["candidate_id"] == 1
    diagnostic_name = (
        samples[0].get("identity_category")
        if category == "identity"
        else samples[0].get("field")
    )
    assert diagnostic_name == field_name
    with pytest.raises(DatasetError) as raised:
        _compare_grid_runs(
            compiled,
            reference,
            spec.plan,
            expected_candidate_count=spec.plan.deduped_candidate_count,
        )
    message = str(raised.value)
    assert f"{counter_name}=1" in message
    assert "candidate_id=1" in message
    assert field_name in message


def test_compiled_reference_mismatch_samples_are_bounded():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    rows = list(fake_rows(spec.plan))
    changed_count = MISMATCH_SAMPLE_LIMIT + 3
    for index in range(changed_count):
        rows[index] = replace(
            rows[index],
            net_profit_pct=rows[index].net_profit_pct + 1.0,
        )
    compiled, reference = _parity_surfaces(spec.plan, rows)

    evidence, samples = _grid_run_parity_facts(
        compiled,
        reference,
        spec.plan,
        expected_candidate_count=spec.plan.deduped_candidate_count,
    )

    assert evidence["floating_mismatch_count"] == changed_count
    assert len(samples) == MISMATCH_SAMPLE_LIMIT
    with pytest.raises(DatasetError) as raised:
        _compare_grid_runs(
            compiled,
            reference,
            spec.plan,
            expected_candidate_count=spec.plan.deduped_candidate_count,
        )
    assert "additional_mismatches_omitted=3" in str(raised.value)


def test_compiled_reference_rejects_small_finite_mtm_difference_as_exact():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    rows = list(fake_rows(spec.plan))
    rows[0] = replace(rows[0], max_drawdown_mtm_pct=1.0 + 1e-13)
    compiled, reference = _parity_surfaces(spec.plan, rows)
    reference.rows = (
        replace(reference.rows[0], max_drawdown_mtm_pct=1.0),
        *reference.rows[1:],
    )

    evidence, samples = _grid_run_parity_facts(
        compiled,
        reference,
        spec.plan,
        expected_candidate_count=spec.plan.deduped_candidate_count,
    )
    assert evidence["availability_pattern_mismatch_count"] == 0
    assert evidence["exact_field_mismatch_count"] == 1
    assert evidence["floating_mismatch_count"] == 0
    assert samples[0]["field"] == "max_drawdown_mtm_pct"
    with pytest.raises(DatasetError, match="exact_field_mismatch_count=1"):
        _compare_grid_runs(
            compiled,
            reference,
            spec.plan,
            expected_candidate_count=spec.plan.deduped_candidate_count,
        )


def test_compiled_reference_missing_row_diagnostic_names_position_and_id():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    compiled, reference = _parity_surfaces(spec.plan, fake_rows(spec.plan))
    reference.rows = reference.rows[:-1]

    with pytest.raises(DatasetError) as raised:
        _compare_grid_runs(
            compiled,
            reference,
            spec.plan,
            expected_candidate_count=spec.plan.deduped_candidate_count,
        )

    message = str(raised.value)
    assert "reference_missing_row" in message
    assert "candidate_id=480" in message
    assert "row_position=479" in message


def test_compiled_reference_row_order_diagnostic_names_position_and_both_ids():
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    compiled, reference = _parity_surfaces(spec.plan, fake_rows(spec.plan))
    reordered = list(reference.rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    reference.rows = tuple(reordered)

    with pytest.raises(DatasetError) as raised:
        _compare_grid_runs(
            compiled,
            reference,
            spec.plan,
            expected_candidate_count=spec.plan.deduped_candidate_count,
            segment_name="is",
        )

    message = str(raised.value)
    assert "identity_category=candidate_id" in message
    assert "candidate_id=1" in message
    assert "reference_candidate_id=2" in message
    assert "row_position=0" in message
    assert "segment=is" in message


def test_selected_slow_mismatch_reports_candidate_segment_and_metric(monkeypatch):
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    projection = project_candidates(spec.plan, spec.generation["strategy"])
    fast = fake_rows(spec.plan)[0]
    projected = projection["candidates"][0]
    slow_values = {
        name: (
            fast.guardrail_summary[name]
            if name in fast.guardrail_summary
            else getattr(fast, name)
        )
        for name in METRIC_AXIS
    }
    slow_values["net_profit_pct"] = float(fast.net_profit_pct) + 1.0

    counts, samples, _maximums = _selected_slow_row_mismatches(
        fast,
        slow_values,
        projected,
        candidate_id=1,
        segment_name="is",
    )
    assert counts["floating_mismatch_count"] == 1
    assert samples == [
        {
            "category": "floating",
            "candidate_id": 1,
            "segment": "is",
            "field": "net_profit_pct",
        }
    ]

    monkeypatch.setattr(certify_module, "run_v2_strategy", lambda **_kwargs: object())
    monkeypatch.setattr(certify_module, "_slow_values", lambda _run: slow_values)
    hooks = SimpleNamespace(build_execution_data=lambda _dataframe, _params: object())
    prepared = {
        name: SimpleNamespace(dataframe=object(), trade_start_idx=0)
        for name in SEGMENT_AXIS
    }
    references = {
        name: SimpleNamespace(rows=(fast,))
        for name in SEGMENT_AXIS
    }

    with pytest.raises(DatasetError) as raised:
        _selected_slow_parity(
            spec.plan,
            hooks,
            prepared,
            references,
            projection,
            (1,),
        )

    message = str(raised.value)
    assert "floating_mismatch_count=2" in message
    assert "candidate_id=1" in message
    assert "segment=is" in message
    assert "field=net_profit_pct" in message


def test_protected_snapshot_diagnostic_names_changed_paths():
    before = {"queue.json": ("a", 1, 2)}
    after = {"queue.json": ("b", 1, 2), "journal.db": ("c", 2, 3)}
    assert changed_snapshot_paths(before, after, label="storage") == [
        "storage:journal.db",
        "storage:queue.json",
    ]

"""Server grid settings contracts."""

import json
import uuid
from copy import deepcopy

import pytest

from ui.server import app
from core.storage import get_db_connection
from ui.server_services import build_grid_settings_view

from ._helpers import _grid_sidebar_config, _temporary_active_db


def test_study_endpoint_includes_single_grid_settings(client):
    preview = {
        "total_space": 20,
        "total_space_label": "20",
        "actual_budget": 10,
        "actual_budget_label": "10",
        "coverage_pct": 50.0,
        "coverage_label": "50.0%",
        "allocation_method": "auto_sqrt_space",
        "modes": [
            {
                "mode": "cc_only",
                "space_size": 8,
                "space_label": "8",
                "budget": 4,
                "budget_label": "4",
                "coverage_pct": 50.0,
                "coverage_label": "50.0%",
                "generation": "LHS",
            }
        ],
    }
    config = _grid_sidebar_config()
    summary = {
        "requested_budget": 10,
        "actual_budget": 10,
        "grid": {"preview": preview},
        "optimization_time_seconds": 26,
    }

    with _temporary_active_db(f"grid_settings_single_{uuid.uuid4().hex[:8]}"):
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO studies (
                    study_id,
                    study_name,
                    strategy_id,
                    optimization_mode,
                    optimizer_mode,
                    config_json,
                    grid_requested_budget,
                    grid_actual_budget,
                    grid_coverage_pct,
                    grid_top_candidates,
                    grid_summary_json,
                    optimization_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "grid_settings_single",
                    "GRID_SETTINGS_SINGLE",
                    "s03_reversal_v10",
                    "grid",
                    "grid",
                    json.dumps(config),
                    10,
                    10,
                    50.0,
                    5,
                    json.dumps(summary),
                    26,
                ),
            )
            conn.commit()

        response = client.get("/api/studies/grid_settings_single")
        assert response.status_code == 200
        study = response.get_json()["study"]
        grid = study["grid_settings"]
        rows = {row["key"]: row["val"] for row in grid["rows"]}
        allocation = {row["key"]: row["val"] for row in grid["allocation_rows"]}

        assert grid["enabled"] is True
        assert grid["is_wfa_grid"] is False
        assert rows["Budget"] == "10 candidates"
        assert rows["Parameter Space"] == "20 combinations"
        assert rows["Coverage"] == "50.0%"
        assert rows["Workers"] == "6 Numba threads"
        assert rows["Fast Objectives"] == "Net Profit %"
        assert rows["Fast Primary"] == "Net Profit %"
        assert rows["Slow Refinement"] == "Off"
        assert rows["Runtime"] == "26s"
        assert allocation["Allocation"] == "Auto sqrt-space"
        assert allocation["CC only"] == "4 / 8 | 50.0% | LHS"


def test_study_endpoint_isolates_only_unexpected_grid_enrichment_failure(
    client, monkeypatch
):
    from ui import server_routes_data

    logged = []
    monkeypatch.setattr(
        server_routes_data,
        "build_grid_settings_view",
        lambda _study: (_ for _ in ()).throw(RuntimeError("preview exploded")),
    )
    monkeypatch.setattr(app.logger, "exception", lambda *args: logged.append(args))

    with _temporary_active_db(f"grid_isolation_{uuid.uuid4().hex[:8]}"):
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO studies (
                    study_id, study_name, strategy_id, optimization_mode, config_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "grid_isolation",
                    "GRID_ISOLATION",
                    "s03_reversal_v10",
                    "grid",
                    json.dumps(_grid_sidebar_config()),
                ),
            )
            conn.commit()

        response = client.get("/api/studies/grid_isolation")
        assert response.status_code == 200
        study = response.get_json()["study"]
        assert study["study_id"] == "grid_isolation"
        assert "grid_settings" not in study
        assert len(logged) == 1
        assert logged[0][1:] == ("grid_isolation", "s03_reversal_v10")


_TWO_ENABLED_CONSTRAINTS = [
    {"metric": "total_trades", "threshold": 30, "enabled": True},
    {"metric": "max_drawdown_pct", "threshold": 30.0, "enabled": True},
    {"metric": "net_profit_pct", "threshold": 5.0, "enabled": False},
]


_EXPECTED_CONSTRAINTS_TEXT = "Total Trades >= 30, Min Drawdown % <= 30"


def _constraints_row(view):
    assert view is not None
    return next((row["val"] for row in view["rows"] if row["key"] == "Constraints"), None)


def _standalone_grid_study(**overrides):
    study = {
        "optimization_mode": "grid",
        "config_json": {
            "grid_budget": 10,
            "grid_seed": 42,
            "grid_top_candidates": 5,
            "grid_fast_objectives": ["net_profit_pct"],
        },
    }
    study.update(overrides)
    return study


def _saved_v2_full_enumeration_grid_summary():
    return {
        "grid": {
            "preview": {
                "full_candidate_count": 48_480,
            },
            "allocation": {
                "allocation_method": "full_enumeration_v2",
                "mode_space_sizes": {"bracket": 480, "trail": 48_000},
                "mode_budgets": {"bracket": 480, "trail": 48_000},
                "mode_coverage_pct": {"bracket": 100.0, "trail": 100.0},
            },
        },
    }


def _grid_settings_rows(view):
    assert view is not None
    return {row["key"]: row["val"] for row in view["rows"]}


def _grid_settings_allocation_rows(view):
    assert view is not None
    return {row["key"]: row["val"] for row in view["allocation_rows"]}


def test_grid_settings_derivation_memo_ignores_study_row_identity(monkeypatch):
    from ui import server_services

    calls = []

    def fake_derive(config, study):
        calls.append((deepcopy(config), deepcopy(study)))
        return {"candidate_count": 10, "modes": []}

    monkeypatch.setattr(server_services, "_derive_grid_preview", fake_derive)
    config = _grid_sidebar_config()
    first = {
        "study_id": "grid-one",
        "strategy_id": "s03_reversal_v10",
        "optimization_mode": "grid",
        "config_json": config,
    }
    second = deepcopy(first)
    second["study_id"] = "grid-two"
    memo = {}

    assert build_grid_settings_view(first, memo=memo)["available"] is True
    assert build_grid_settings_view(second, memo=memo)["available"] is True
    assert len(calls) == 1


def test_grid_settings_memo_caches_expected_failure_and_keeps_envelope_truthful(
    monkeypatch,
):
    from ui import server_services

    calls = []

    def fail_derive(_config, study):
        calls.append(study["study_id"])
        raise server_services._grid_settings_error(
            "stored_runtime_unavailable",
            study["strategy_id"],
            "config_json.v2_runtime",
            "Stored runtime is unavailable.",
        )

    monkeypatch.setattr(server_services, "_derive_grid_preview", fail_derive)
    base = {
        "study_id": "one",
        "strategy_id": "s06_r_trend_v02_b2",
        "optimization_mode": "grid",
        "config_json": _grid_sidebar_config(),
        "grid_summary": {"requested_budget": 10},
    }
    memo = {}
    first = build_grid_settings_view(base, memo=memo)
    second_study = deepcopy(base)
    second_study["study_id"] = "two"
    second = build_grid_settings_view(second_study, memo=memo)

    assert calls == ["one"]
    for view in (first, second):
        assert view["available"] is True
        assert view["unavailable_reason"] is None
        assert view["derivation_status"] == "unavailable"
        assert view["derivation_unavailable_reason"] == "stored_runtime_unavailable"
        assert view["diagnostics"][0]["code"] == "V2_STORED_RUNTIME_METADATA_INCOMPATIBLE"
    first["diagnostics"].clear()
    assert second["diagnostics"]


def test_grid_settings_without_memo_skips_memo_key_serialization(monkeypatch):
    from ui import server_services

    monkeypatch.setattr(
        server_services, "_derive_grid_preview", lambda _config, _study: {"modes": []}
    )
    monkeypatch.setattr(
        server_services,
        "_grid_settings_memo_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("memo key computed")
        ),
    )
    view = build_grid_settings_view(
        {
            "strategy_id": "s03_reversal_v10",
            "optimization_mode": "grid",
            "config_json": _grid_sidebar_config(),
        }
    )
    assert view["available"] is True
    assert view["derivation_status"] == "succeeded"


@pytest.mark.parametrize("noncanonical", [float("nan"), float("inf"), float("-inf")])
def test_grid_settings_noncanonical_memo_key_derives_without_caching(
    monkeypatch, noncanonical
):
    from ui import server_services

    calls = []
    monkeypatch.setattr(
        server_services,
        "_derive_grid_preview",
        lambda _config, _study: calls.append(True) or {"modes": []},
    )
    config = _grid_sidebar_config()
    config["noncanonicalMemoValue"] = noncanonical
    memo = {"sentinel": object()}

    study = {
        "strategy_id": "s03_reversal_v10",
        "optimization_mode": "grid",
        "config_json": config,
    }
    first_view = build_grid_settings_view(study, memo=memo)
    second_view = build_grid_settings_view(study, memo=memo)

    assert first_view["available"] is True
    assert first_view["derivation_status"] == "succeeded"
    assert second_view == first_view
    assert calls == [True, True]
    assert set(memo) == {"sentinel"}
    assert None not in memo


def test_grid_settings_unusable_derivation_sets_only_unavailable_reason(monkeypatch):
    from ui import server_services

    monkeypatch.setattr(
        server_services,
        "_derive_grid_preview",
        lambda _config, study: (_ for _ in ()).throw(
            server_services._grid_settings_error(
                "stored_runtime_unavailable",
                study["strategy_id"],
                "config_json.v2_runtime",
                "Stored runtime is unavailable.",
            )
        ),
    )
    view = build_grid_settings_view(
        {
            "strategy_id": "s06_r_trend_v02_b2",
            "optimization_mode": "grid",
            "config_json": _grid_sidebar_config(),
        }
    )
    assert view["available"] is False
    assert view["unavailable_reason"] == "stored_runtime_unavailable"
    assert view["derivation_unavailable_reason"] == "stored_runtime_unavailable"


def test_grid_settings_unknown_strategy_is_viewable_without_substitution():
    study = {
        "strategy_id": "removed_strategy",
        "optimization_mode": "grid",
        "config_json": _grid_sidebar_config(),
    }
    view = build_grid_settings_view(study)

    assert view["available"] is False
    assert view["unavailable_reason"] == "strategy_unavailable"
    assert view["diagnostics"][0]["strategy_id"] == "removed_strategy"


def _assert_saved_v2_full_enumeration_grid_settings(view):
    rows = _grid_settings_rows(view)
    allocation_rows = _grid_settings_allocation_rows(view)

    assert rows["Sampling"] == "Full enumeration"
    assert "Seed" not in rows
    assert rows["Parameter Space"] != "-"
    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert allocation_rows["Allocation"] == "Full enumeration"
    assert "Bracket" in allocation_rows
    assert "Trail" in allocation_rows


def test_grid_settings_saved_v2_full_enumeration_labels():
    study = _standalone_grid_study(grid_summary=_saved_v2_full_enumeration_grid_summary())

    view = build_grid_settings_view(study)

    assert view["is_wfa_grid"] is False
    _assert_saved_v2_full_enumeration_grid_settings(view)


def test_grid_settings_saved_wfa_v2_full_enumeration_labels():
    study = {
        "optimization_mode": "wfa",
        "optimizer_mode": "grid",
        "config_json": {"optimization_mode": "grid", "grid_budget": 10},
        "grid_summary": _saved_v2_full_enumeration_grid_summary(),
    }

    view = build_grid_settings_view(study)

    assert view["is_wfa_grid"] is True
    _assert_saved_v2_full_enumeration_grid_settings(view)


def test_grid_settings_s03_v1_keeps_mode_ma_period_diversity_wording():
    study = _standalone_grid_study(
        config_json={
            "strategy_id": "s03_reversal_v10",
            "grid_v2_planning_policy": "full",
            "grid_allocation_method": "auto_sqrt_space",
            "grid_budget": 10,
            "grid_seed": 42,
            "grid_diversity_enabled": True,
            "grid_diversity_max_per_group": 2,
        },
    )

    rows = _grid_settings_rows(build_grid_settings_view(study))

    assert rows["Diversity"] == "On, max 2 / Mode+MA+Period"


def test_grid_settings_effective_full_overrides_configured_automatic_allocation():
    study = _standalone_grid_study(
        config_json={
            "grid_v2_planning_policy": "sampled",
            "grid_allocation_method": "auto_sqrt_space",
            "grid_budget": 100_000,
        },
        grid_summary={
            "engine": "v2",
            "grid_v2_plan_fingerprint": "current-v2-plan",
            "grid": {
                "backend": {"engine": "v2", "profile": "full_enumeration_v2"},
                "planning": {
                    "requested_policy": "sampled",
                    "effective_policy": "full",
                    "effective_allocation_method": "full_enumeration_v2",
                    "planned_candidate_count": 30,
                    "requested_budget": 100_000,
                    "plan_identity_schema_version": "grid_v2_plan_identity_v2",
                    "planning_policy_version": "grid_v2_planning_policy_v2",
                },
                "preview": {
                    "full_candidate_count": 30,
                    "planned_candidate_count": 30,
                    "coverage_pct": 100.0,
                },
            },
        },
    )

    view = build_grid_settings_view(study)
    rows = _grid_settings_rows(view)
    allocation_rows = _grid_settings_allocation_rows(view)
    assert rows["Planning"].startswith("sampled")
    assert rows["Planning"].endswith("full")
    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert "Seed" not in rows
    assert allocation_rows["Allocation"] == "Full enumeration"


def test_grid_settings_sampled_v2_uses_generic_diversity_wording():
    study = _standalone_grid_study(
        config_json={
            "grid_v2_planning_policy": "sampled",
            "grid_allocation_method": "proportional_space",
            "grid_seed": 42,
            "grid_diversity_enabled": True,
            "grid_diversity_max_per_group": 2,
        },
        grid_summary={
            "grid": {
                "planning": {"requested_policy": "sampled", "effective_policy": "sampled"},
                "preview": {
                    "engine": "v2",
                    "profile": "full_enumeration_v2",
                    "full_candidate_count": 100,
                    "planned_candidate_count": 10,
                },
                "allocation": {"allocation_method": "proportional_space"},
            },
        },
    )

    rows = _grid_settings_rows(build_grid_settings_view(study))
    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert rows["Seed"] == "42"


@pytest.mark.parametrize(
    ("marker_section", "marker_key", "marker_value"),
    [
        ("preview", "engine", "v2"),
        ("summary", "engine", "v2"),
        ("backend", "engine", "v2"),
        ("summary", "grid_v2_plan_fingerprint", "v2-plan-fingerprint"),
        ("planning", "plan_identity_schema_version", "grid_v2_plan_identity_v2"),
        ("planning", "planning_policy_version", "grid_v2_planning_policy_v2"),
        ("preview", "profile", "full_enumeration_v2"),
        ("backend", "profile", "full_enumeration_v2"),
        ("allocation", "allocation_method", "full_enumeration_v2"),
    ],
)
def test_grid_settings_recognizes_authoritative_v2_markers(
    marker_section,
    marker_key,
    marker_value,
):
    grid_summary = {
        "grid": {
            "backend": {},
            "planning": {"requested_policy": "sampled", "effective_policy": "sampled"},
            "preview": {"full_candidate_count": 100, "planned_candidate_count": 10},
            "allocation": {"allocation_method": "proportional_space"},
        },
    }
    marker_target = grid_summary if marker_section == "summary" else grid_summary["grid"][marker_section]
    marker_target[marker_key] = marker_value
    study = _standalone_grid_study(grid_summary=grid_summary)

    rows = _grid_settings_rows(build_grid_settings_view(study))

    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert rows["Seed"] == "42"


def test_grid_settings_prefers_v2_planning_json_over_ambiguous_legacy_columns():
    study = _standalone_grid_study(
        grid_requested_budget=999,
        grid_actual_budget=888,
        grid_coverage_pct=88.8,
        grid_summary={
            "requested_budget": 100,
            "actual_budget": 50,
            "grid": {
                "planning": {
                    "requested_policy": "sampled",
                    "effective_policy": "sampled",
                    "requested_budget": 100,
                    "planned_candidate_count": 50,
                },
                "preview": {
                    "full_candidate_count": 1_000,
                    "planned_candidate_count": 50,
                    "coverage_pct": 5.0,
                    "requested_planning_policy": "sampled",
                    "effective_planning_policy": "sampled",
                },
            },
        },
    )

    rows = _grid_settings_rows(build_grid_settings_view(study))
    assert rows["Planning"] == "sampled"
    assert rows["Budget"] == "50 candidates"
    assert rows["Parameter Space"] == "1k combinations"
    assert rows["Coverage"] == "5.0%"


def test_grid_settings_constraints_standalone_two_enabled():
    study = _standalone_grid_study(constraints=list(_TWO_ENABLED_CONSTRAINTS))
    assert _constraints_row(build_grid_settings_view(study)) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_wfa_grid_two_enabled():
    study = {
        "optimization_mode": "wfa",
        "optimizer_mode": "grid",
        "config_json": {"optimization_mode": "grid", "grid_budget": 10},
        "constraints_json": json.dumps(_TWO_ENABLED_CONSTRAINTS),
    }
    view = build_grid_settings_view(study)
    assert view["is_wfa_grid"] is True
    assert _constraints_row(view) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_none_when_absent():
    assert _constraints_row(build_grid_settings_view(_standalone_grid_study())) == "None"


def test_grid_settings_constraints_excludes_disabled():
    disabled = [
        {"metric": "total_trades", "threshold": 30, "enabled": False},
        {"metric": "max_drawdown_pct", "threshold": 30.0, "enabled": False},
    ]
    study = _standalone_grid_study(constraints=disabled)
    assert _constraints_row(build_grid_settings_view(study)) == "None"


def test_grid_settings_constraints_from_constraints_json_string():
    # Analytics passes the raw constraints_json column (a JSON string).
    study = _standalone_grid_study(constraints_json=json.dumps(_TWO_ENABLED_CONSTRAINTS))
    assert _constraints_row(build_grid_settings_view(study)) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_config_fallback_shapes():
    top_level = _standalone_grid_study()
    top_level["config_json"]["constraints"] = list(_TWO_ENABLED_CONSTRAINTS)
    assert _constraints_row(build_grid_settings_view(top_level)) == _EXPECTED_CONSTRAINTS_TEXT

    nested = _standalone_grid_study()
    nested["config_json"]["optuna_config"] = {"constraints": list(_TWO_ENABLED_CONSTRAINTS)}
    assert _constraints_row(build_grid_settings_view(nested)) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_results_and_analytics_row_identical():
    # Results loads constraints into a parsed list; Analytics keeps the raw JSON
    # string column.  Both must yield an identical Constraints row.
    results_shape = _standalone_grid_study(
        constraints=list(_TWO_ENABLED_CONSTRAINTS),
        constraints_json=list(_TWO_ENABLED_CONSTRAINTS),
    )
    analytics_shape = _standalone_grid_study(
        constraints_json=json.dumps(_TWO_ENABLED_CONSTRAINTS),
    )
    assert _constraints_row(build_grid_settings_view(results_shape)) == _constraints_row(
        build_grid_settings_view(analytics_shape)
    )

"""Server grid contracts."""

from strategies import get_strategy_config

from ._helpers import _s03_regime_er_grid_preview_payload, _v2_runtime_diagnostic


def test_grid_availability_reason_uses_generic_backend_label(client):
    response = client.get("/api/strategy/s01_trailing_ma/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["grid_optimizer"]["reason"] == (
        "No fast Grid backend is available for this strategy."
    )


def test_grid_preview_strategy_aliases_agree_and_conflict(client):
    payload = {
        "config": _s03_regime_er_grid_preview_payload(
            strategy_id="s03_reversal_v11_regime_er_b2"
        ),
        "strategyId": "s03_reversal_v11_regime_er_b2",
    }
    agreed = client.post("/api/grid/preview", json=payload)
    assert agreed.status_code == 200

    payload["strategyId"] = "s03_reversal_v10"
    conflict = client.post("/api/grid/preview", json=payload)
    assert conflict.status_code == 400
    diagnostic = _v2_runtime_diagnostic(conflict)
    assert diagnostic["code"] == "V2_CONFLICTING_STRATEGY_ID"
    assert diagnostic["path"] == "strategy_id"


def test_s03_regime_er_grid_metadata_hides_internal_variants(client):
    response = client.get("/api/strategy/s03_reversal_v11_regime_er_b2/config")

    assert response.status_code == 200
    grid_optimizer = response.get_json()["grid_optimizer"]
    assert grid_optimizer["profile"] == "full_enumeration_v2"
    assert grid_optimizer["modes"] == []
    assert grid_optimizer["diversity_group_fields"] == ["grid_mode_name"]


def test_s03_regime_er_grid_preview_uses_logical_modes_not_internal_variants(client):
    response = client.post("/api/grid/preview", json=_s03_regime_er_grid_preview_payload())

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    modes = {row["mode"]: row for row in preview["modes"]}

    assert list(modes) == ["cc_only", "tbands_only", "both"]
    assert [row["label"] for row in preview["modes"]] == ["Close Count only", "T Bands only", "Both"]
    assert "plain" not in modes
    assert "emergency" not in modes
    assert preview["mode_space_sizes"] == {"cc_only": 7_200, "tbands_only": 20_000, "both": 720_000}
    assert preview["full_candidate_count"] == 747_200
    assert preview["allocation_method"] == "full_enumeration_v2"


def test_s03_regime_er_budgeted_grid_preview_uses_generic_logical_block_allocation(client):
    response = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=1_000,
            grid_allocation_method="manual",
            grid_manual_percents={"cc_only": 20, "tbands_only": 30, "both": 50},
        ),
    )

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    assert preview["requested_planning_policy"] == "sampled"
    assert preview["effective_planning_policy"] == "sampled"
    assert preview["planned_candidate_count"] == 1_000
    assert preview["allocation_method"] == "manual"
    assert preview["mode_budgets"] == {"cc_only": 200, "tbands_only": 300, "both": 500}
    assert [row["mode"] for row in preview["modes"]] == ["cc_only", "tbands_only", "both"]


def test_s03_regime_er_manual_zero_percentages_round_trip_through_preview_api(client):
    response = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=1,
            grid_allocation_method="manual",
            grid_manual_percents={"cc_only": 0, "tbands_only": 0, "both": 100},
        ),
    )

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    assert preview["allocation_method"] == "manual"
    assert preview["planned_candidate_count"] == 1
    assert preview["mode_budgets"] == {"cc_only": 0, "tbands_only": 0, "both": 1}


def test_s03_regime_er_budgeted_grid_preview_rejects_unsafe_budget_and_unknown_manual_block(client):
    too_small = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=2,
        ),
    )
    unknown = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=100,
            grid_allocation_method="manual",
            grid_manual_percents={"cc_only": 20, "tbands_only": 30, "both": 40, "ghost": 10},
        ),
    )

    assert too_small.status_code == 400
    assert "non-empty planning blocks" in too_small.get_json()["error"]
    assert unknown.status_code == 400
    assert "ghost" in unknown.get_json()["error"]


def test_s03_regime_er_grid_preview_uses_posted_numeric_ranges(client):
    base_response = client.post(
        "/api/grid/preview",
        json={
            "config": _s03_regime_er_grid_preview_payload(),
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": 1000,
        },
    )
    ma_length_response = client.post(
        "/api/grid/preview",
        json={
            "config": _s03_regime_er_grid_preview_payload(
                param_ranges={"maLength3": [25, 250, 25]},
            ),
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": 1000,
        },
    )
    close_count_response = client.post(
        "/api/grid/preview",
        json={
            "config": _s03_regime_er_grid_preview_payload(
                param_ranges={"closeCountLong": [2, 7, 2]},
            ),
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": 1000,
        },
    )

    assert base_response.status_code == 200
    assert ma_length_response.status_code == 200
    assert close_count_response.status_code == 200

    base_preview = base_response.get_json()["preview"]
    ma_preview = ma_length_response.get_json()["preview"]
    close_preview = close_count_response.get_json()["preview"]

    assert base_preview["full_candidate_count"] == 747_200
    assert ma_preview["mode_space_sizes"] == {"cc_only": 3_600, "tbands_only": 10_000, "both": 360_000}
    assert ma_preview["full_candidate_count"] == 373_600
    assert close_preview["mode_space_sizes"] == {"cc_only": 3_600, "tbands_only": 20_000, "both": 360_000}
    assert close_preview["full_candidate_count"] == 383_600


def test_s03_regime_er_grid_enabled_modes_must_be_empty_for_internal_variants(client):
    from ui import server as server_module

    payload = _s03_regime_er_grid_preview_payload()
    config = server_module._build_optimization_config(
        "grid-preview.csv",
        payload,
        worker_processes=1,
        strategy_id="s03_reversal_v11_regime_er_b2",
        warmup_bars=1000,
    )
    explicit_empty = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(grid_enabled_modes=[]),
    )
    stale_modes = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(grid_enabled_modes=["plain"]),
    )

    assert config.grid_enabled_modes == []
    assert explicit_empty.status_code == 200
    assert stale_modes.status_code == 400
    assert "internal variant selector" in stale_modes.get_json()["error"]


def test_s03_regime_er_optimizer_defaults_are_user_facing_grid_defaults():
    config = get_strategy_config("s03_reversal_v11_regime_er_b2")
    params = config["parameters"]
    default_enabled = {
        name
        for name, spec in params.items()
        if spec.get("optimize", {}).get("enabled")
        and spec.get("optimize", {}).get("default_enabled", True) is not False
    }

    assert {
        "maType3",
        "maLength3",
        "useCloseCount",
        "closeCountLong",
        "closeCountShort",
        "useTBands",
        "tBandLongPct",
        "tBandShortPct",
        "regimeErLength",
        "regimeErThresh",
    } <= default_enabled
    assert "maOffset3" not in default_enabled
    assert "emergencySlPct" not in default_enabled
    assert params["useEmergencySL"]["optimize"] == {"enabled": False}
    assert params["useRegime"]["optimize"] == {"enabled": False}


def test_s06_b2_grid_preview_keeps_user_facing_modes(client):
    response = client.post(
        "/api/grid/preview",
        json={
            "strategy_id": "s06_r_trend_v02_b2",
            "optimization_mode": "grid",
            "enabled_params": {},
            "param_ranges": {},
            "param_types": {},
            "fixed_params": {},
            "objectives": ["net_profit_pct"],
            "grid_fast_objectives": ["net_profit_pct"],
            "grid_budget": "200k",
            "grid_top_candidates": 5,
        },
    )

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    assert [row["mode"] for row in preview["modes"]] == ["bracket", "trail"]
    assert preview["mode_space_sizes"] == {"bracket": 480, "trail": 48_000}

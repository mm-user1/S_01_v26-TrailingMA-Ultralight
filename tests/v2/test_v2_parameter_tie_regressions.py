"""Independent inactive-axis and automatic-eligibility tie regressions."""
from dataclasses import replace
import itertools

import pytest

from core.grid_v2 import GridV2Settings, build_grid_v2_plan, preview_grid_v2_counts
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy


@pytest.mark.parametrize("ties,inactive,raw,unique", [
    ((), False, 49, 49), ((), True, 98, 49),
    (("symmetricLongShort",), False, 7, 7), (("symmetricLongShort",), True, 14, 7),
])
def test_retained_inactive_axis_enumeration(ties, inactive, raw, unique):
    config = strategy.load_config()
    settings = GridV2Settings(
        enabled_axes=("closeCountLong", "closeCountShort", "emergencySlPct"),
        enabled_tie_groups=ties, include_inactive_axes_for_dedup=inactive,
    )
    plan = build_grid_v2_plan(config, settings, {"useEmergencySL": False})
    preview = preview_grid_v2_counts(config, settings, {"useEmergencySL": False})
    assert plan.raw_candidate_count == plan.enumerated_candidate_count == raw
    assert plan.candidate_table.enumerated_candidate_count == raw
    assert plan.deduped_candidate_count == unique
    assert preview.raw_candidate_count == preview.enumerated_candidate_count == raw
    assert ("emergencySlPct" in preview.axis_names_by_variant["plain"]) is inactive
    assert preview.deduped_candidate_count == (None if inactive else unique)
    if inactive:
        with pytest.raises(ValueError, match="include_inactive_axes_for_dedup"):
            build_grid_v2_plan(config, replace(settings, planning_policy="sampled", requested_budget=2))


def test_reversed_pair_with_interleaved_inactive_axis_preserves_order():
    config = strategy.load_config()
    config["optimization_rules"]["parameter_tie_groups"][0]["pairs"][0].reverse()
    names = list(config["parameters"])
    for name in ("emergencySlPct", "tBandLongPct"):
        names.remove(name)
        names.insert(names.index("closeCountShort"), name)
    config["parameters"] = {name: config["parameters"][name] for name in names}
    for name in ("closeCountLong", "closeCountShort", "tBandLongPct", "tBandShortPct"):
        opt = config["parameters"][name]["optimize"]
        opt["max"] = opt["min"] + opt["step"]
    axes = ("closeCountLong", "emergencySlPct", "tBandLongPct", "closeCountShort", "tBandShortPct")
    settings = GridV2Settings(enabled_axes=axes, include_inactive_axes_for_dedup=True)
    untied = build_grid_v2_plan(config, settings)
    tied = build_grid_v2_plan(config, replace(settings, enabled_tie_groups=("symmetricLongShort",)))
    # Independent Cartesian enumeration retains the interleaved inactive dimension.
    filtered = []
    for values in itertools.product(*(untied.parameter_domains[name].values for name in axes)):
        row = dict(zip(axes, values))
        if row["closeCountLong"] == row["closeCountShort"] and row["tBandLongPct"] == row["tBandShortPct"]:
            filtered.append(row)
    assert tied.enumerated_candidate_count == len(filtered) == 8
    expected = list(dict.fromkeys((p["closeCountLong"], p["tBandLongPct"]) for p in filtered))
    assert [(c.params["closeCountLong"], c.params["tBandLongPct"]) for c in tied.candidates] == expected
    filtered_keys = [c.semantic_key for c in untied.candidates
                     if c.params["closeCountLong"] == c.params["closeCountShort"]
                     and c.params["tBandLongPct"] == c.params["tBandShortPct"]]
    assert [c.semantic_key for c in tied.candidates] == filtered_keys


@pytest.mark.parametrize("ties,count", [((), 196000), (("symmetricLongShort",), 2800)])
def test_auto_axes_ignore_selector_and_runtime_but_explicit_requests_fail(ties, count):
    config = strategy.load_config()
    config["parameters"]["useEmergencySL"]["optimize"]["enabled"] = True
    settings = GridV2Settings(enabled_tie_groups=ties)
    preview = preview_grid_v2_counts(config, settings)
    assert preview.raw_candidate_count == count
    assert "useEmergencySL" not in preview.axis_names_by_variant["plain"]
    for name in ("useEmergencySL", "warmupBars", "dateFilter", "start", "end"):
        with pytest.raises(ValueError, match="runtime|optimized non-runtime"):
            preview_grid_v2_counts(config, replace(settings, enabled_axes=(name,)))
    fixed = build_grid_v2_plan(config, replace(settings, enabled_axes=()))
    assert fixed.deduped_candidate_count == 1

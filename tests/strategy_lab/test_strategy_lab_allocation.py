from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from tools.strategy_lab.analysis.allocation import (
    DatasetInput,
    SelectedISTickerView,
    TickerScorer,
    _portfolio,
    _turnover,
    evaluate_allocation,
    random_percentile_fraction,
)
from tools.strategy_lab.analysis.cli import main as analysis_main
from tools.strategy_lab.analysis.allocation_certify import certify
from tools.strategy_lab.analysis.dataset import AnalysisError, open_dataset
from tools.strategy_lab.analysis.json_utils import canonical_json_bytes
from tools.strategy_lab.analysis.output import (
    ALLOCATION_OUTPUT_FILES,
    OUTPUT_FILES,
    render_files,
    write_allocation,
)
from test_strategy_lab_analysis import _sha, _synthetic_dataset, _write_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _fresh_python(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _assert_clean_process(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "Discovered " not in completed.stdout
    assert "strategy(ies)" not in completed.stdout


def _published_snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    return {
        path.name: (_sha(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def _input(root: Path, label: str = "canonical") -> DatasetInput:
    dataset = open_dataset(root)
    return DatasetInput(label, dataset, dataset.resolve_scope())


def _mutate_groups(root: Path, mutate) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["groups"]:
        path = root / record["path"]
        matrix = np.load(path)
        ticker = Path(record["path"]).parent.name
        window_id = int(Path(record["path"]).stem.split("_")[-1])
        mutate(matrix, ticker, window_id, manifest["identity"]["metric_axis"])
        np.save(path, matrix)
        record.update(sha256=_sha(path), size=path.stat().st_size)
    _write_json(manifest_path, manifest)


def _change_calendar(root: Path, *, id_offset: int = 0, day_offset: int = 0) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for window in manifest["identity"]["windows"]:
        window["window_id"] += id_offset
        if day_offset:
            window["oos_start"] = window["oos_start"].replace("-01T", f"-{1 + day_offset:02d}T")
    if id_offset:
        for record in manifest["groups"]:
            old = root / record["path"]
            old_id = int(old.stem.split("_")[-1])
            new = old.with_name(f"window_{old_id + id_offset:02d}.npy")
            old.rename(new)
            record["path"] = new.relative_to(root).as_posix()
    _write_json(manifest_path, manifest)
    if id_offset:
        runspec_path = root / "normalized_runspec.json"
        runspec = json.loads(runspec_path.read_text(encoding="utf-8"))
        for scope in runspec["preregistration"]["analysis_scopes"]:
            scope["window_numbers"] = [value + id_offset for value in scope["window_numbers"]]
        _write_json(runspec_path, runspec)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["identity"]["pre_registration_sha256"] = hashlib.sha256(
            canonical_json_bytes(runspec)
        ).hexdigest()
        artifact = manifest["artifacts"]["normalized_runspec.json"]
        artifact.update(sha256=_sha(runspec_path), size=runspec_path.stat().st_size)
        _write_json(manifest_path, manifest)


def _distinguish(root: Path, marker: str) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["test_dataset_identity"] = marker
    _write_json(path, manifest)


@pytest.mark.parametrize(
    ("arguments", "expected_prefix"),
    [
        (("-c", "import tools.strategy_lab.analysis"), ""),
        (("-m", "tools.strategy_lab.analysis.cli", "--help"), "usage:"),
        (("-m", "tools.strategy_lab.analysis.allocation_certify", "--help"), "usage:"),
    ],
)
def test_fresh_analysis_imports_and_help_have_clean_streams(arguments, expected_prefix):
    completed = _fresh_python(*arguments)
    _assert_clean_process(completed)
    if expected_prefix:
        assert completed.stdout.startswith(expected_prefix)
    else:
        assert completed.stdout == ""


def test_fresh_analysis_import_graph_excludes_production_discovery_and_grid():
    completed = _fresh_python(
        "-c",
        (
            "import sys; import tools.strategy_lab.analysis; "
            "assert 'tools.strategy_lab.' + 'config' not in sys.modules; "
            "assert not any(name == 'strategies' or name.startswith('strategies.') "
            "for name in sys.modules); "
            "assert not any('grid_v2' in name or 'grid_planning' in name "
            "for name in sys.modules)"
        ),
    )
    _assert_clean_process(completed)
    assert completed.stdout == ""


@pytest.mark.parametrize("command", ["analyze", "allocate"])
def test_fresh_cli_json_is_clean_and_second_run_is_immutable_noop(tmp_path, command):
    dataset = _synthetic_dataset(tmp_path / "input")
    output = tmp_path / command
    if command == "analyze":
        arguments = (
            "-m", "tools.strategy_lab.analysis.cli", "analyze",
            "--dataset", str(dataset), "--scope", "development",
            "--output", str(output),
        )
        expected_files = set(OUTPUT_FILES)
    else:
        arguments = (
            "-m", "tools.strategy_lab.analysis.cli", "allocate",
            "--dataset", f"canonical={dataset}", "--scope", "development",
            "--rule", "primary_profit", "--output", str(output),
        )
        expected_files = set(ALLOCATION_OUTPUT_FILES)

    first = _fresh_python(*arguments)
    _assert_clean_process(first)
    assert json.loads(first.stdout)["status"] == "published"
    before = _published_snapshot(output)
    assert set(before) == expected_files

    second = _fresh_python(*arguments)
    _assert_clean_process(second)
    assert json.loads(second.stdout)["status"] == "verified_noop"
    assert _published_snapshot(output) == before


def test_analysis_canonical_json_matches_frozen_bytes_and_identities():
    nested = {"z": 1, "a": {"β": "Привет", "x": [True, 3, 1.25, None]}}
    permuted = {"a": {"x": [True, 3, 1.25, None], "β": "Привет"}, "z": 1}
    expected_nested = '{"a":{"x":[true,3,1.25,null],"β":"Привет"},"z":1}'.encode()
    assert canonical_json_bytes(nested) == canonical_json_bytes(permuted) == expected_nested
    assert hashlib.sha256(expected_nested).hexdigest() == (
        "2b91e21596671405ad8517a4fa32a35f9e7efe345c1dc796f337b3e30d980697"
    )

    random_payload = {
        "schema": "strategy_lab_random_k_v1",
        "base_seed": 17,
        "dataset_manifest_sha256": (
            "a96167ade4e0907219aa3035994961a7b80c088069dbcd1d794a60b6959fafd1"
        ),
        "dataset_label": "canonical",
        "candidate_rule": "primary_profit",
        "ticker_scorer": {
            "name": "selected_is_net_profit",
            "version": "strategy_lab_ticker_score_v1",
            "configuration": {},
        },
        "allocation_kind": "random_k",
        "k": 6,
        "oos_start_utc": "2024-03-01T00:00:00Z",
        "oos_end_utc": "2024-03-31T23:30:00Z",
    }
    assert hashlib.sha256(canonical_json_bytes(random_payload)).hexdigest() == (
        "25461ae598da3c6cab48a2a75879d8abd06a1783c14d0324c14dc5284b3ddaa7"
    )

    scorer_identity = {
        "name": "custom_β",
        "version": "v1",
        "configuration": {"threshold": 1.5, "enabled": True, "items": [2, None]},
        "exploratory": True,
    }
    assert hashlib.sha256(canonical_json_bytes(scorer_identity)).hexdigest() == (
        "f113e3d806d26a44456a2161c8adb1934c8ea48f36be740f2f2cb925a742baa2"
    )
    for value, expected in (
        (None, b"null"),
        (True, b"true"),
        ("text", b'"text"'),
        ([1, False, None], b"[1,false,null]"),
    ):
        assert canonical_json_bytes(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "Out of range float values"),
        (float("inf"), "Out of range float values"),
        (object(), "Object of type object is not JSON serializable"),
    ],
)
def test_analysis_canonical_json_rejects_noncanonical_values(value, message):
    with pytest.raises(AnalysisError, match=f"canonical JSON: {message}"):
        canonical_json_bytes(value)


def test_two_level_freeze_uses_scalar_is_view_before_oos_and_not_analysis_result(
    tmp_path, monkeypatch
):
    import tools.strategy_lab.analysis.evaluate as evaluate_module

    item = _input(_synthetic_dataset(tmp_path))
    events = []
    original_oos = item.dataset.load_oos_window

    def oos(scope, window):
        events.append(("oos", window.window_id))
        return original_oos(scope, window)

    monkeypatch.setattr(item.dataset, "load_oos_window", oos)
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_scope",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not be used")),
    )

    def score(view, context):
        assert isinstance(view, SelectedISTickerView)
        assert all(not isinstance(getattr(view, field.name), np.ndarray) for field in fields(view))
        assert not hasattr(view, "oos") and not hasattr(view, "dataset")
        assert set(context) == {"dataset_label", "candidate_rule", "ticker_scorer"}
        assert not any(event[0] == "oos" and event[1] == view.window_id for event in events)
        events.append(("score", view.window_id))
        return view.is_net_profit_pct

    scorer = TickerScorer("proof", "v1", {"mode": "scalar"}, score)
    result = evaluate_allocation([item], candidate_rule="primary_profit", ticker_scorer=scorer)
    for window_id in (1, 2):
        assert max(index for index, event in enumerate(events) if event == ("score", window_id)) < events.index(("oos", window_id))
    assert result.run_metadata["ticker_scorer"]["exploratory"] is True
    assert len(result.run_metadata["ticker_scorer"]["identity_sha256"]) == 64


def test_candidate_rule_selection_is_distinct_from_official_ticker_score(tmp_path):
    result = evaluate_allocation(
        [_input(_synthetic_dataset(tmp_path))],
        candidate_rule="trade_gate15_profit_factor",
        primary_k=1,
    )
    row = next(row for row in result.pair_decisions if row["ticker"] == "AAA")
    assert row["candidate_id"] == 20
    assert row["candidate_rule_score"] == 2.0
    assert row["ticker_score"] == row["is_net_profit_pct"] == 3.0


@pytest.mark.parametrize(
    ("scores", "trades", "expected", "level"),
    [
        ({"AAA": 1.0, "BBB": 2.0}, None, "BBB", "ticker_score"),
        ({"AAA": 1.0, "BBB": 1.0}, {"AAA": 20.0, "BBB": 30.0}, "BBB", "is_total_trades"),
        ({"AAA": 1.0, "BBB": 1.0}, {"AAA": 20.0, "BBB": 20.0}, "AAA", "canonical_ticker"),
        ({"AAA": 1.0, "BBB": 1.0}, {"AAA": 20.0, "BBB": np.nan}, "AAA", "is_total_trades"),
    ],
)
def test_ranking_tie_levels_nonfinite_trades_and_order_invariance(
    tmp_path, scores, trades, expected, level
):
    tickers = (("BBB", "dev"), ("AAA", "dev"), ("HHH", "holdout"))
    root = _synthetic_dataset(tmp_path, tickers=tickers)
    if trades is not None:
        def mutate(matrix, ticker, window_id, axis):
            del window_id
            matrix[1, 0, axis.index("total_trades")] = trades[ticker] if ticker in trades else 20.0
        _mutate_groups(root, mutate)
    scorer = TickerScorer("tie", "v1", {}, lambda view, context: scores[view.ticker])
    result = evaluate_allocation([_input(root)], candidate_rule="primary_profit", primary_k=1, ticker_scorer=scorer)
    block = result.summary["datasets"]["canonical"]["blocks"][0]["variants"]["primary"]
    assert block["selected_tickers"] == [expected]
    assert block["boundary_tie_break_level"] == level


def test_k_capacity_underfill_fractions_negative_zero_and_matched_fraction(tmp_path):
    root = _synthetic_dataset(tmp_path, declared_dev_pool=10)
    scorer = TickerScorer(
        "signed", "v1", {}, lambda view, context: {"AAA": 0.0, "BBB": -1.0}[view.ticker]
    )
    result = evaluate_allocation(
        [_input(root)], candidate_rule="primary_profit", primary_k=6, sensitivity_k=8,
        ticker_scorer=scorer,
    )
    block = result.summary["datasets"]["canonical"]["blocks"][0]
    primary = block["variants"]["primary"]
    assert primary["selected_count"] == 2 and primary["cash_slots"] == 4
    assert primary["requested_capacity_fraction"] == 3.0
    assert primary["realized_selected_fraction"] == primary["selectivity"] == 1.0
    expected = sum(row["oos_return_pct"] for row in result.pair_decisions if row["window_id"] == 1) / 6
    assert primary["capacity_return_pct"] == expected
    assert block["variants"]["matched_fraction"]["k"] == 2
    assert block["variants"]["matched_fraction"]["label"] == "diagnostic"


@pytest.mark.parametrize("bad", [0, -1, True])
def test_invalid_declared_pool_and_exact_positive_k(tmp_path, bad):
    root = _synthetic_dataset(tmp_path, declared_dev_pool=bad)
    with pytest.raises(AnalysisError, match="declared development pool"):
        evaluate_allocation([_input(root)], candidate_rule="primary_profit")
    good = _input(_synthetic_dataset(tmp_path / "good"))
    with pytest.raises(AnalysisError, match="primary_k"):
        evaluate_allocation([good], candidate_rule="primary_profit", primary_k=bad)


def test_zero_available_and_absent_candidate_preserve_unavailable_semantics(tmp_path):
    root = _synthetic_dataset(tmp_path)
    scorer = TickerScorer("none", "v1", {}, lambda view, context: None)
    result = evaluate_allocation([_input(root)], candidate_rule="primary_profit", ticker_scorer=scorer)
    primary = result.summary["datasets"]["canonical"]["blocks"][0]["variants"]["primary"]
    assert primary["status"] == "unavailable"
    assert primary["requested_capacity_fraction"] is None
    assert primary["realized_selected_fraction"] is None
    assert primary["selectivity"] is None
    matched = result.summary["datasets"]["canonical"]["blocks"][0]["variants"]["matched_fraction"]
    assert matched["k"] is None and matched["cash_slots"] is None
    assert all(row["candidate_id"] == 20 and row["oos_return_pct"] is None for row in result.pair_decisions)

    def no_candidates(matrix, ticker, window_id, axis):
        del ticker, window_id
        matrix[:, 0, axis.index("net_profit_pct")] = np.nan

    root2 = _synthetic_dataset(tmp_path / "absent")
    _mutate_groups(root2, no_candidates)
    absent = evaluate_allocation([_input(root2)], candidate_rule="primary_profit")
    assert all(row["candidate_id"] is None for row in absent.pair_decisions)


@pytest.mark.parametrize("value", [True, "bad", np.array([1.0]), {}, np.nan, np.inf])
def test_custom_scorer_rejects_nonfinite_and_nonscalar_values(tmp_path, value):
    item = _input(_synthetic_dataset(tmp_path))
    scorer = TickerScorer("bad", "v1", {}, lambda view, context: value)
    with pytest.raises(AnalysisError, match="non-scalar|non-finite"):
        evaluate_allocation([item], candidate_rule="primary_profit", ticker_scorer=scorer)


def test_custom_scorer_wraps_exceptions_and_requires_json_metadata(tmp_path):
    item = _input(_synthetic_dataset(tmp_path))
    scorer = TickerScorer("explode", "v1", {}, lambda view, context: 1 / 0)
    with pytest.raises(AnalysisError, match="explode.*AAA.*block"):
        evaluate_allocation([item], candidate_rule="primary_profit", ticker_scorer=scorer)
    invalid = TickerScorer("bad-config", "v1", {"x": {1}}, lambda view, context: 1.0)
    with pytest.raises(AnalysisError, match="strict JSON"):
        evaluate_allocation([item], candidate_rule="primary_profit", ticker_scorer=invalid)
    non_exploratory = TickerScorer(
        "custom", "v1", {}, lambda view, context: 1.0, exploratory=False
    )
    with pytest.raises(AnalysisError, match="exploratory=true"):
        evaluate_allocation(
            [item], candidate_rule="primary_profit", ticker_scorer=non_exploratory
        )


def test_complete_k_controls_match_independent_oracle_and_random_contract(tmp_path):
    tickers = tuple((f"T{index:02d}", "dev") for index in range(10)) + (("HHH", "holdout"),)
    root = _synthetic_dataset(tmp_path, tickers=tickers, declared_dev_pool=10)
    result = evaluate_allocation([_input(root)], candidate_rule="primary_profit")
    block = result.summary["datasets"]["canonical"]["blocks"][0]
    pair = {row["ticker"]: row for row in result.pair_decisions if row["window_id"] == 1}
    ranked = sorted(pair, key=lambda ticker: (-pair[ticker]["ticker_score"], ticker))
    for variant, k in (("primary", 6), ("sensitivity", 8), ("matched_fraction", 6)):
        facts = block["variants"][variant]
        assert facts["k"] == k
        assert facts["capacity_return_pct"] == pytest.approx(sum(pair[t]["oos_return_pct"] for t in ranked[:k]) / k)
        bottom = sorted(pair, key=lambda ticker: (pair[ticker]["ticker_score"], ticker))[:k]
        assert facts["bottom_k"]["capacity_return_pct"] == pytest.approx(sum(pair[t]["oos_return_pct"] for t in bottom) / k)
        oracle = sorted(pair, key=lambda ticker: (-pair[ticker]["oos_return_pct"], ticker))[:k]
        anti = sorted(pair, key=lambda ticker: (pair[ticker]["oos_return_pct"], ticker))[:k]
        assert facts["oracle_k"]["selected_tickers"] == oracle
        assert facts["anti_oracle_k"]["selected_tickers"] == anti
        assert facts["all_available_mean_pct"] == np.mean([row["oos_return_pct"] for row in pair.values()])
        random = facts["random_k"]
        payload = {
            "schema": "strategy_lab_random_k_v1",
            "base_seed": 17,
            "dataset_manifest_sha256": open_dataset(root).manifest_sha256.lower(),
            "dataset_label": "canonical",
            "candidate_rule": "primary_profit",
            "ticker_scorer": {"name": "selected_is_net_profit", "version": "strategy_lab_ticker_score_v1", "configuration": {}},
            "allocation_kind": "random_k",
            "k": k,
            "oos_start_utc": "2026-03-01T00:00:00Z",
            "oos_end_utc": "2026-03-28T23:30:00Z",
        }
        digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
        assert random["seed_payload_sha256"] == digest.hex()
        assert random["derived_seed"] == int.from_bytes(digest[:8], "big", signed=False)
    random_rows = [row for row in result.ticker_allocations if row["row_kind"] == "random_summary"]
    assert len(random_rows) == 2 * 3
    assert all(row["ticker"] is None and row["draw_count"] == 50 for row in random_rows)
    assert not any(row["allocation_kind"] == "all_available" for row in result.ticker_allocations)
    reordered_dataset = open_dataset(root)
    reordered_scope = reordered_dataset.subset_scope(
        reordered_dataset.resolve_scope(),
        tickers=tuple(reversed(reordered_dataset.resolve_scope().tickers)),
    )
    reordered = evaluate_allocation(
        [DatasetInput("canonical", reordered_dataset, reordered_scope)],
        candidate_rule="primary_profit",
    )
    assert (
        reordered.summary["datasets"]["canonical"]["blocks"][0]["variants"]["primary"]["random_k"]
        == block["variants"]["primary"]["random_k"]
    )


@pytest.mark.parametrize(
    ("random", "observed", "expected"),
    [([1, 1], 1, 0.5), ([1, 2], 3, 1.0), ([2, 3], 1, 0.0), ([1, 2, 2, 3], 2, 0.5)],
)
def test_exact_tie_aware_random_percentile(random, observed, expected):
    assert random_percentile_fraction(random, observed) == expected


def test_turnover_cash_varying_k_compounding_and_full_tail_drawdown():
    fixed = _turnover([("AAA",), ("AAA", "BBB"), ("BBB",)], [3, 3, 3], varying=False)
    assert fixed["transitions"][0]["value"] is None
    assert [row["value"] for row in fixed["transitions"][1:]] == pytest.approx([1 / 3, 1 / 3])
    varying = _turnover([("AAA",), ("AAA", "BBB")], [1, 2], varying=True)
    assert varying["transitions"][1] == {"status": "unavailable", "reason": "varying_capacity", "value": None}
    portfolio = _portfolio([100.0, -25.0, -25.0])
    assert portfolio["compounded_return_pct"] == pytest.approx(12.5)
    assert portfolio["monthly_series_max_drawdown_pct"] == pytest.approx(43.75)
    assert _portfolio([None])["status"] == "unavailable"
    assert _portfolio([-100.0])["reason"] == "monthly gross factor is non-finite or non-positive"


def test_n1_n2_exact_calendar_alignment_window_ids_and_omissions(tmp_path):
    one_root = _synthetic_dataset(tmp_path / "one")
    n1 = evaluate_allocation([_input(one_root, "one")], candidate_rule="primary_profit")
    assert n1.run_metadata["alignment"]["common_block_count"] == 2

    two_root = _synthetic_dataset(
        tmp_path / "two",
        tickers=(("AAA", "dev"), ("CCC", "dev"), ("HHH", "holdout")),
    )
    _distinguish(two_root, "two")
    n2 = evaluate_allocation([_input(one_root, "one"), _input(two_root, "two")], candidate_rule="primary_profit")
    assert n2.run_metadata["alignment"]["common_tickers"] == ["AAA"]
    assert n2.summary["datasets"]["one"]["omitted_tickers"] == ["BBB"]
    assert n2.summary["datasets"]["two"]["omitted_tickers"] == ["CCC"]

    different_utc = _synthetic_dataset(tmp_path / "different_utc")
    _change_calendar(different_utc, day_offset=1)
    _distinguish(different_utc, "different")
    with pytest.raises(AnalysisError, match="no exact common"):
        evaluate_allocation([_input(one_root, "one"), _input(different_utc, "other")], candidate_rule="primary_profit")

    different_ids = _synthetic_dataset(tmp_path / "different_ids")
    _change_calendar(different_ids, id_offset=2)
    _distinguish(different_ids, "different-ids")
    aligned = evaluate_allocation([_input(one_root, "one"), _input(different_ids, "other")], candidate_rule="primary_profit")
    assert aligned.run_metadata["alignment"]["common_block_count"] == 2
    assert {row["window_id"] for row in aligned.pair_decisions if row["dataset_label"] == "other"} == {3, 4}


def test_n2_pool_intersects_candidate_and_score_availability(tmp_path):
    one_root = _synthetic_dataset(tmp_path / "one")
    two_root = _synthetic_dataset(tmp_path / "two")
    _distinguish(two_root, "two")
    scorer = TickerScorer(
        "aligned",
        "v1",
        {},
        lambda view, context: (
            None
            if view.dataset_label == "two" and view.ticker == "BBB"
            else view.is_net_profit_pct
        ),
    )
    result = evaluate_allocation(
        [_input(one_root, "one"), _input(two_root, "two")],
        candidate_rule="primary_profit",
        ticker_scorer=scorer,
    )
    for label in ("one", "two"):
        block = result.summary["datasets"][label]["blocks"][0]
        assert block["locally_available_tickers"] in (1, 2)
        assert block["available_tickers"] == 1
        assert block["variants"]["primary"]["selected_tickers"] == ["AAA"]


def test_arbitrary_universe_sizes_and_label_identity_validation(tmp_path):
    tickers = tuple((f"D{index:02d}", "dev") for index in range(13)) + (("H", "holdout"),)
    item = _input(_synthetic_dataset(tmp_path, tickers=tickers, declared_dev_pool=13))
    result = evaluate_allocation([item], candidate_rule="primary_profit", primary_k=1, sensitivity_k=13)
    assert result.run_metadata["alignment"]["common_ticker_count"] == 13
    for label in ("", "bad=name"):
        with pytest.raises(AnalysisError, match="labels"):
            evaluate_allocation([DatasetInput(label, item.dataset, item.scope)], candidate_rule="primary_profit")
    with pytest.raises(AnalysisError, match="unique"):
        evaluate_allocation([item, item], candidate_rule="primary_profit")
    duplicate = DatasetInput("copy", item.dataset, item.scope)
    with pytest.raises(AnalysisError, match="duplicate dataset identities"):
        evaluate_allocation([item, duplicate], candidate_rule="primary_profit")


def test_allocation_publication_noop_incompatible_preservation_and_analyze_files(tmp_path):
    item = _input(_synthetic_dataset(tmp_path / "input"))
    result = evaluate_allocation([item], candidate_rule="primary_profit")
    output = tmp_path / "allocation-output"
    first = write_allocation(result, output, dataset_roots=(item.dataset.root,))
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    second = write_allocation(result, output, dataset_roots=(item.dataset.root,))
    assert first["status"] == "published" and second["status"] == "verified_noop"
    assert set(before) == set(ALLOCATION_OUTPUT_FILES)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    (output / "summary.json").write_text("{}", encoding="utf-8")
    damaged = (output / "summary.json").read_bytes()
    with pytest.raises(AnalysisError, match="nonmatching"):
        write_allocation(result, output, dataset_roots=(item.dataset.root,))
    assert (output / "summary.json").read_bytes() == damaged

    from tools.strategy_lab.analysis.evaluate import evaluate_scope
    analysis = evaluate_scope(item.dataset, item.scope)
    rendered_before = render_files(analysis)
    assert tuple(rendered_before) == OUTPUT_FILES
    assert "ticker_allocations.csv" not in rendered_before


def test_cli_allocate_labels_errors_and_scope_lock(tmp_path, capsys):
    root = _synthetic_dataset(tmp_path / "input")
    output = tmp_path / "cli-output"
    assert analysis_main([
        "allocate", "--dataset", f"canonical={root}", "--scope", "development",
        "--rule", "primary_profit", "--output", str(output),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "published"
    assert analysis_main([
        "allocate", "--dataset", str(root), "--rule", "primary_profit",
        "--output", str(tmp_path / "bad"),
    ]) == 2
    assert "label=path" in capsys.readouterr().err
    assert analysis_main([
        "allocate", "--dataset", f"canonical={root}", "--scope", "holdout",
        "--rule", "primary_profit", "--output", str(tmp_path / "holdout"),
    ]) == 2
    assert "requires --unlock-scope" in capsys.readouterr().err


def test_opt_in_certifier_uses_bounded_underfill_and_full_common_path(tmp_path):
    tickers = tuple((f"D{index:02d}", "dev") for index in range(10)) + (("H", "holdout"),)
    root = _synthetic_dataset(tmp_path, tickers=tickers, declared_dev_pool=10)
    evidence = certify((("canonical", root),), "primary_profit")
    assert evidence["status"] == "passed"
    assert evidence["bounded"]["independent_oracle"] == "passed"
    assert evidence["underfill"]["independent_oracle"] == "passed"
    assert evidence["full_development"]["audit"]["canonical"]["holdout_loaded"] is False

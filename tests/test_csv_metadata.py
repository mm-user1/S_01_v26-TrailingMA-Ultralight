"""CSV metadata is portable; filesystem paths and stored payloads stay intact."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.csv_metadata import csv_basename
from core import storage
from core.bundle_export import _parse_csv_symbol_and_timeframe, build_lancelot_partial_bundle
from core.export import _extract_symbol_from_csv_filename
from core.optuna_engine import OptunaOptimizer
from ui import server_services


NAME = "OKX_LINKUSDT.P, 15 2025.01.01-2025.01.02.csv"
PATHS = [NAME, "C:\\data_dir\\" + NAME, "/data_dir/" + NAME,
         "C:\\data_dir/mixed\\" + NAME, "C:" + NAME, "\\\\server\\share\\" + NAME]


@pytest.mark.parametrize("value,expected", [(p, NAME) for p in PATHS] + [
    (None, ""), ("", ""), ("trailing/", ""), ("trailing\\", ""),
    (" Папка/ Имя, Файл.CSV ", " Имя, Файл.CSV "), (Path("file.csv"), "file.csv"),
])
def test_csv_basename(value, expected):
    assert csv_basename(value) == expected


@pytest.mark.parametrize("value", PATHS)
def test_csv_metadata_consumers(value, monkeypatch):
    assert storage._get_csv_display_name({"csv_original_name": value}, "other.csv") == NAME
    optimizer = SimpleNamespace(base_config=SimpleNamespace(csv_original_name=value))
    assert OptunaOptimizer._get_dataset_label(optimizer) == NAME
    assert _extract_symbol_from_csv_filename(value) == "OKX:LINKUSDT.P"
    assert _parse_csv_symbol_and_timeframe(value) == ("LINKUSDT.P", "15m")
    monkeypatch.setattr(server_services, "load_data", lambda path: pd.DataFrame())
    assert server_services._validate_csv_for_study("/selected_dir/" + NAME, {"csv_file_name": value}) == (True, [], None)


@pytest.mark.parametrize("missing", [None, "", "trailing/", "trailing\\"])
def test_csv_empty_normalization_fallbacks(missing, monkeypatch, tmp_path):
    config = {"csv_original_name": missing, "csv_file_name": "C:\\dir\\" + NAME}
    assert storage._get_csv_display_name(config, "last.csv") == NAME
    config["csv_file_name"] = missing
    assert storage._get_csv_display_name(config, "/dir/last.csv") == "last.csv"
    assert storage._get_csv_display_name(config, missing) == "upload"
    obj = SimpleNamespace(csv_original_name=missing, csv_file_name="unsupported.csv", csv_file="last.csv")
    assert storage._get_csv_display_name(obj, "last.csv") == "last.csv"
    assert storage._get_csv_display_name(obj, missing) == "upload"
    assert OptunaOptimizer._get_dataset_label(SimpleNamespace(base_config=obj)) == "last.csv"
    obj.csv_file = ""
    assert OptunaOptimizer._get_dataset_label(SimpleNamespace(base_config=obj)) == "upload"
    assert _extract_symbol_from_csv_filename(missing) == "UNKNOWN:UNKNOWN"
    with pytest.raises(ValueError, match="missing csv_file_name"):
        _parse_csv_symbol_and_timeframe(missing)
    monkeypatch.setattr(server_services, "load_data", lambda path: pd.DataFrame())
    assert server_services._validate_csv_for_study(NAME, {"csv_file_name": missing}) == (True, [], None)
    path = tmp_path / NAME
    path.write_text("data", encoding="utf-8")
    bundle = build_lancelot_partial_bundle(
        study={"csv_file_name": missing, "strategy_id": "s03_reversal_v10", "strategy_version": "v10",
               "study_name": "test", "study_id": "test"},
        params={"closeCountLong": 3}, trial_number=1, csv_path=str(path),
    )
    assert (bundle["symbol"], bundle["timeframe"]) == ("LINK/USDT:USDT", "15m")


def test_csv_different_filename_warns(monkeypatch):
    monkeypatch.setattr(server_services, "load_data", lambda path: pd.DataFrame())
    valid, warnings, error = server_services._validate_csv_for_study(
        "/selected_dir/other.csv", {"csv_file_name": "C:\\old_dir\\" + NAME})
    assert valid and error is None
    assert warnings == [f"Filename differs from original ({NAME} vs other.csv)."]


@pytest.mark.parametrize("original", ["C:\\data_dir\\" + NAME, "/data_dir/" + NAME, "trailing/", None])
def test_csv_grid_save_preserves_path_and_config(original, tmp_path):
    import json
    import time
    from core.grid_engine import GridSettings
    from core.optuna_engine import OptimizationConfig

    path = tmp_path / NAME
    config = OptimizationConfig(
        csv_file=str(path), csv_original_name=original, strategy_id="s03_reversal_v10",
        enabled_params={}, param_ranges={}, param_types={}, fixed_params={},
    )
    study_id = storage.save_grid_study_to_db(
        config=config, grid_settings=GridSettings(), grid_summary={}, trial_results=[],
        csv_file_path=str(path), start_time=time.time(),
    )
    with storage.get_db_connection() as conn:
        row = conn.execute("SELECT csv_file_name, csv_file_path, config_json FROM studies WHERE study_id=?", (study_id,)).fetchone()
    assert row["csv_file_name"] == NAME
    assert row["csv_file_path"] == str(path.resolve())
    assert json.loads(row["config_json"])["csv_original_name"] == original
    assert config.csv_original_name == original and config.csv_file == str(path)
    assert storage.load_study_from_db(study_id)["study"]["csv_file_name"] == NAME


@pytest.mark.parametrize("value", PATHS + [None, "", "trailing/", "trailing\\"])
def test_csv_historical_read_preserves_stored_columns(value):
    import json
    import uuid

    study_id = uuid.uuid4().hex
    path = "C:\\original_dir\\" + NAME
    config = json.dumps({"csv_original_name": value, "csv_file": path})
    with storage.get_db_connection() as conn:
        conn.execute("INSERT INTO studies (study_id, study_name, strategy_id, optimization_mode, csv_file_name, csv_file_path, config_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (study_id, study_id, "s03_reversal_v10", "optuna", value, path, config))
        conn.commit()
        before = tuple(conn.execute("SELECT csv_file_name, csv_file_path, config_json FROM studies WHERE study_id=?", (study_id,)).fetchone())
    expected = None if value is None else csv_basename(value)
    listed = next(row for row in storage.list_studies() if row["study_id"] == study_id)
    loaded = storage.load_study_from_db(study_id)["study"]
    assert listed["csv_file_name"] == loaded["csv_file_name"] == expected
    assert loaded["study_name"] == study_id
    assert loaded["csv_file_path"] == path
    assert loaded["config_json"] == json.loads(config)
    from ui.server import app
    with app.test_client() as client:
        response = client.get(f"/api/studies/{study_id}")
        assert response.status_code == 200
        assert response.get_json()["study"]["csv_file_name"] == expected
        rows = client.get("/api/studies").get_json()["studies"]
        assert next(row for row in rows if row["study_id"] == study_id)["csv_file_name"] == expected
    if expected == NAME:
        # Results extracts the dataset prefix at the first underscore.
        assert response.get_json()["study"]["csv_file_name"].split("_", 1)[1] == NAME.split("_", 1)[1]
    with storage.get_db_connection() as conn:
        after = tuple(conn.execute("SELECT csv_file_name, csv_file_path, config_json FROM studies WHERE study_id=?", (study_id,)).fetchone())
    assert after == before

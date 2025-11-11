import io
import json
import re
import time
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_file, send_from_directory

from backtest_engine import StrategyParams, load_data, run_strategy
from optimizer_engine import (
    CSV_COLUMN_SPECS,
    OptimizationResult,
    OptimizationConfig,
    PARAMETER_MAP,
    export_to_csv,
    run_optimization,
)

app = Flask(__name__)


MA_TYPES: Tuple[str, ...] = (
    "EMA",
    "SMA",
    "HMA",
    "WMA",
    "ALMA",
    "KAMA",
    "TMA",
    "T3",
    "DEMA",
    "VWMA",
    "VWAP",
)

SCORE_METRIC_KEYS: Tuple[str, ...] = (
    "romad",
    "sharpe",
    "pf",
    "ulcer",
    "recovery",
    "consistency",
)

DEFAULT_OPTIMIZER_SCORE_CONFIG: Dict[str, Any] = {
    "filter_enabled": False,
    "min_score_threshold": 60.0,
    "weights": {
        "romad": 0.25,
        "sharpe": 0.20,
        "pf": 0.20,
        "ulcer": 0.15,
        "recovery": 0.10,
        "consistency": 0.10,
    },
    "enabled_metrics": {
        "romad": True,
        "sharpe": True,
        "pf": True,
        "ulcer": True,
        "recovery": True,
        "consistency": True,
    },
    "invert_metrics": {"ulcer": True},
    "normalization_method": "percentile",
}

PRESETS_DIR = Path(__file__).resolve().parent / "Presets"
DEFAULT_PRESET_NAME = "defaults"
VALID_PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9 _\-]{1,64}$")

DEFAULT_PRESET: Dict[str, Any] = {
    "dateFilter": True,
    "backtester": True,
    "startDate": "2025-04-01",
    "startTime": "00:00",
    "endDate": "2025-09-01",
    "endTime": "00:00",
    "trendMATypes": list(MA_TYPES),
    "maLength": 45,
    "closeCountLong": 7,
    "closeCountShort": 5,
    "stopLongX": 2.0,
    "stopLongRR": 3.0,
    "stopLongLP": 2,
    "stopShortX": 2.0,
    "stopShortRR": 3.0,
    "stopShortLP": 2,
    "stopLongMaxPct": 3.0,
    "stopShortMaxPct": 3.0,
    "stopLongMaxDays": 2,
    "stopShortMaxDays": 4,
    "trailRRLong": 1.0,
    "trailRRShort": 1.0,
    "trailLongTypes": ["SMA"],
    "trailLongLength": 160,
    "trailLongOffset": -1.0,
    "trailShortTypes": ["SMA"],
    "trailLock": False,
    "trailShortLength": 160,
    "trailShortOffset": 1.0,
    "riskPerTrade": 2.0,
    "contractSize": 0.01,
    "workerProcesses": 6,
    "minProfitFilter": False,
    "minProfitThreshold": 0.0,
    "scoreFilterEnabled": False,
    "scoreThreshold": 60.0,
    "scoreWeights": {
        "romad": 0.25,
        "sharpe": 0.20,
        "pf": 0.20,
        "ulcer": 0.15,
        "recovery": 0.10,
        "consistency": 0.10,
    },
    "scoreEnabledMetrics": {
        "romad": True,
        "sharpe": True,
        "pf": True,
        "ulcer": True,
        "recovery": True,
        "consistency": True,
    },
    "scoreInvertMetrics": {"ulcer": True},
    "csvPath": "",
}
BOOL_FIELDS = {"dateFilter", "backtester", "trailLock", "minProfitFilter", "scoreFilterEnabled"}
INT_FIELDS = {
    "maLength",
    "closeCountLong",
    "closeCountShort",
    "stopLongLP",
    "stopShortLP",
    "stopLongMaxDays",
    "stopShortMaxDays",
    "trailLongLength",
    "trailShortLength",
    "workerProcesses",
}
FLOAT_FIELDS = {
    "stopLongX",
    "stopLongRR",
    "stopShortX",
    "stopShortRR",
    "stopLongMaxPct",
    "stopShortMaxPct",
    "trailRRLong",
    "trailRRShort",
    "trailLongOffset",
    "trailShortOffset",
    "riskPerTrade",
    "contractSize",
    "minProfitThreshold",
    "scoreThreshold",
}

LIST_FIELDS = {"trendMATypes", "trailLongTypes", "trailShortTypes"}
STRING_FIELDS = {"startDate", "startTime", "endDate", "endTime", "csvPath"}
ALLOWED_PRESET_FIELDS = set(DEFAULT_PRESET.keys())


def _clone_default_template() -> Dict[str, Any]:
    try:
        current_defaults = _load_preset(DEFAULT_PRESET_NAME)
        if not isinstance(current_defaults, dict):
            raise ValueError
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        current_defaults = DEFAULT_PRESET
    base = json.loads(json.dumps(DEFAULT_PRESET))
    base.update(json.loads(json.dumps(current_defaults)))
    return base


def _ensure_presets_directory() -> None:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    defaults_path = PRESETS_DIR / f"{DEFAULT_PRESET_NAME}.json"
    if not defaults_path.exists():
        _write_preset(DEFAULT_PRESET_NAME, DEFAULT_PRESET)


def _validate_preset_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Preset name must be a string.")
    normalized = name.strip()
    if not normalized:
        raise ValueError("Preset name cannot be empty.")
    if normalized.lower() == DEFAULT_PRESET_NAME:
        raise ValueError("Use the defaults endpoint to overwrite default settings.")
    if not VALID_PRESET_NAME_RE.match(normalized):
        raise ValueError(
            "Preset name may only contain letters, numbers, spaces, hyphens, and underscores."
        )
    return normalized


def _preset_path(name: str) -> Path:
    safe_name = Path(name).name
    return PRESETS_DIR / f"{safe_name}.json"


def _write_preset(name: str, values: Dict[str, Any]) -> None:
    path = _preset_path(name)
    serialized = json.loads(json.dumps(values))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _load_preset(name: str) -> Dict[str, Any]:
    path = _preset_path(name)
    if not path.exists():
        raise FileNotFoundError(name)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Preset file is corrupted.")
    return data


def _list_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    for path in sorted(PRESETS_DIR.glob("*.json")):
        name = path.stem
        presets.append({"name": name, "is_default": name.lower() == DEFAULT_PRESET_NAME})
    return presets


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return False


def _split_timestamp(value: str) -> Tuple[str, str]:
    normalized = (value or "").strip()
    if not normalized:
        return "", ""
    candidate = normalized.replace(" ", "T", 1)
    candidate = candidate.rstrip("Zz")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        if "T" in normalized:
            date_part, _, time_part = normalized.partition("T")
        elif " " in normalized:
            date_part, _, time_part = normalized.partition(" ")
        else:
            return normalized, ""
        return date_part.strip(), time_part.strip()
    date_part = parsed.date().isoformat()
    if parsed.time().second == 0 and parsed.time().microsecond == 0:
        time_part = parsed.time().strftime("%H:%M")
    else:
        time_part = parsed.time().strftime("%H:%M:%S")
    return date_part, time_part


def _convert_import_value(name: str, raw_value: str) -> Any:
    if name in BOOL_FIELDS:
        return _coerce_bool(raw_value)
    if name in INT_FIELDS:
        try:
            return int(round(float(raw_value)))
        except (TypeError, ValueError):
            return 0
    if name in FLOAT_FIELDS:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0
    return raw_value


def _parse_csv_parameter_block(file_storage) -> Tuple[Dict[str, Any], List[str]]:
    content = file_storage.read()
    if isinstance(content, bytes):
        text = content.decode("utf-8-sig", errors="replace")
    else:
        text = str(content)

    lines = text.splitlines()
    parameters: Dict[str, Any] = {}
    applied: List[str] = []

    header_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_seen:
                break
            continue
        if not header_seen:
            header_seen = True
            continue
        name, _, value = line.partition(",")
        param_name = name.strip()
        if not param_name:
            continue
        parameters[param_name] = value.strip()

    updates: Dict[str, Any] = {}
    for name, raw_value in parameters.items():
        if name == "start":
            date_part, time_part = _split_timestamp(raw_value)
            if date_part:
                updates["startDate"] = date_part
                applied.append("startDate")
            if time_part:
                updates["startTime"] = time_part
                applied.append("startTime")
            continue
        if name == "end":
            date_part, time_part = _split_timestamp(raw_value)
            if date_part:
                updates["endDate"] = date_part
                applied.append("endDate")
            if time_part:
                updates["endTime"] = time_part
                applied.append("endTime")
            continue
        if name == "maType":
            value = str(raw_value or "").strip().upper()
            if value:
                updates["trendMATypes"] = [value]
                applied.append("trendMATypes")
            continue
        if name == "trailLongType":
            value = str(raw_value or "").strip().upper()
            if value:
                updates["trailLongTypes"] = [value]
                applied.append("trailLongTypes")
            continue
        if name == "trailShortType":
            value = str(raw_value or "").strip().upper()
            if value:
                updates["trailShortTypes"] = [value]
                applied.append("trailShortTypes")
            continue

        converted = _convert_import_value(name, raw_value)
        updates[name] = converted
        applied.append(name)

    return updates, applied


_ensure_presets_directory()


def _normalize_preset_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError("Preset values must be provided as a dictionary.")
    normalized = _clone_default_template()
    for key, value in values.items():
        if key not in ALLOWED_PRESET_FIELDS:
            continue
        if key in LIST_FIELDS:
            if isinstance(value, (list, tuple)):
                cleaned = [str(item).strip().upper() for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                cleaned = [value.strip().upper()]
            else:
                cleaned = []
            if cleaned:
                normalized[key] = cleaned
            continue
        if key in BOOL_FIELDS:
            normalized[key] = _coerce_bool(value)
            continue
        if key in INT_FIELDS:
            try:
                converted = int(round(float(value)))
            except (TypeError, ValueError):
                continue
            if key == "workerProcesses":
                if converted < 1:
                    converted = 1
                elif converted > 32:
                    converted = 32
            normalized[key] = converted
            continue
        if key in FLOAT_FIELDS:
            try:
                converted_float = float(value)
            except (TypeError, ValueError):
                continue
            if key == "minProfitThreshold":
                converted_float = max(0.0, min(99000.0, converted_float))
            normalized[key] = converted_float
            continue
        if key in STRING_FIELDS:
            normalized[key] = str(value).strip()
            continue
        normalized[key] = value
    return normalized


def _resolve_csv_path(raw_path: str) -> Path:
    if raw_path is None:
        raise ValueError("CSV path is empty.")
    raw_value = str(raw_path).strip()
    if not raw_value:
        raise ValueError("CSV path is empty.")
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(candidate)) from exc
    if not resolved.is_file():
        raise IsADirectoryError(str(resolved))
    return resolved


@app.route("/")
def index() -> object:
    return send_from_directory(Path(app.root_path), "index.html")


@app.get("/api/presets")
def list_presets_endpoint() -> object:
    presets = _list_presets()
    return jsonify({"presets": presets})


@app.get("/api/presets/<string:name>")
def load_preset_endpoint(name: str) -> object:
    target = Path(name).stem
    try:
        values = _load_preset(target)
    except FileNotFoundError:
        return ("Preset not found.", HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        app.logger.exception("Failed to load preset '%s'", name)
        return (str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
    return jsonify({"name": target, "values": values})


@app.post("/api/presets")
def create_preset_endpoint() -> object:
    if not request.is_json:
        return ("Expected JSON body.", HTTPStatus.BAD_REQUEST)
    payload = request.get_json() or {}
    try:
        name = _validate_preset_name(payload.get("name"))
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)

    normalized_name_lower = name.lower()
    for entry in _list_presets():
        if entry["name"].lower() == normalized_name_lower:
            return ("Preset with this name already exists.", HTTPStatus.CONFLICT)

    try:
        values = _normalize_preset_payload(payload.get("values", {}))
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)

    try:
        _write_preset(name, values)
    except Exception:  # pragma: no cover - defensive
        app.logger.exception("Failed to save preset '%s'", name)
        return ("Failed to save preset.", HTTPStatus.INTERNAL_SERVER_ERROR)

    return jsonify({"name": name, "values": values}), HTTPStatus.CREATED


@app.put("/api/presets/<string:name>")
def overwrite_preset_endpoint(name: str) -> object:
    if not request.is_json:
        return ("Expected JSON body.", HTTPStatus.BAD_REQUEST)
    try:
        normalized_name = _validate_preset_name(name)
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)

    preset_path = _preset_path(normalized_name)
    if not preset_path.exists():
        return ("Preset not found.", HTTPStatus.NOT_FOUND)

    payload = request.get_json() or {}
    try:
        values = _normalize_preset_payload(payload.get("values", {}))
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)

    try:
        _write_preset(normalized_name, values)
    except Exception:  # pragma: no cover - defensive
        app.logger.exception("Failed to overwrite preset '%s'", name)
        return ("Failed to save preset.", HTTPStatus.INTERNAL_SERVER_ERROR)

    return jsonify({"name": normalized_name, "values": values})


@app.put("/api/presets/defaults")
def overwrite_defaults_endpoint() -> object:
    if not request.is_json:
        return ("Expected JSON body.", HTTPStatus.BAD_REQUEST)
    payload = request.get_json() or {}
    try:
        values = _normalize_preset_payload(payload.get("values", {}))
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)

    try:
        _write_preset(DEFAULT_PRESET_NAME, values)
    except Exception:  # pragma: no cover - defensive
        app.logger.exception("Failed to overwrite default preset")
        return ("Failed to save default preset.", HTTPStatus.INTERNAL_SERVER_ERROR)

    return jsonify({"name": DEFAULT_PRESET_NAME, "values": values})


@app.delete("/api/presets/<string:name>")
def delete_preset_endpoint(name: str) -> object:
    target = Path(name).stem
    if target.lower() == DEFAULT_PRESET_NAME:
        return ("Default preset cannot be deleted.", HTTPStatus.BAD_REQUEST)
    path = _preset_path(target)
    if not path.exists():
        return ("Preset not found.", HTTPStatus.NOT_FOUND)
    try:
        path.unlink()
    except Exception:  # pragma: no cover - defensive
        app.logger.exception("Failed to delete preset '%s'", name)
        return ("Failed to delete preset.", HTTPStatus.INTERNAL_SERVER_ERROR)
    return ("", HTTPStatus.NO_CONTENT)


@app.post("/api/presets/import-csv")
def import_preset_from_csv() -> object:
    if "file" not in request.files:
        return ("CSV file is required.", HTTPStatus.BAD_REQUEST)
    csv_file = request.files["file"]
    if not csv_file or csv_file.filename == "":
        return ("CSV file is required.", HTTPStatus.BAD_REQUEST)
    try:
        updates, applied = _parse_csv_parameter_block(csv_file)
    except Exception:  # pragma: no cover - defensive
        app.logger.exception("Failed to parse CSV for preset import")
        return ("Failed to parse CSV file.", HTTPStatus.BAD_REQUEST)
    if not updates:
        return ("No fixed parameters found in CSV.", HTTPStatus.BAD_REQUEST)
    return jsonify({"values": updates, "applied": applied})


@app.post("/api/walkforward")
def run_walkforward_optimization() -> object:
    """Run Walk-Forward Analysis"""
    data = request.form
    csv_file = request.files.get("file")
    csv_path_raw = (data.get("csvPath") or "").strip()
    data_source = None
    opened_file = None

    try:
        if csv_file and csv_file.filename:
            data_source = csv_file
        elif csv_path_raw:
            resolved_path = _resolve_csv_path(csv_path_raw)
            opened_file = resolved_path.open("rb")
            data_source = opened_file
        else:
            return jsonify({"error": "CSV file is required."}), HTTPStatus.BAD_REQUEST
    except (FileNotFoundError, IsADirectoryError, ValueError):
        return jsonify({"error": "CSV file is required."}), HTTPStatus.BAD_REQUEST
    except OSError:
        return jsonify({"error": "Failed to access CSV file."}), HTTPStatus.BAD_REQUEST

    config_raw = data.get("config")
    if not config_raw:
        if opened_file:
            opened_file.close()
        return jsonify({"error": "Missing optimization config."}), HTTPStatus.BAD_REQUEST

    try:
        config_payload = json.loads(config_raw)
    except json.JSONDecodeError:
        if opened_file:
            opened_file.close()
        return jsonify({"error": "Invalid optimization config JSON."}), HTTPStatus.BAD_REQUEST

    try:
        optimization_config = _build_optimization_config(data_source, config_payload)
    except ValueError as exc:
        if opened_file:
            opened_file.close()
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
    except Exception:  # pragma: no cover - defensive
        if opened_file:
            opened_file.close()
        app.logger.exception("Failed to build optimization config for walk-forward")
        return jsonify({"error": "Failed to prepare optimization config."}), HTTPStatus.INTERNAL_SERVER_ERROR

    if optimization_config.optimization_mode != "optuna":
        if opened_file:
            opened_file.close()
        return jsonify({"error": "Walk-Forward requires Optuna optimization mode."}), HTTPStatus.BAD_REQUEST

    if hasattr(data_source, "seek"):
        try:
            data_source.seek(0)
        except Exception:  # pragma: no cover - defensive
            pass

    try:
        df = load_data(data_source)
    except ValueError as exc:
        if opened_file:
            opened_file.close()
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
    except Exception:  # pragma: no cover - defensive
        if opened_file:
            opened_file.close()
        app.logger.exception("Failed to load CSV for walk-forward")
        return jsonify({"error": "Failed to load CSV data."}), HTTPStatus.INTERNAL_SERVER_ERROR

    base_template = {
        "enabled_params": json.loads(json.dumps(optimization_config.enabled_params)),
        "param_ranges": json.loads(json.dumps(optimization_config.param_ranges)),
        "fixed_params": json.loads(json.dumps(optimization_config.fixed_params)),
        "ma_types_trend": list(optimization_config.ma_types_trend),
        "ma_types_trail_long": list(optimization_config.ma_types_trail_long),
        "ma_types_trail_short": list(optimization_config.ma_types_trail_short),
        "lock_trail_types": bool(optimization_config.lock_trail_types),
        "risk_per_trade_pct": float(optimization_config.risk_per_trade_pct),
        "contract_size": float(optimization_config.contract_size),
        "commission_rate": float(optimization_config.commission_rate),
        "atr_period": int(optimization_config.atr_period),
        "worker_processes": int(optimization_config.worker_processes),
        "filter_min_profit": bool(optimization_config.filter_min_profit),
        "min_profit_threshold": float(optimization_config.min_profit_threshold),
        "score_config": json.loads(json.dumps(optimization_config.score_config or {})),
    }

    optuna_settings = {
        "target": getattr(optimization_config, "optuna_target", "score"),
        "budget_mode": getattr(optimization_config, "optuna_budget_mode", "trials"),
        "n_trials": int(getattr(optimization_config, "optuna_n_trials", 100)),
        "time_limit": int(getattr(optimization_config, "optuna_time_limit", 3600)),
        "convergence_patience": int(getattr(optimization_config, "optuna_convergence", 50)),
        "enable_pruning": bool(getattr(optimization_config, "optuna_enable_pruning", True)),
        "sampler": getattr(optimization_config, "optuna_sampler", "tpe"),
        "pruner": getattr(optimization_config, "optuna_pruner", "median"),
        "warmup_trials": int(getattr(optimization_config, "optuna_warmup_trials", 20)),
        "save_study": bool(getattr(optimization_config, "optuna_save_study", False)),
    }

    try:
        num_windows = int(data.get("wf_num_windows", 5))
        gap_bars = int(data.get("wf_gap_bars", 100))
        topk = int(data.get("wf_topk", 20))
    except (TypeError, ValueError):
        if opened_file:
            opened_file.close()
        return jsonify({"error": "Invalid Walk-Forward parameters."}), HTTPStatus.BAD_REQUEST

    num_windows = max(1, min(20, num_windows))
    gap_bars = max(0, gap_bars)
    topk = max(1, min(200, topk))

    from walkforward_engine import WFConfig, WalkForwardEngine, export_wf_results_csv

    wf_config = WFConfig(num_windows=num_windows, gap_bars=gap_bars, topk_per_window=topk)
    engine = WalkForwardEngine(wf_config, base_template, optuna_settings)

    try:
        result = engine.run_wf_optimization(df)
    except ValueError as exc:
        if opened_file:
            opened_file.close()
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
    except Exception:  # pragma: no cover - defensive
        if opened_file:
            opened_file.close()
        app.logger.exception("Walk-forward optimization failed")
        return jsonify({"error": "Walk-forward optimization failed."}), HTTPStatus.INTERNAL_SERVER_ERROR

    import uuid
    from pathlib import Path

    results_dir = Path(app.root_path) / "static" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"wf_results_{uuid.uuid4().hex[:8]}.csv"
    output_path = results_dir / output_filename
    export_wf_results_csv(result, str(output_path))

    top10: List[Dict[str, Any]] = []
    for rank, agg in enumerate(result.aggregated[:10], 1):
        forward_profit = result.forward_profits[rank - 1] if rank <= len(result.forward_profits) else None
        top10.append(
            {
                "rank": rank,
                "param_id": agg.param_id,
                "appearances": f"{agg.appearances}/{len(result.windows)}",
                "avg_oos_profit": round(agg.avg_oos_profit, 2),
                "oos_win_rate": round(agg.oos_win_rate * 100, 1),
                "forward_profit": round(forward_profit, 2) if isinstance(forward_profit, (int, float)) else None,
            }
        )

    response_payload = {
        "status": "success",
        "summary": {
            "total_windows": len(result.windows),
            "top_param_id": result.aggregated[0].param_id if result.aggregated else "N/A",
            "top_avg_oos_profit": round(result.aggregated[0].avg_oos_profit, 2) if result.aggregated else 0.0,
        },
        "top10": top10,
        "csv_url": f"/static/results/{output_filename}",
    }

    if opened_file:
        opened_file.close()

    return jsonify(response_payload)


@app.post("/api/backtest")
def run_backtest() -> object:
    csv_file = request.files.get("file")
    csv_path_raw = (request.form.get("csvPath") or "").strip()
    data_source = None
    opened_file = None

    if csv_file and csv_file.filename:
        data_source = csv_file
    elif csv_path_raw:
        try:
            resolved_path = _resolve_csv_path(csv_path_raw)
        except FileNotFoundError:
            return ("CSV file not found.", HTTPStatus.BAD_REQUEST)
        except IsADirectoryError:
            return ("CSV path must point to a file.", HTTPStatus.BAD_REQUEST)
        except ValueError:
            return ("CSV file is required.", HTTPStatus.BAD_REQUEST)
        except OSError:
            return ("Failed to access CSV file.", HTTPStatus.BAD_REQUEST)
        try:
            opened_file = resolved_path.open("rb")
        except OSError:
            return ("Failed to access CSV file.", HTTPStatus.BAD_REQUEST)
        data_source = opened_file
    else:
        return ("CSV file is required.", HTTPStatus.BAD_REQUEST)

    payload_raw = request.form.get("payload", "{}")
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return ("Invalid payload JSON.", HTTPStatus.BAD_REQUEST)

    try:
        params = StrategyParams.from_dict(payload)
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)

    try:
        df = load_data(data_source)
    except ValueError as exc:
        if opened_file:
            opened_file.close()
            opened_file = None
        return (str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        if opened_file:
            opened_file.close()
            opened_file = None
        app.logger.exception("Failed to load CSV")
        return ("Failed to load CSV data.", HTTPStatus.INTERNAL_SERVER_ERROR)
    finally:
        if opened_file:
            try:
                opened_file.close()
            except OSError:  # pragma: no cover - defensive
                pass
            opened_file = None

    try:
        result = run_strategy(df, params)
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.exception("Backtest execution failed")
        return ("Backtest execution failed.", HTTPStatus.INTERNAL_SERVER_ERROR)

    return jsonify({
        "metrics": result.to_dict(),
        "parameters": params.to_dict(),
    })


def _build_optimization_config(csv_file, payload: dict, worker_processes=None) -> OptimizationConfig:
    if not isinstance(payload, dict):
        raise ValueError("Invalid optimization config payload.")

    def _parse_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return default

    def _sanitize_score_config(raw_config: Any) -> Dict[str, Any]:
        source = raw_config if isinstance(raw_config, dict) else {}
        normalized = json.loads(json.dumps(DEFAULT_OPTIMIZER_SCORE_CONFIG))

        filter_value = source.get("filter_enabled")
        if filter_value is None:
            filter_value = source.get("filterEnabled")
        normalized["filter_enabled"] = _parse_bool(
            filter_value, normalized.get("filter_enabled", False)
        )

        threshold_value = source.get("min_score_threshold")
        if threshold_value is None:
            threshold_value = source.get("minScoreThreshold")
        try:
            threshold = float(threshold_value)
        except (TypeError, ValueError):
            threshold = normalized.get("min_score_threshold", 0.0)
        normalized["min_score_threshold"] = max(0.0, min(100.0, threshold))

        weights_raw = source.get("weights")
        if isinstance(weights_raw, dict):
            weights: Dict[str, float] = {}
            for key in SCORE_METRIC_KEYS:
                try:
                    weight_value = float(weights_raw.get(key, normalized["weights"].get(key, 0.0)))
                except (TypeError, ValueError):
                    weight_value = normalized["weights"].get(key, 0.0)
                weights[key] = max(0.0, min(1.0, weight_value))
            normalized["weights"].update(weights)

        enabled_raw = source.get("enabled_metrics")
        if enabled_raw is None:
            enabled_raw = source.get("enabledMetrics")
        if isinstance(enabled_raw, dict):
            enabled: Dict[str, bool] = {}
            for key in SCORE_METRIC_KEYS:
                enabled[key] = _parse_bool(
                    enabled_raw.get(key, normalized["enabled_metrics"].get(key, False)),
                    normalized["enabled_metrics"].get(key, False),
                )
            normalized["enabled_metrics"].update(enabled)

        invert_raw = source.get("invert_metrics")
        if invert_raw is None:
            invert_raw = source.get("invertMetrics")
        invert_flags: Dict[str, bool] = {}
        if isinstance(invert_raw, dict):
            for key in SCORE_METRIC_KEYS:
                invert_flags[key] = _parse_bool(
                    invert_raw.get(key, False),
                    False,
                )
        else:
            for key in SCORE_METRIC_KEYS:
                invert_flags[key] = normalized["invert_metrics"].get(key, False)
        normalized["invert_metrics"] = {
            key: value for key, value in invert_flags.items() if value
        }

        normalization_value = source.get("normalization_method")
        if normalization_value is None:
            normalization_value = source.get("normalizationMethod")
        if isinstance(normalization_value, str) and normalization_value.strip():
            normalized["normalization_method"] = normalization_value.strip().lower()

        return normalized

    enabled_params = payload.get("enabled_params")
    if not isinstance(enabled_params, dict):
        raise ValueError("enabled_params must be a dictionary.")

    param_ranges_raw = payload.get("param_ranges", {})
    if not isinstance(param_ranges_raw, dict):
        raise ValueError("param_ranges must be a dictionary.")
    param_ranges = {}
    for name, values in param_ranges_raw.items():
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"Invalid range for parameter '{name}'.")
        start, stop, step = values
        param_ranges[name] = (float(start), float(stop), float(step))

    fixed_params = payload.get("fixed_params", {})
    if not isinstance(fixed_params, dict):
        raise ValueError("fixed_params must be a dictionary.")

    ma_types_trend = payload.get("ma_types_trend") or payload.get("maTypesTrend") or []
    ma_types_trail_long = (
        payload.get("ma_types_trail_long")
        or payload.get("maTypesTrailLong")
        or []
    )
    ma_types_trail_short = (
        payload.get("ma_types_trail_short")
        or payload.get("maTypesTrailShort")
        or []
    )

    lock_trail_types_raw = (
        payload.get("lock_trail_types")
        or payload.get("lockTrailTypes")
        or payload.get("trailLock")
    )
    lock_trail_types = _parse_bool(lock_trail_types_raw, False)

    risk_per_trade = payload.get("risk_per_trade_pct")
    if risk_per_trade is None:
        risk_per_trade = payload.get("riskPerTrade", 2.0)
    contract_size = payload.get("contract_size")
    if contract_size is None:
        contract_size = payload.get("contractSize", 0.01)
    commission_rate = payload.get("commission_rate")
    if commission_rate is None:
        commission_rate = payload.get("commissionRate", 0.0005)
    atr_period = payload.get("atr_period")
    if atr_period is None:
        atr_period = payload.get("atrPeriod", 14)

    filter_min_profit_raw = payload.get("filter_min_profit")
    if filter_min_profit_raw is None:
        filter_min_profit_raw = payload.get("filterMinProfit")
    filter_min_profit = _parse_bool(filter_min_profit_raw, False)

    threshold_raw = payload.get("min_profit_threshold")
    if threshold_raw is None:
        threshold_raw = payload.get("minProfitThreshold", 0.0)
    try:
        min_profit_threshold = float(threshold_raw)
    except (TypeError, ValueError):
        min_profit_threshold = 0.0
    min_profit_threshold = max(0.0, min(99000.0, min_profit_threshold))

    if hasattr(csv_file, "seek"):
        try:
            csv_file.seek(0)
        except Exception:  # pragma: no cover - defensive
            pass
    elif hasattr(csv_file, "stream") and hasattr(csv_file.stream, "seek"):
        csv_file.stream.seek(0)
    worker_processes_value = 6 if worker_processes is None else int(worker_processes)
    if worker_processes_value < 1:
        worker_processes_value = 1
    elif worker_processes_value > 32:
        worker_processes_value = 32

    score_config_payload = payload.get("score_config")
    if score_config_payload is None:
        score_config_payload = payload.get("scoreConfig")
    score_config = _sanitize_score_config(score_config_payload)

    optimization_mode_raw = (
        payload.get("optimization_mode")
        or payload.get("optimizationMode")
        or "grid"
    )
    optimization_mode = str(optimization_mode_raw).strip().lower()
    if optimization_mode not in {"grid", "optuna"}:
        raise ValueError(f"Invalid optimization mode: {optimization_mode_raw}")

    optuna_params: Dict[str, Any] = {}
    if optimization_mode == "optuna":
        optuna_target = str(payload.get("optuna_target", "score")).strip().lower()
        optuna_budget_mode = str(payload.get("optuna_budget_mode", "trials")).strip().lower()

        try:
            optuna_n_trials = int(payload.get("optuna_n_trials", 500))
        except (TypeError, ValueError):
            optuna_n_trials = 500

        try:
            optuna_time_limit = int(payload.get("optuna_time_limit", 3600))
        except (TypeError, ValueError):
            optuna_time_limit = 3600

        try:
            optuna_convergence = int(payload.get("optuna_convergence", 50))
        except (TypeError, ValueError):
            optuna_convergence = 50

        try:
            optuna_warmup_trials = int(payload.get("optuna_warmup_trials", 20))
        except (TypeError, ValueError):
            optuna_warmup_trials = 20

        optuna_enable_pruning = _parse_bool(payload.get("optuna_enable_pruning", True), True)
        optuna_sampler = str(payload.get("optuna_sampler", "tpe")).strip().lower()
        optuna_pruner = str(payload.get("optuna_pruner", "median")).strip().lower()
        optuna_save_study = _parse_bool(payload.get("optuna_save_study", False), False)

        allowed_targets = {"score", "net_profit", "romad", "sharpe", "max_drawdown"}
        allowed_budget_modes = {"trials", "time", "convergence"}
        allowed_samplers = {"tpe", "random"}
        allowed_pruners = {"median", "percentile", "patient", "none"}

        if optuna_target not in allowed_targets:
            raise ValueError(f"Invalid Optuna target: {optuna_target}")
        if optuna_budget_mode not in allowed_budget_modes:
            raise ValueError(f"Invalid Optuna budget mode: {optuna_budget_mode}")
        if optuna_sampler not in allowed_samplers:
            raise ValueError(f"Invalid Optuna sampler: {optuna_sampler}")
        if optuna_pruner not in allowed_pruners:
            raise ValueError(f"Invalid Optuna pruner: {optuna_pruner}")

        optuna_n_trials = max(10, optuna_n_trials)
        optuna_time_limit = max(60, optuna_time_limit)
        optuna_convergence = max(10, optuna_convergence)
        optuna_warmup_trials = max(0, optuna_warmup_trials)

        optuna_params = {
            "optuna_target": optuna_target,
            "optuna_budget_mode": optuna_budget_mode,
            "optuna_n_trials": optuna_n_trials,
            "optuna_time_limit": optuna_time_limit,
            "optuna_convergence": optuna_convergence,
            "optuna_enable_pruning": optuna_enable_pruning,
            "optuna_sampler": optuna_sampler,
            "optuna_pruner": optuna_pruner,
            "optuna_warmup_trials": optuna_warmup_trials,
            "optuna_save_study": optuna_save_study,
        }

    config = OptimizationConfig(
        csv_file=csv_file,
        enabled_params=enabled_params,
        param_ranges=param_ranges,
        fixed_params=fixed_params,
        ma_types_trend=[str(ma).upper() for ma in ma_types_trend],
        ma_types_trail_long=[str(ma).upper() for ma in ma_types_trail_long],
        ma_types_trail_short=[str(ma).upper() for ma in ma_types_trail_short],
        risk_per_trade_pct=float(risk_per_trade),
        contract_size=float(contract_size),
        commission_rate=float(commission_rate),
        atr_period=int(atr_period),
        worker_processes=worker_processes_value,
        filter_min_profit=filter_min_profit,
        min_profit_threshold=min_profit_threshold,
        score_config=score_config,
        lock_trail_types=lock_trail_types,
        optimization_mode=optimization_mode,
    )

    if optimization_mode == "optuna":
        for key, value in optuna_params.items():
            setattr(config, key, value)

    return config


_DATE_PREFIX_RE = re.compile(r"\b\d{4}[.\-/]\d{2}[.\-/]\d{2}\b")
_DATE_VALUE_RE = re.compile(r"(\d{4})[.\-/]?(\d{2})[.\-/]?(\d{2})")


def _format_date_component(value: object) -> str:
    if value in (None, ""):
        return "0000.00.00"
    value_str = str(value).strip()
    if not value_str:
        return "0000.00.00"
    match = _DATE_VALUE_RE.search(value_str)
    if match:
        year, month, day = match.groups()
        return f"{year}.{month}.{day}"
    normalized = value_str.rstrip("Zz")
    normalized = normalized.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return "0000.00.00"
    return parsed.strftime("%Y.%m.%d")


def _unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


_PARAMETER_FRONTEND_ORDER = [
    frontend_name
    for _, frontend_name, _, _ in CSV_COLUMN_SPECS
    if frontend_name is not None
]


def _format_ma_segment(config: OptimizationConfig) -> str:
    ma_types = _unique_preserve_order([ma.upper() for ma in config.ma_types_trend])
    if not ma_types:
        return ""
    if len(ma_types) == 11:
        return "_ALL"
    is_ma_length_optimized = bool(config.enabled_params.get("maLength"))
    if len(ma_types) == 1 and not is_ma_length_optimized:
        ma_length_value = config.fixed_params.get("maLength")
        if ma_length_value is not None:
            try:
                ma_length_int = int(round(float(ma_length_value)))
                ma_length_str = str(ma_length_int)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                ma_length_str = str(ma_length_value)
            return f"_{ma_types[0]} {ma_length_str}"
        return f"_{ma_types[0]}"
    return "_" + "+".join(ma_types)


def generate_output_filename(csv_filename: str, config: OptimizationConfig) -> str:
    original_name = Path(csv_filename or "").name
    stem = Path(original_name).stem if original_name else ""
    prefix = stem
    if stem:
        match = _DATE_PREFIX_RE.search(stem)
        if match:
            prefix = stem[: match.start()].rstrip()
    prefix = prefix.strip() or "optimization"

    start_formatted = _format_date_component(config.fixed_params.get("start"))
    end_formatted = _format_date_component(config.fixed_params.get("end"))
    date_segment = f"{start_formatted}-{end_formatted}"
    ma_segment = _format_ma_segment(config)

    return f"{prefix} {date_segment}{ma_segment}.csv"


@app.post("/api/optimize")
def run_optimization_endpoint() -> object:
    csv_file = request.files.get("file")
    csv_path_raw = (request.form.get("csvPath") or "").strip()
    opened_file = None
    source_name = ""

    if csv_file and csv_file.filename:
        data_source = csv_file
        source_name = csv_file.filename
    elif csv_path_raw:
        try:
            resolved_path = _resolve_csv_path(csv_path_raw)
        except FileNotFoundError:
            return ("CSV file not found.", HTTPStatus.BAD_REQUEST)
        except IsADirectoryError:
            return ("CSV path must point to a file.", HTTPStatus.BAD_REQUEST)
        except ValueError:
            return ("CSV file is required.", HTTPStatus.BAD_REQUEST)
        except OSError:
            return ("Failed to access CSV file.", HTTPStatus.BAD_REQUEST)
        try:
            opened_file = resolved_path.open("rb")
        except OSError:
            return ("Failed to access CSV file.", HTTPStatus.BAD_REQUEST)
        data_source = opened_file
        source_name = str(resolved_path)
    else:
        return ("CSV file is required.", HTTPStatus.BAD_REQUEST)

    config_raw = request.form.get("config")
    if not config_raw:
        return ("Optimization config is required.", HTTPStatus.BAD_REQUEST)
    try:
        config_payload = json.loads(config_raw)
    except json.JSONDecodeError:
        return ("Invalid optimization config JSON.", HTTPStatus.BAD_REQUEST)

    try:
        worker_processes_raw = config_payload.get("worker_processes")
        if worker_processes_raw is None:
            worker_processes_raw = config_payload.get("workerProcesses")
        if worker_processes_raw is None:
            worker_processes = 6
        else:
            try:
                worker_processes = int(worker_processes_raw)
            except (TypeError, ValueError):
                return ("Invalid worker process count.", HTTPStatus.BAD_REQUEST)
            if worker_processes < 1:
                worker_processes = 1
            elif worker_processes > 32:
                worker_processes = 32

        optimization_config = _build_optimization_config(
            data_source, config_payload, worker_processes
        )
    except ValueError as exc:
        if opened_file:
            opened_file.close()
            opened_file = None
        return (str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        if opened_file:
            opened_file.close()
            opened_file = None
        app.logger.exception("Failed to construct optimization config")
        return ("Failed to prepare optimization config.", HTTPStatus.INTERNAL_SERVER_ERROR)

    results: List[OptimizationResult] = []
    optimization_metadata: Optional[Dict[str, Any]] = None
    try:
        start_time = time.time()
        results = run_optimization(optimization_config)
        end_time = time.time()

        optimization_time_seconds = max(0.0, end_time - start_time)
        minutes = int(optimization_time_seconds // 60)
        seconds = int(optimization_time_seconds % 60)
        optimization_time_str = f"{minutes}m {seconds}s"

        if optimization_config.optimization_mode == "optuna":
            target_labels = {
                "score": "Composite Score",
                "net_profit": "Net Profit %",
                "romad": "RoMaD",
                "sharpe": "Sharpe Ratio",
                "max_drawdown": "Max Drawdown %",
            }

            summary = getattr(optimization_config, "optuna_summary", {})
            total_trials = int(summary.get("total_trials", getattr(optimization_config, "optuna_n_trials", 0)))
            completed_trials = int(summary.get("completed_trials", len(results)))
            pruned_trials = int(summary.get("pruned_trials", 0))
            best_value = summary.get("best_value")

            if best_value is None and results:
                best_result = results[0]
                if optimization_config.optuna_target == "score":
                    best_value = best_result.score
                elif optimization_config.optuna_target == "net_profit":
                    best_value = best_result.net_profit_pct
                elif optimization_config.optuna_target == "romad":
                    best_value = best_result.romad
                elif optimization_config.optuna_target == "sharpe":
                    best_value = best_result.sharpe_ratio
                elif optimization_config.optuna_target == "max_drawdown":
                    best_value = best_result.max_drawdown_pct

            best_value_str = "-"
            if best_value is not None:
                try:
                    best_value_str = f"{float(best_value):.4f}"
                except (TypeError, ValueError):
                    best_value_str = str(best_value)

            optimization_metadata = {
                "method": "Optuna",
                "target": target_labels.get(optimization_config.optuna_target, "Composite Score"),
                "total_trials": total_trials,
                "completed_trials": completed_trials,
                "pruned_trials": pruned_trials,
                "best_trial_number": summary.get("best_trial_number"),
                "best_value": best_value_str,
                "optimization_time": optimization_time_str,
            }
        else:
            optimization_metadata = {
                "method": "Grid Search",
                "total_combinations": len(results),
                "optimization_time": optimization_time_str,
            }
    except ValueError as exc:
        if opened_file:
            opened_file.close()
            opened_file = None
        return (str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        if opened_file:
            opened_file.close()
            opened_file = None
        app.logger.exception("Optimization run failed")
        return ("Optimization execution failed.", HTTPStatus.INTERNAL_SERVER_ERROR)
    finally:
        if opened_file:
            try:
                opened_file.close()
            except OSError:  # pragma: no cover - defensive
                pass
            opened_file = None

    fixed_parameters = []
    trend_types = _unique_preserve_order(optimization_config.ma_types_trend)
    trail_long_types = _unique_preserve_order(optimization_config.ma_types_trail_long)
    trail_short_types = _unique_preserve_order(optimization_config.ma_types_trail_short)

    for name in _PARAMETER_FRONTEND_ORDER:
        if name == "maType":
            if len(trend_types) == 1:
                fixed_parameters.append((name, trend_types[0]))
            continue
        if name == "trailLongType":
            if len(trail_long_types) == 1:
                fixed_parameters.append((name, trail_long_types[0]))
            continue
        if name == "trailShortType":
            if len(trail_short_types) == 1:
                fixed_parameters.append((name, trail_short_types[0]))
            continue

        if bool(optimization_config.enabled_params.get(name, False)):
            continue

        value = optimization_config.fixed_params.get(name)
        if value is None:
            param_info = PARAMETER_MAP.get(name)
            if param_info and results:
                attr_name = param_info[0]
                value = getattr(results[0], attr_name, None)
        fixed_parameters.append((name, value))

    csv_content = export_to_csv(
        results,
        fixed_parameters,
        filter_min_profit=optimization_config.filter_min_profit,
        min_profit_threshold=optimization_config.min_profit_threshold,
        optimization_metadata=optimization_metadata,
    )
    buffer = io.BytesIO(csv_content.encode("utf-8"))
    filename = generate_output_filename(source_name, optimization_config)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

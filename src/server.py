import io
import json
import re
from datetime import datetime
from http import HTTPStatus
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from backtest_engine import StrategyParams, load_data, run_strategy
from optimizer_engine import (
    OptimizationConfig,
    PARAMETER_MAP,
    export_to_csv,
    run_optimization,
)

app = Flask(__name__)


@app.route("/")
def index() -> object:
    return send_from_directory(Path(app.root_path), "index.html")


@app.post("/api/backtest")
def run_backtest() -> object:
    if "file" not in request.files:
        return ("CSV file is required.", HTTPStatus.BAD_REQUEST)

    csv_file = request.files["file"]
    if csv_file.filename == "":
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
        df = load_data(csv_file)
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.exception("Failed to load CSV")
        return ("Failed to load CSV data.", HTTPStatus.INTERNAL_SERVER_ERROR)

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
    return OptimizationConfig(
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
    )


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
    if "file" not in request.files:
        return ("CSV file is required.", HTTPStatus.BAD_REQUEST)

    csv_file = request.files["file"]
    if csv_file.filename == "":
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
            csv_file, config_payload, worker_processes
        )
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.exception("Failed to construct optimization config")
        return ("Failed to prepare optimization config.", HTTPStatus.INTERNAL_SERVER_ERROR)

    try:
        results = run_optimization(optimization_config)
    except ValueError as exc:
        return (str(exc), HTTPStatus.BAD_REQUEST)
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.exception("Optimization run failed")
        return ("Optimization execution failed.", HTTPStatus.INTERNAL_SERVER_ERROR)

    fixed_parameters = {}
    for name, enabled in optimization_config.enabled_params.items():
        if not bool(enabled):
            value = optimization_config.fixed_params.get(name)
            if value is None:
                param_info = PARAMETER_MAP.get(name)
                if param_info and results:
                    attr_name = param_info[0]
                    value = getattr(results[0], attr_name, None)
            fixed_parameters[name] = value

    csv_content = export_to_csv(results, fixed_parameters)
    buffer = io.BytesIO(csv_content.encode("utf-8"))
    filename = generate_output_filename(csv_file.filename, optimization_config)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

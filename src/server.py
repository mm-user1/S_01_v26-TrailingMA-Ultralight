import json
from http import HTTPStatus
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from backtest_engine import StrategyParams, load_data, run_strategy

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

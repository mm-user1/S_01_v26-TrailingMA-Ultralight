"""Server queue contracts."""

import json
from pathlib import Path

import pytest

from ._helpers import _s03_regime_er_grid_preview_payload


def _patch_queue_storage_path(monkeypatch, tmp_path: Path, filename: str) -> Path:
    from ui import server_services

    queue_file = tmp_path / filename

    monkeypatch.setattr(
        server_services,
        "_queue_storage_file_path",
        lambda: queue_file,
    )
    return queue_file


def test_queue_api_roundtrip_persists_in_file_storage(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_roundtrip.json")

    payload = {
        "items": [
            {
                "id": "q_test_1",
                "index": 1,
                "label": "#1 example",
                "sources": [{"type": "path", "path": r"C:\data\file_1.csv"}],
                "sourceCursor": 0,
                "successCount": 0,
                "failureCount": 0,
            }
        ],
        "nextIndex": 2,
        "runtime": {"active": False, "updatedAt": 0},
    }

    response_put = client.put("/api/queue", json=payload)
    assert response_put.status_code == 200
    put_data = response_put.get_json()
    assert put_data["nextIndex"] == 2
    assert len(put_data["items"]) == 1
    assert queue_file.exists()

    response_get = client.get("/api/queue")
    assert response_get.status_code == 200
    get_data = response_get.get_json()
    assert len(get_data["items"]) == 1
    assert get_data["items"][0]["id"] == "q_test_1"
    assert get_data["items"][0]["sources"][0]["path"] == r"C:\data\file_1.csv"

    response_delete = client.delete("/api/queue")
    assert response_delete.status_code == 200
    delete_data = response_delete.get_json()
    assert delete_data["items"] == []
    assert delete_data["nextIndex"] == 1
    assert delete_data["runtime"]["active"] is False
    assert not queue_file.exists()


def test_queue_api_empty_items_removes_queue_file(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_empty_cleanup.json")

    seed_payload = {
        "items": [
            {
                "id": "q_test_2",
                "index": 1,
                "label": "#1 seed",
                "sources": [{"type": "path", "path": r"C:\data\seed.csv"}],
            }
        ],
        "nextIndex": 2,
        "runtime": {"active": True, "updatedAt": 123},
    }

    response_seed = client.put("/api/queue", json=seed_payload)
    assert response_seed.status_code == 200
    assert queue_file.exists()

    response_clear = client.put(
        "/api/queue",
        json={
            "items": [],
            "nextIndex": 999,
            "runtime": {"active": True, "updatedAt": 999},
        },
    )
    assert response_clear.status_code == 200
    clear_data = response_clear.get_json()
    assert clear_data["items"] == []
    assert clear_data["nextIndex"] == 1
    assert clear_data["runtime"]["active"] is False
    assert clear_data["runtime"]["updatedAt"] == 0
    assert not queue_file.exists()


def test_queue_api_roundtrip_preserves_extended_item_metadata(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_extended_metadata.json")

    payload = {
        "items": [
            {
                "id": "q_test_meta",
                "index": 73,
                "label": "#73 example",
                "mode": "wfa",
                "finalState": "completed",
                "dbTarget": "analytics_01.sqlite",
                "sources": [
                    {"type": "path", "path": r"C:\data\alpha_30m.csv"},
                    {"type": "path", "path": r"C:\data\beta_30m.csv"},
                ],
                "sourceCursor": 2,
                "successCount": 2,
                "failureCount": 0,
                "studySet": {
                    "autoCreate": True,
                    "completedStudyIds": ["study_a", "study_b"],
                    "createdSetId": 11,
                    "createdSetName": "#73 · S03 · 30m · NSGA-2 (357) · 1.5k · WFA-F 60/30",
                    "status": "created",
                    "error": "",
                    "lastUpdatedAt": "2026-03-12T10:15:00Z",
                },
                "uiSnapshot": {
                    "selectedTab": "optimizer",
                    "dbTarget": {"value": "analytics_01.sqlite"},
                },
                "wfa": {
                    "isPeriodDays": 60,
                    "oosPeriodDays": 30,
                    "storeTopNTrials": 50,
                    "adaptiveMode": True,
                    "cooldownEnabled": True,
                    "cooldownDays": 15,
                    "maxOosPeriodDays": 120,
                    "minOosTrades": 7,
                    "checkIntervalTrades": 4,
                    "cusumThreshold": 5.5,
                    "ddThresholdMultiplier": 1.7,
                    "inactivityMultiplier": 6.2,
                },
            }
        ],
        "nextIndex": 74,
        "runtime": {"active": False, "updatedAt": 0},
    }

    response_put = client.put("/api/queue", json=payload)
    assert response_put.status_code == 200
    stored = response_put.get_json()
    assert stored["items"][0]["finalState"] == "completed"
    assert stored["items"][0]["studySet"]["createdSetId"] == 11
    assert stored["items"][0]["studySet"]["completedStudyIds"] == ["study_a", "study_b"]
    assert stored["items"][0]["uiSnapshot"]["dbTarget"]["value"] == "analytics_01.sqlite"
    assert stored["items"][0]["wfa"]["cooldownEnabled"] is True
    assert stored["items"][0]["wfa"]["cooldownDays"] == 15

    response_get = client.get("/api/queue")
    assert response_get.status_code == 200
    loaded = response_get.get_json()
    assert loaded["items"][0]["studySet"]["createdSetName"].startswith("#73")
    assert loaded["items"][0]["dbTarget"] == "analytics_01.sqlite"
    assert loaded["items"][0]["wfa"]["cooldownEnabled"] is True
    assert loaded["items"][0]["wfa"]["cooldownDays"] == 15

    on_disk = json.loads(queue_file.read_text(encoding="utf-8"))
    assert on_disk["items"][0]["studySet"]["status"] == "created"
    assert on_disk["items"][0]["uiSnapshot"]["selectedTab"] == "optimizer"
    assert on_disk["items"][0]["wfa"]["cooldownEnabled"] is True
    assert on_disk["items"][0]["wfa"]["cooldownDays"] == 15


def test_queue_api_roundtrip_preserves_b2_wfa_grid_transport(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_b2_wfa_grid.json")
    item = {
        "id": "q_b2_wfa_grid",
        "index": 22,
        "label": "#22 B2 WFA Grid",
        "strategyId": "s06_r_trend_v02_b2",
        "mode": "wfa",
        "warmupBars": 1500,
        "sources": [{"type": "path", "path": r"C:\data\b2.csv"}],
        "sourceCursor": 1,
        "successCount": 1,
        "failureCount": 0,
        "config": {
            "optimization_mode": "grid",
            "fixed_params": {
                "dateFilter": True,
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-06-30T23:59:59.999999Z",
            },
            "grid_v2_planning_policy": "sampled",
            "grid_budget": 1200,
            "grid_seed": 41,
            "grid_allocation_method": "manual",
            "grid_manual_percents": {"bracket": 35, "trail": 65},
            "planned_candidate_count": 1200,
            "planned_candidate_policy": "sampled",
            "planning_policy_version": "grid_v2_planning_policy_v1",
            "future": {"nested": ["opaque", 7]},
        },
        "planned_candidate_count": 1200,
        "planned_candidate_policy": "sampled",
        "planning_policy_version": "grid_v2_planning_policy_v1",
        "wfa": {
            "isPeriodDays": 90,
            "oosPeriodDays": 30,
            "adaptiveMode": True,
            "cooldownEnabled": True,
            "cooldownDays": 15,
            "storeTopNTrials": 25,
        },
        "studySet": {"completedStudyIds": ["study_1"]},
        "forwardCompatible": {"schemaHint": "future-v3"},
    }
    payload = {
        "items": [item],
        "nextIndex": 23,
        "runtime": {"active": True, "updatedAt": 123456},
    }

    response = client.put("/api/queue", json=payload)
    assert response.status_code == 200
    assert response.get_json() == payload
    assert json.loads(queue_file.read_text(encoding="utf-8")) == payload
    assert client.get("/api/queue").get_json() == payload


def _assert_queue_get_preserves_unreadable_file(client, queue_file: Path, raw: bytes):
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 409
    assert response.get_json() == {
        "error": "Stored Queue state is unreadable. The source file was preserved."
    }
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


def test_queue_get_malformed_json_is_non_mutating(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_malformed.json")
    _assert_queue_get_preserves_unreadable_file(client, queue_file, b'{"items": [')


def test_queue_get_invalid_utf8_is_non_mutating(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_invalid_utf8.json")
    _assert_queue_get_preserves_unreadable_file(client, queue_file, b"\xff\xfe\x80")


@pytest.mark.parametrize("raw", [b"[]", b"null", b'{"items": null}', b'{"items": {}}'])
def test_queue_get_invalid_top_level_shape_is_non_mutating(
    client,
    monkeypatch,
    tmp_path,
    raw,
):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_invalid_shape.json")
    _assert_queue_get_preserves_unreadable_file(client, queue_file, raw)


def test_queue_get_accepts_utf8_bom_without_rewrite(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_bom.json")
    raw = b"\xef\xbb\xbf" + json.dumps({"items": []}, separators=(",", ":")).encode("utf-8")
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 200
    assert response.get_json()["items"] == []
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


@pytest.mark.parametrize("payload", [{}, {"items": []}])
def test_queue_get_valid_empty_state_does_not_delete_or_rewrite(
    client,
    monkeypatch,
    tmp_path,
    payload,
):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_valid_empty.json")
    raw = json.dumps(payload, indent=2).encode("utf-8")
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 200
    assert response.get_json() == {
        "items": [],
        "nextIndex": 1,
        "runtime": {"active": False, "updatedAt": 0},
    }
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


def test_queue_get_keeps_lenient_item_normalization_without_rewriting(
    client,
    monkeypatch,
    tmp_path,
):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_lenient_items.json")
    raw = json.dumps({"items": [7, {"sources": []}], "nextIndex": 9}, indent=2).encode("utf-8")
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 200
    assert response.get_json()["items"] == []
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


def test_queue_progress_put_and_v1_transport_remain_compatible(client, monkeypatch, tmp_path):
    _patch_queue_storage_path(monkeypatch, tmp_path, "queue_v1_progress.json")
    payload = {
        "items": [{
            "id": "q_v1",
            "index": 1,
            "label": "#1 V1",
            "strategyId": "s03_reversal_v10",
            "mode": "grid",
            "warmupBars": 1000,
            "sources": [{"type": "path", "path": r"C:\data\v1.csv"}],
            "sourceCursor": 1,
            "successCount": 1,
            "failureCount": 0,
            "config": {"optimization_mode": "grid", "grid_seed": 7},
        }],
        "nextIndex": 2,
        "runtime": {"active": True, "updatedAt": 999},
    }

    response = client.put("/api/queue", json=payload)

    assert response.status_code == 200
    assert response.get_json() == payload
    assert client.get("/api/queue").get_json() == payload


def test_queue_legacy_missing_warmup_reaches_v2_runtime_default_once(
    monkeypatch,
    client,
    tmp_path,
):
    from ui import server_routes_run, server_services

    csv_path = tmp_path / "queue_legacy_warmup.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    payload = _s03_regime_er_grid_preview_payload(optimization_mode="grid")
    captured = []
    normalization_calls = []
    original_builder = server_routes_run._build_optimization_config
    original_normalizer = server_services.normalize_v2_runtime_values

    def capture_builder(*args, **kwargs):
        config = original_builder(*args, **kwargs)
        captured.append(config)
        return config

    def count_normalization(*args, **kwargs):
        normalization_calls.append((args, kwargs))
        return original_normalizer(*args, **kwargs)

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "_build_optimization_config", capture_builder)
    monkeypatch.setattr(server_routes_run, "run_optimization", lambda _config: ([], None))
    monkeypatch.setattr(server_services, "normalize_v2_runtime_values", count_normalization)

    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    assert len(normalization_calls) == 1
    assert len(captured) == 1
    assert captured[0].warmup_bars == 1000


def test_queue_api_rejects_non_object_payload(client):
    response = client.put("/api/queue", json=["not", "an", "object"])
    assert response.status_code == 400
    payload = response.get_json()
    assert "json object" in payload["error"].lower()

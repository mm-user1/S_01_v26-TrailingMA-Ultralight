"""Tie selection transport through HTTP, stored Preview and Queue."""
import json
import uuid
from copy import deepcopy

import pytest

from core import storage
from core.grid_engine import _grid_v2_settings_from_config, preview_grid_parameter_space
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy
from ui import server_services

STRATEGY = strategy.S03ReversalV164AAdaptiveMAB2.STRATEGY_ID


def request(ties=None):
    return {
        "strategy_id": STRATEGY, "optimization_mode":"grid",
        "enabled_params":dict.fromkeys(strategy.SIGNAL_CACHE_PARAM_NAMES, True),
        "param_ranges":{}, "param_types":{}, "fixed_params":{"dateFilter":False},
        "grid_v2_enabled_tie_groups": [] if ties is None else ties,
        "grid_fast_objectives":["net_profit_pct"], "objectives":["net_profit_pct"],
        "worker_processes":1,
    }


@pytest.mark.parametrize("ties,count", [([],196000),(["symmetricLongShort"],2800)])
def test_http_preview_and_core_builder(client,ties,count):
    payload=request(ties)
    original=deepcopy(payload)
    response=client.post('/api/grid/preview',json=payload)
    assert response.status_code == 200, response.get_json()
    assert response.get_json()['preview']['full_candidate_count'] == count
    config=server_services._build_optimization_config('unused.csv',payload,1,STRATEGY)
    assert _grid_v2_settings_from_config(config).enabled_tie_groups == tuple(ties)
    assert preview_grid_parameter_space(config)['full_candidate_count'] == count
    assert payload == original


@pytest.mark.parametrize("canonical,nested,count", [(None,["symmetricLongShort"],2800),([],["symmetricLongShort"],196000),(None,None,196000),(["symmetricLongShort"],[],2800)])
def test_stored_nested_only_preview_roundtrip(canonical,nested,count):
    payload=request()
    if canonical is None: payload.pop('grid_v2_enabled_tie_groups')
    else: payload['grid_v2_enabled_tie_groups']=canonical
    if nested is not None: payload['grid_config']={'enabled_tie_groups':nested}
    study_id = str(uuid.uuid4())
    with storage.get_db_connection() as connection:
        connection.execute('INSERT INTO studies(study_id,study_name,strategy_id,optimization_mode,config_json) VALUES(?,?,?,?,?)',(study_id,study_id,STRATEGY,"grid",json.dumps(payload)))
        connection.commit()
        row=connection.execute('SELECT config_json FROM studies WHERE study_id=?',(study_id,)).fetchone()
    reloaded=json.loads(row[0])
    preview=server_services._derive_grid_preview(reloaded,{'strategy_id':STRATEGY})
    assert preview['full_candidate_count'] == count
    assert reloaded == payload


@pytest.mark.parametrize("value", [None, "symmetricLongShort", ["unknown"], ["symmetricLongShort","symmetricLongShort"], [1]])
def test_invalid_selection_rejected_before_market(client,value):
    payload=request(value)
    payload['grid_v2_enabled_tie_groups']=value
    response=client.post('/api/grid/preview',json=payload)
    assert response.status_code == 400, response.get_json()


def test_stale_target_ranges_are_inert_and_source_ranges_apply(client):
    payload=request(['symmetricLongShort'])
    payload['param_ranges']={'closeCountShort':'stale','tBandShortPct':[None,None,None], 'closeCountLong':[2,3,1]}
    payload['fixed_params'].update(closeCountShort='stale',tBandShortPct='stale')
    response=client.post('/api/grid/preview',json=payload)
    assert response.status_code == 200, response.get_json()
    assert response.get_json()['preview']['full_candidate_count'] == 800
    payload['enabled_params']['closeCountLong']=False
    assert client.post('/api/grid/preview',json=payload).status_code == 400


@pytest.mark.parametrize("mode", ['optimize','wfa'])
def test_queue_preserves_request_and_optional_restoration(client,mode,tmp_path,monkeypatch):
    monkeypatch.setattr(server_services, "_queue_storage_file_path", lambda: tmp_path / "queue.json")
    item={'id':'ties-queue','index':1,'strategyId':STRATEGY,'mode':mode,'config':request(['symmetricLongShort']),
          'sources':[{'type':'path','path':str(tmp_path / 'task-owned.csv')}],
          'uiSnapshot':{'version':1,'controls':{},'parameterTies':{'strategyId':STRATEGY,'groups':{'symmetricLongShort':{'active':True,'fields':{'backtest_closeCountLong':{'value':'7'},'backtest_closeCountShort':{'value':'5'}}}}}}}
    response=client.put('/api/queue',json={'items':[item],'nextIndex':2,'runtime':{'active':False,'updatedAt':0}})
    assert response.status_code == 200,response.get_json()
    loaded=client.get('/api/queue').get_json()['items'][0]
    assert loaded['config'] == item['config']
    assert loaded['uiSnapshot'] == item['uiSnapshot']
    config=server_services._build_optimization_config('unused.csv',loaded['config'],1,STRATEGY)
    assert preview_grid_parameter_space(config)['full_candidate_count'] == 2800

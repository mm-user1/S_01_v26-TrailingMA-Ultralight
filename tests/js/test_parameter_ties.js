'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const config = JSON.parse(fs.readFileSync(path.join(root, 'src/strategies/s03_reversal_v16_4_a_adaptive_ma_b2/config.json')));
const controls = new Map();
function control(id, value='', type='number') {
  const element = {id, value:String(value), type, checked:false, disabled:false, dataset:{}, style:{},
    innerHTML:'', textContent:'', classList:{add(){},remove(){},toggle(){}}, addEventListener(){}};
  controls.set(id, element);
  return element;
}
const sandbox = {console, JSON, Set, Object, Array, String, Boolean, document:{
  getElementById: id => controls.get(id),
  querySelector: () => null,
  querySelectorAll: () => Array.from(controls.values()),
}, window:{}, alert(){}, resetGridPreviewState(){ sandbox.invalidations++; }, invalidations:0,
scheduleGridPreviewUpdate(){ sandbox.previews++; }, previews:0};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root,'src/ui/static/js/strategy-config.js'),'utf8'),sandbox);
const queue = fs.readFileSync(path.join(root,'src/ui/static/js/queue.js'),'utf8');
vm.runInContext(queue.slice(queue.indexOf('function collectQueueUiSnapshot()'),queue.indexOf('function triggerControlEvent(')),sandbox);
function setup() {
  sandbox.clearStrategyGeneratedState(); controls.clear();
  sandbox.window.currentStrategyId=config.id;
  sandbox.window.currentStrategyConfig=config;
  sandbox.window.currentStrategyConfigId=config.id;
  for (const name of config.optimization_rules.parameter_tie_groups[0].pairs.flat()) {
    const spec=config.parameters[name];
    control(`backtest_${name}`,spec.default);
    control(`opt-${name}`,'','checkbox').checked=true;
    for (const [suffix,key] of [['from','min'],['to','max'],['step','step']]) control(`opt-${name}-${suffix}`,spec.optimize[key]);
  }
  control('grid-tie-symmetricLongShort','','checkbox');
}
const group=config.optimization_rules.parameter_tie_groups[0];
const read = id => controls.get(id);
const copy = value => JSON.parse(JSON.stringify(value));
setup();
const original=copy(sandbox.captureParameterTieFields(group));
assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),[]);
const invalidationsBefore=sandbox.invalidations;
sandbox.toggleParameterTie(group,true);
assert.equal(sandbox.invalidations,invalidationsBefore+1);
assert.equal(sandbox.previews,1);
assert.equal(read('backtest_closeCountShort').value,'7');
assert.equal(read('backtest_closeCountShort').disabled,true);
read('backtest_closeCountLong').value='4';
read('opt-closeCountLong-from').value='3';
sandbox.syncParameterTieControls();
assert.equal(read('backtest_closeCountShort').value,'4');
assert.equal(read('opt-closeCountShort-from').value,'3');
const item={strategyId:config.id,config:{grid_v2_enabled_tie_groups:['symmetricLongShort']},uiSnapshot:copy(sandbox.collectQueueUiSnapshot())};
assert.equal(item.uiSnapshot.parameterTies.groups.symmetricLongShort.fields.backtest_closeCountLong.value,'7');
for (let i=0;i<2;i++) {
  setup();
  sandbox.applyQueueUiSnapshot(item.uiSnapshot);
  sandbox.applyQueueParameterTies(item);
  assert.equal(read('backtest_closeCountLong').value,'4');
  assert.equal(read('backtest_closeCountShort').value,'4');
  sandbox.toggleParameterTie(group,false);
  assert.deepEqual(copy(sandbox.captureParameterTieFields(group)),original);
}
// Request selection overrides stale checkbox and restoration metadata.
setup(); sandbox.applyQueueUiSnapshot(item.uiSnapshot);
sandbox.applyQueueParameterTies({...item,config:{grid_v2_enabled_tie_groups:[]}});
assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),[]);
assert.equal(read('grid-tie-symmetricLongShort').checked,false);
sandbox.toggleParameterTie(group,true);
sandbox.toggleParameterTie(group,false);
sandbox.toggleParameterTie(group,true);
sandbox.toggleParameterTie(group,false);
// API item fallback values, without a snapshot, are captured before mirroring.
setup(); read('backtest_closeCountLong').value='3'; read('backtest_closeCountShort').value='6';
sandbox.applyQueueParameterTies({config:item.config});
sandbox.toggleParameterTie(group,false);
assert.equal(read('backtest_closeCountLong').value,'3');
assert.equal(read('backtest_closeCountShort').value,'6');
// Invalid snapshot membership cannot restore unrelated controls.
setup(); control('globalSetting','preserved');
const corrupt=copy(item); corrupt.uiSnapshot.parameterTies.groups.symmetricLongShort.fields.globalSetting={value:'bad'};
sandbox.applyQueueUiSnapshot(item.uiSnapshot); sandbox.applyQueueParameterTies(corrupt); sandbox.toggleParameterTie(group,false);
assert.equal(read('globalSetting').value,'preserved');
setup(); sandbox.toggleParameterTie(group,true); sandbox.applyQueueParameterTies({config:{}});
assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),[]);
sandbox.clearStrategyGeneratedState();
assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),[]);
// Restored source optimization flags also restore range editability.
setup(); read('opt-closeCountLong').checked=false;
sandbox.syncParameterTieControls();
sandbox.toggleParameterTie(group,true);
read('opt-closeCountLong').checked=true;
sandbox.syncParameterTieControls();
assert.equal(read('opt-closeCountLong-from').disabled,false);
sandbox.toggleParameterTie(group,false);
assert.equal(read('opt-closeCountLong').checked,false);
assert.equal(read('opt-closeCountLong-from').disabled,true);
assert(sandbox.invalidations>0);
assert(sandbox.previews>0);
console.log('Parameter ties: toggle/edit/restore, repeated Queue reload, config authority, fallback and reset passed.');

async function checkAsyncReload() {
  // Exercise real load boundaries, with rendering isolated from these state checks.
  sandbox.updateStrategyInfo=()=>{};
  sandbox.generateBacktestForm=()=>{};
  sandbox.generateOptimizerForm=()=>{};
  setup(); sandbox.toggleParameterTie(group,true);
  let resolveOld;
  sandbox.fetchStrategyConfig=()=>new Promise(resolve=>{resolveOld=resolve;});
  const oldLoad=sandbox.loadStrategyConfig(config.id);
  assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),[]);
  sandbox.fetchStrategyConfig=async()=>({...config,name:'Newest'});
  assert.equal(await sandbox.loadStrategyConfig(config.id),true);
  sandbox.toggleParameterTie(group,true);
  resolveOld({...config,name:'Stale same-ID response'});
  assert.equal(await oldLoad,false);
  assert.equal(sandbox.window.currentStrategyConfig.name,'Newest');
  assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),['symmetricLongShort']);
  sandbox.fetchStrategyConfig=async()=>null;
  assert.equal(await sandbox.loadStrategyConfig(config.id),false);
  assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),[]);
  setup(); sandbox.toggleParameterTie(group,true);
  sandbox.window.currentStrategyId='another-strategy';
  sandbox.fetchStrategyConfig=async()=>({...config,optimization_rules:{}});
  assert.equal(await sandbox.loadStrategyConfig('another-strategy'),true);
  assert.deepEqual(copy(sandbox.getEnabledParameterTieGroups()),[]);
  console.log('Parameter ties: async reload, stale same-ID response, empty load and strategy switch passed.');
}
checkAsyncReload().catch(error=>{console.error(error);process.exitCode=1;});

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repoRoot = path.resolve(__dirname, '..', '..');
const queuePath = path.join(repoRoot, 'src', 'ui', 'static', 'js', 'queue.js');
const source = fs.readFileSync(queuePath, 'utf8');

const storage = new Map();
let storageReads = 0;
let storageRemovals = 0;
let fetchCalls = 0;
let saveCalls = 0;
let clearCalls = 0;
let fetchOutcomes = [];
let wfaSyncCalls = 0;
const formElements = {};

const context = {
  console: {warn() {}, error() {}, log() {}},
  window: {},
  document: {
    getElementById(id) { return formElements[id] || null; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
  },
  setCheckboxValue(id, value) {
    formElements[id] ||= {checked: false, value: ''};
    formElements[id].checked = Boolean(value);
  },
  setInputValue(id, value) {
    formElements[id] ||= {checked: false, value: ''};
    formElements[id].value = value == null ? '' : String(value);
  },
  parseISOTimestamp() { return {date: '', time: ''}; },
  syncWfaModeUi() { wfaSyncCalls += 1; },
  localStorage: {
    getItem(key) {
      storageReads += 1;
      return storage.has(key) ? storage.get(key) : null;
    },
    removeItem(key) {
      storageRemovals += 1;
      storage.delete(key);
    },
  },
  async fetchQueueStateRequest() {
    fetchCalls += 1;
    const outcome = fetchOutcomes.shift();
    if (outcome instanceof Error) throw outcome;
    return outcome;
  },
  async saveQueueStateRequest(queueState) {
    saveCalls += 1;
    return JSON.parse(JSON.stringify(queueState));
  },
  async clearQueueStateRequest() {
    clearCalls += 1;
    return {items: [], nextIndex: 1, runtime: {active: false, updatedAt: 0}};
  },
  clonePreset(value) {
    return JSON.parse(JSON.stringify(value));
  },
};
vm.createContext(context);
vm.runInContext(
  `${source}\nthis.__queueTest = {
    appendQueueWarmupField,
    appendQueueWfaPeriodFields,
    applyQueueConfigFallback,
    buildQueueAutoSetModeLabel,
    buildQueueTooltip,
    buildStateForItem,
    generateQueueLabel,
    getQueueWfaPeriodFacts,
    ensureQueueStateLoaded,
    loadQueue,
    reset() {
      queueState = {items: [], nextIndex: 1, runtime: {active: false, updatedAt: 0}};
      queueStateLoaded = false;
      queueStateLoadPromise = null;
    },
    isLoaded() { return queueStateLoaded; }
  };`,
  context,
);

function sourceItem(id = 'q1') {
  return {
    id,
    index: 1,
    label: '#1',
    sources: [{type: 'path', path: 'C:\\data\\sample.csv'}],
    strategyId: 's03_reversal_v11_regime_er_b2',
    config: {
      optimization_mode: 'grid',
      grid_fast_objectives: ['sharpe_ratio', 'sqn', 'net_profit_pct'],
      grid_fast_primary_objective: 'sqn',
    },
  };
}

function captureFormData() {
  const entries = [];
  return {
    append(name, value) { entries.push([name, value]); },
    entries,
  };
}

async function main() {
  assert.equal(typeof context.fetchQueueStateRequest, 'function');
  assert.equal(typeof context.saveQueueStateRequest, 'function');
  assert.equal(typeof context.clearQueueStateRequest, 'function');

  const present = captureFormData();
  context.__queueTest.appendQueueWarmupField(present, {warmupBars: 1500});
  assert.deepEqual(present.entries, [['warmupBars', '1500']]);

  const missing = captureFormData();
  context.__queueTest.appendQueueWarmupField(missing, {});
  assert.deepEqual(missing.entries, []);

  const malformed = captureFormData();
  context.__queueTest.appendQueueWarmupField(malformed, {warmupBars: 'bad'});
  assert.deepEqual(malformed.entries, [['warmupBars', 'bad']]);

  const monthItem = sourceItem('month');
  monthItem.mode = 'wfa';
  monthItem.wfa = {
    periodUnit: 'months',
    isPeriodMonths: 2,
    oosPeriodMonths: 1,
    adaptiveMode: false,
  };
  const monthFields = captureFormData();
  context.__queueTest.appendQueueWfaPeriodFields(monthFields, monthItem);
  assert.deepEqual(monthFields.entries, [
    ['wf_period_unit', 'months'],
    ['wf_is_period_months', '2'],
    ['wf_oos_period_months', '1'],
  ]);
  assert.match(context.__queueTest.generateQueueLabel(monthItem), /WFA-F 2m\/1m/);
  assert.equal(context.__queueTest.buildQueueAutoSetModeLabel(monthItem), 'WFA-F 2m/1m');
  assert.match(context.__queueTest.buildQueueTooltip(monthItem), /IS: 2m, OOS: 1m/);
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.__queueTest.buildStateForItem(monthItem, 'running').wfa)),
    monthItem.wfa,
  );

  const legacyDayItem = sourceItem('day');
  legacyDayItem.mode = 'wfa';
  legacyDayItem.wfa = {isPeriodDays: 90, oosPeriodDays: 30, adaptiveMode: false};
  const dayFields = captureFormData();
  context.__queueTest.appendQueueWfaPeriodFields(dayFields, legacyDayItem);
  assert.deepEqual(dayFields.entries, [
    ['wf_is_period_days', '90'],
    ['wf_oos_period_days', '30'],
  ]);
  assert.match(context.__queueTest.generateQueueLabel(legacyDayItem), /WFA-F 90\/30/);

  const invalidPeriods = [undefined, null, 0, -5, 'not-a-number'];
  invalidPeriods.forEach((value) => {
    const invalidItem = sourceItem(`invalid-${String(value)}`);
    invalidItem.mode = 'wfa';
    invalidItem.wfa = {isPeriodDays: value, oosPeriodDays: value, adaptiveMode: false};
    const facts = context.__queueTest.getQueueWfaPeriodFacts(invalidItem);
    assert.equal(facts.isPeriod, '?');
    assert.equal(facts.oosPeriod, '?');
    assert.equal(facts.compact, '?/?');
    const invalidFields = captureFormData();
    context.__queueTest.appendQueueWfaPeriodFields(invalidFields, invalidItem);
    assert.deepEqual(invalidFields.entries, [
      ['wf_is_period_days', '?'],
      ['wf_oos_period_days', '?'],
    ]);
    assert.doesNotMatch(context.__queueTest.generateQueueLabel(invalidItem), /1\/1/);
  });

  const roundedDayItem = sourceItem('rounded-day');
  roundedDayItem.mode = 'wfa';
  roundedDayItem.wfa = {isPeriodDays: 90.4, oosPeriodDays: '30.6', adaptiveMode: false};
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.__queueTest.getQueueWfaPeriodFacts(roundedDayItem))),
    {unit: 'days', isPeriod: 90, oosPeriod: 31, compact: '90/31'},
  );

  formElements.wfCalendarMonths = {checked: true, value: ''};
  wfaSyncCalls = 0;
  const nonWfaLegacyItem = sourceItem('non-wfa-legacy');
  nonWfaLegacyItem.mode = 'optuna';
  context.__queueTest.applyQueueConfigFallback(nonWfaLegacyItem);
  assert.equal(formElements.wfCalendarMonths.checked, false);
  assert.equal(wfaSyncCalls, 1);

  wfaSyncCalls = 0;
  context.__queueTest.applyQueueConfigFallback(monthItem);
  assert.equal(formElements.wfCalendarMonths.checked, true);
  assert.equal(formElements.wfIsPeriodDays.value, '2');
  assert.equal(formElements.wfOosPeriodDays.value, '1');
  assert.equal(wfaSyncCalls, 1);

  context.__queueTest.reset();
  storage.clear();
  storageReads = 0;
  storageRemovals = 0;
  fetchCalls = 0;
  saveCalls = 0;
  const loadFailure = new Error('Stored Queue state is unreadable.');
  fetchOutcomes = [loadFailure, {items: [], nextIndex: 1, runtime: {active: false, updatedAt: 0}}];

  await assert.rejects(context.__queueTest.ensureQueueStateLoaded(), loadFailure);
  assert.equal(context.__queueTest.isLoaded(), false);
  assert.equal(storageReads, 0, 'legacy migration must not run after a failed GET');
  assert.equal(storageRemovals, 0);
  assert.equal(saveCalls, 0);
  assert.equal(fetchCalls, 1);

  await context.__queueTest.ensureQueueStateLoaded();
  assert.equal(context.__queueTest.isLoaded(), true);
  assert.equal(fetchCalls, 2, 'a later explicit call must retry GET');
  assert.equal(saveCalls, 0);

  context.__queueTest.reset();
  storage.clear();
  const legacyMonth = sourceItem('legacy');
  legacyMonth.mode = 'wfa';
  legacyMonth.wfa = {
    periodUnit: 'months',
    isPeriodMonths: 2,
    oosPeriodMonths: 1,
    adaptiveMode: false,
  };
  storage.set('merlinRunQueue', JSON.stringify({items: [legacyMonth], nextIndex: 2}));
  storage.set('merlinQueueRuntime', JSON.stringify({active: false, updatedAt: 0}));
  storageReads = 0;
  storageRemovals = 0;
  fetchOutcomes = [{items: [], nextIndex: 1, runtime: {active: false, updatedAt: 0}}];
  await context.__queueTest.ensureQueueStateLoaded();
  assert.equal(context.__queueTest.loadQueue().items[0].id, 'legacy');
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.__queueTest.loadQueue().items[0].config)),
    {
      optimization_mode: 'grid',
      grid_fast_objectives: ['sharpe_ratio', 'sqn', 'net_profit_pct'],
      grid_fast_primary_objective: 'sqn',
    },
  );
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.__queueTest.loadQueue().items[0].wfa)),
    legacyMonth.wfa,
  );
  assert.equal(saveCalls, 1, 'valid legacy state must still migrate after a successful empty GET');
  assert.equal(storage.size, 0);
  assert.equal(storageRemovals, 2);

  context.__queueTest.reset();
  storage.set('merlinRunQueue', JSON.stringify({items: [sourceItem('obsolete')]}));
  storage.set('merlinQueueRuntime', JSON.stringify({active: false, updatedAt: 0}));
  fetchOutcomes = [{items: [sourceItem('server')], nextIndex: 2, runtime: {active: false, updatedAt: 0}}];
  await context.__queueTest.ensureQueueStateLoaded();
  assert.equal(context.__queueTest.loadQueue().items[0].id, 'server');
  assert.equal(storage.size, 0, 'valid non-empty server state must outrank legacy storage');
  assert.equal(clearCalls, 0);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

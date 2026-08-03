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

const context = {
  console: {warn() {}, error() {}, log() {}},
  window: {},
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
};
vm.createContext(context);
vm.runInContext(
  `${source}\nthis.__queueTest = {
    appendQueueWarmupField,
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
  storage.set('merlinRunQueue', JSON.stringify({items: [sourceItem('legacy')], nextIndex: 2}));
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

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repoRoot = path.resolve(__dirname, '..', '..');
const source = fs.readFileSync(
  path.join(repoRoot, 'src', 'ui', 'static', 'js', 'ui-handlers.js'),
  'utf8',
);

function sourceSlice(startSignature, endSignature) {
  const start = source.indexOf(startSignature);
  const end = source.indexOf(endSignature, start);
  assert.notEqual(start, -1, `Missing ${startSignature}`);
  assert.notEqual(end, -1, `Missing ${endSignature}`);
  return source.slice(start, end);
}

function checkbox(objective) {
  return {
    checked: true,
    disabled: false,
    dataset: {objective},
    classList: {contains(name) { return name === 'grid-fast-objective-checkbox'; }},
  };
}

const objectives = [
  'net_profit_pct',
  'max_drawdown_pct',
  'romad',
  'profit_factor',
  'win_rate',
  'sharpe_ratio',
  'sqn',
];
const checkboxes = objectives.map(checkbox);
const alerts = [];
const context = {
  document: {
    querySelectorAll(selector) {
      return selector === '.grid-fast-objective-checkbox' ? checkboxes : [];
    },
    getElementById() { return null; },
  },
  alert(message) { alerts.push(message); },
};
vm.createContext(context);
vm.runInContext(
  sourceSlice('const GRID_SUPPORTED_OBJECTIVES', 'const GRID_SUPPORTED_CONSTRAINTS')
    + sourceSlice('function getGridObjectiveElements', 'function getEnabledGridMetadata')
    + '\nthis.testApi = {collectGridObjectiveSelection, enforceGridFastObjectiveLimit};',
  context,
);

assert.equal(context.testApi.enforceGridFastObjectiveLimit(checkboxes[6]), false);
assert.equal(checkboxes[6].checked, false);
assert.deepEqual(alerts, ['Select no more than 6 Grid fast objectives.']);
assert.equal(context.testApi.collectGridObjectiveSelection('fast').objectives.length, 6);

checkboxes[5].checked = false;
checkboxes[6].checked = true;
assert.equal(context.testApi.enforceGridFastObjectiveLimit(checkboxes[6]), true);
assert.equal(checkboxes[6].checked, true);
assert.equal(alerts.length, 1);

console.log('Grid Fast objective JavaScript behavior passed.');

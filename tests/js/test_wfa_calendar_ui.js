'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repoRoot = path.resolve(__dirname, '..', '..');
const uiSource = fs.readFileSync(
  path.join(repoRoot, 'src', 'ui', 'static', 'js', 'ui-handlers.js'),
  'utf8',
);
const previewSource = fs.readFileSync(
  path.join(repoRoot, 'src', 'ui', 'static', 'js', 'dataset-preview.js'),
  'utf8',
);
const resultsTablesSource = fs.readFileSync(
  path.join(repoRoot, 'src', 'ui', 'static', 'js', 'results-tables.js'),
  'utf8',
);
const analyticsSource = fs.readFileSync(
  path.join(repoRoot, 'src', 'ui', 'static', 'js', 'analytics.js'),
  'utf8',
);

function element(values = {}) {
  return {
    checked: false,
    disabled: false,
    value: '',
    min: '',
    max: '',
    step: '',
    textContent: '',
    innerHTML: '',
    style: {display: ''},
    dataset: {},
    ...values,
  };
}

const elements = {
  enableWF: element({checked: true}),
  wfSettings: element(),
  enableAdaptiveWF: element(),
  wfCalendarMonths: element(),
  adaptiveWFSettings: element(),
  wfIsPeriodDays: element({value: '90'}),
  wfOosPeriodDays: element({value: '30'}),
  wfIsPeriodLabel: element(),
  wfOosPeriodLabel: element(),
  wfCooldownEnabled: element(),
  wfCooldownDays: element({value: '15'}),
  wfStoreTopNTrials: element({value: '77'}),
  gridBudget: element({value: 'global-grid-sentinel'}),
  datasetPreview: element(),
  dateFilter: element({checked: true}),
  startDate: element({value: '2025-10-01'}),
  endDate: element({value: '2026-08-01'}),
  enablePostProcess: element(),
  enableOosTest: element(),
  ftPeriodDays: element({value: '30'}),
  oosPeriodDays: element({value: '30'}),
};

const context = {
  console: {warn() {}, error() {}, log() {}},
  document: {
    getElementById(id) { return elements[id] || null; },
    querySelectorAll() { return []; },
  },
  window: {},
  setTimeout,
  clearTimeout,
  Date,
};
context.window = context;
vm.createContext(context);
vm.runInContext(
  `${uiSource}\nthis.__wfaUiTest = {syncWfaModeUi, buildWfaPeriodState, appendWfaPeriodFields};`,
  context,
);
vm.runInContext(previewSource, context);

function formCapture() {
  const entries = [];
  return {entries, append(name, value) { entries.push([name, String(value)]); }};
}

context.__wfaUiTest.syncWfaModeUi();
assert.equal(elements.wfIsPeriodDays.value, '90');
assert.equal(elements.wfOosPeriodDays.value, '30');
assert.equal(elements.wfIsPeriodDays.min, '30');
assert.equal(elements.wfOosPeriodDays.min, '15');

elements.enableAdaptiveWF.checked = true;
context.__wfaUiTest.syncWfaModeUi();
assert.equal(elements.wfOosPeriodDays.disabled, true);
assert.equal(elements.adaptiveWFSettings.style.display, 'block');

elements.wfCalendarMonths.checked = true;
context.__wfaUiTest.syncWfaModeUi();
assert.equal(elements.wfIsPeriodDays.value, '2');
assert.equal(elements.wfOosPeriodDays.value, '1');
assert.equal(elements.wfIsPeriodLabel.textContent, 'In-Sample (months):');
assert.equal(elements.wfOosPeriodLabel.textContent, 'Out-of-Sample (months):');
assert.equal(elements.wfIsPeriodDays.min, '1');
assert.equal(elements.wfIsPeriodDays.max, '120');
assert.equal(elements.wfOosPeriodDays.step, '1');
assert.equal(elements.enableAdaptiveWF.checked, false);
assert.equal(elements.enableAdaptiveWF.disabled, true);
assert.equal(elements.wfOosPeriodDays.disabled, false);
assert.equal(elements.adaptiveWFSettings.style.display, 'none');

elements.wfIsPeriodDays.value = '4';
elements.wfOosPeriodDays.value = '2';
elements.wfCalendarMonths.checked = false;
context.__wfaUiTest.syncWfaModeUi();
assert.equal(elements.wfIsPeriodDays.value, '90');
assert.equal(elements.wfOosPeriodDays.value, '30');
assert.equal(elements.enableAdaptiveWF.disabled, false);
elements.wfIsPeriodDays.value = '60';
elements.wfOosPeriodDays.value = '20';
elements.wfCalendarMonths.checked = true;
context.__wfaUiTest.syncWfaModeUi();
assert.equal(elements.wfIsPeriodDays.value, '4');
assert.equal(elements.wfOosPeriodDays.value, '2');
elements.enableWF.checked = false;
context.__wfaUiTest.syncWfaModeUi();
assert.equal(elements.wfSettings.style.display, 'none');
assert.equal(elements.enableAdaptiveWF.disabled, true);
elements.enableWF.checked = true;
elements.wfCalendarMonths.checked = false;
context.__wfaUiTest.syncWfaModeUi();
assert.equal(elements.wfIsPeriodDays.value, '60');
assert.equal(elements.wfOosPeriodDays.value, '20');
assert.equal(elements.enableAdaptiveWF.checked, false);

assert.deepEqual(
  JSON.parse(JSON.stringify(context.__wfaUiTest.buildWfaPeriodState('months', '2', '1'))),
  {periodUnit: 'months', isPeriodMonths: 2, oosPeriodMonths: 1},
);
const transport = formCapture();
context.__wfaUiTest.appendWfaPeriodFields(transport, 'months', '2', '1');
assert.deepEqual(transport.entries, [
  ['wf_period_unit', 'months'],
  ['wf_is_period_months', '2'],
  ['wf_oos_period_months', '1'],
]);

elements.wfCalendarMonths.checked = true;
elements.wfIsPeriodDays.value = '2';
elements.wfOosPeriodDays.value = '1';
context.updateDatasetPreview();
assert.doesNotMatch(elements.datasetPreview.innerHTML, /Insufficient|Preview Error/);
assert.match(elements.datasetPreview.innerHTML, /W8/);
assert.match(elements.datasetPreview.innerHTML, /08\.01/);

elements.startDate.value = '2025-09-28';
context.updateDatasetPreview();
assert.match(elements.datasetPreview.innerHTML, /Preview Error/);

elements.startDate.value = '2025-01-01';
elements.endDate.value = '2025-03-31';
elements.wfCalendarMonths.checked = false;
elements.wfIsPeriodDays.value = '10';
elements.wfOosPeriodDays.value = '5';
context.updateDatasetPreview();
assert.doesNotMatch(elements.datasetPreview.innerHTML, /Insufficient|Preview Error/);
assert.match(elements.datasetPreview.innerHTML, /W16/);
assert.equal(elements.wfStoreTopNTrials.value, '77');
assert.equal(elements.gridBudget.value, 'global-grid-sentinel');

const displayContext = {
  console: {warn() {}, error() {}, log() {}},
  document: {
    addEventListener() {},
    getElementById() { return null; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
  },
  window: {},
  localStorage: {getItem() { return null; }, setItem() {}, removeItem() {}},
  sessionStorage: {getItem() { return null; }, setItem() {}, removeItem() {}},
  URLSearchParams,
  Date,
};
displayContext.window = displayContext;
vm.createContext(displayContext);
vm.runInContext(
  `${resultsTablesSource}\nthis.__resolveWfaPeriodDisplay = resolveWfaPeriodDisplay;`,
  displayContext,
);
vm.runInContext(
  `${analyticsSource}\nthis.__resolveAnalyticsWfaPeriods = resolveAnalyticsWfaPeriods;`,
  displayContext,
);
assert.deepEqual(
  JSON.parse(JSON.stringify(displayContext.__resolveWfaPeriodDisplay({
    periodUnit: 'months', isPeriodMonths: 2, oosPeriodMonths: 1,
  }))),
  {periodUnit: 'months', isPeriod: 2, oosPeriod: 1},
);
assert.deepEqual(
  JSON.parse(JSON.stringify(displayContext.__resolveWfaPeriodDisplay({
    isPeriodDays: 90, oosPeriodDays: 30,
  }))),
  {periodUnit: 'days', isPeriod: 90, oosPeriod: 30},
);
assert.deepEqual(
  JSON.parse(JSON.stringify(displayContext.__resolveAnalyticsWfaPeriods({
    period_unit: 'months', is_period_months: 2, oos_period_months: 1,
  }))),
  {periodUnit: 'months', isPeriod: 2, oosPeriod: 1},
);
assert.deepEqual(
  JSON.parse(JSON.stringify(displayContext.__resolveAnalyticsWfaPeriods({
    is_period_days: 90, oos_period_days: 30,
  }))),
  {periodUnit: 'days', isPeriod: 90, oosPeriod: 30},
);

console.log('WFA calendar UI and preview behavior passed');

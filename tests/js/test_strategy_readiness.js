'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repoRoot = path.resolve(__dirname, '..', '..');
const source = fs.readFileSync(
  path.join(repoRoot, 'src', 'ui', 'static', 'js', 'strategy-config.js'),
  'utf8',
);
const uiHandlersSource = fs.readFileSync(
  path.join(repoRoot, 'src', 'ui', 'static', 'js', 'ui-handlers.js'),
  'utf8',
);

function functionSource(fullSource, signature, nextSignature) {
  const start = fullSource.indexOf(signature);
  const end = fullSource.indexOf(nextSignature, start);
  assert.notEqual(start, -1, `Missing ${signature}`);
  assert.notEqual(end, -1, `Missing ${nextSignature}`);
  return fullSource.slice(start, end);
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return {promise, resolve, reject};
}

function element(value = '') {
  return {
    innerHTML: value,
    textContent: value,
    value,
    checked: false,
    disabled: false,
    dataset: {},
    style: {display: 'block'},
    classList: {add() {}, remove() {}, toggle() {}},
    addEventListener() {},
    removeEventListener() {},
  };
}

const elements = new Map();
[
  'backtestParamsContent', 'optimizerParamsContainer', 'strategyInfo',
  'strategyName', 'strategyVersion', 'strategyDescription', 'strategyParamCount',
  'gridPreviewSummary', 'gridPreviewRows', 'gridPreviewError',
  'gridEnabledModes', 'gridV2ManualAllocation',
  'optimizerModeGrid', 'optimizerModeOptuna', 'gridSettings', 'optunaSettings', 'gridModeHelp',
  'dateFilter', 'startDate', 'startTime', 'endDate', 'endTime', 'warmupBars',
  'csvDirectory', 'dbTarget', 'gridBudget', 'gridSeed', 'gridV2PlanningPolicy',
  'gridAllocAuto', 'gridFastPrimaryObjective', 'wfIsPeriodDays',
].forEach((id) => elements.set(id, element(id)));

elements.get('optimizerModeGrid').value = 'grid';
elements.get('optimizerModeGrid').checked = true;
elements.get('optimizerModeOptuna').value = 'optuna';
elements.get('gridEnabledModes').dataset.profileKey = 'stale-profile';

const alerts = [];
const warnings = [];
const coverageConfigs = [];
const renderReadiness = [];
const context = {
  window: {
    lastGridPreview: {stale: true},
    lastGridPreviewConfigKey: 'stale',
    OptunaUI: {
      updateCoverageInfo() {
        coverageConfigs.push(context.window.currentStrategyConfig);
        renderReadiness.push(context.__readinessTest.isCurrentStrategyConfigReady());
      },
    },
  },
  document: {
    getElementById(id) { return elements.get(id) || null; },
    querySelector(selector) {
      if (selector === 'input[name="optimizerMode"]:checked') {
        return elements.get('optimizerModeGrid').checked
          ? elements.get('optimizerModeGrid')
          : elements.get('optimizerModeOptuna');
      }
      return null;
    },
    querySelectorAll() { return []; },
  },
  alert(message) { alerts.push(message); },
  console: {
    log() {},
    error() {},
    warn(...args) { warnings.push(args.join(' ')); },
  },
  async fetchStrategyConfig() { throw new Error('not configured'); },
  resetGridPreviewState() {
    context.window.lastGridPreview = null;
    context.window.lastGridPreviewConfigKey = null;
    elements.get('gridPreviewSummary').textContent = '';
    elements.get('gridPreviewRows').innerHTML = '';
    elements.get('gridPreviewError').textContent = '';
    elements.get('gridPreviewError').style.display = 'none';
  },
  syncGridBudgetHelp() {},
  syncGridProfileUi() {},
  syncGridAllocationUi() {},
  syncGridParameterOptions() {},
  syncGridObjectiveUi() {},
  syncGridObjectiveAndConstraintUi() {},
  scheduleGridPreviewUpdate() {},
  handleOptimizerCheckboxChange() {},
};

vm.createContext(context);
vm.runInContext(
  `${source}\n${functionSource(uiHandlersSource, 'function getOptimizerMode()', 'function parseCompactCount')}`
    + `\n${functionSource(uiHandlersSource, 'function getEnabledGridMetadata()', 'function isFullEnumerationProfile')}`
    + `\n${functionSource(uiHandlersSource, 'function syncOptimizerModeUI()', 'async function submitOptimization')}`
    + `\n${functionSource(uiHandlersSource, 'function bindOptimizerInputs()', 'function handleOptimizerCheckboxChange')}`
    + '\nthis.__readinessTest = {'
    + 'loadStrategyConfig, isCurrentStrategyConfigReady, syncOptimizerModeUI, bindOptimizerInputs, getOptimizerMode};',
  context,
);

context.updateStrategyInfo = (config) => {
  renderReadiness.push(context.__readinessTest.isCurrentStrategyConfigReady());
  elements.get('strategyInfo').style.display = 'block';
  elements.get('strategyName').textContent = config.name;
};
context.generateBacktestForm = (config) => {
  renderReadiness.push(context.__readinessTest.isCurrentStrategyConfigReady());
  if (config.throwDuringRender) throw new Error('render failed');
  elements.get('backtestParamsContent').innerHTML = `backtest:${config.name}`;
};
context.generateOptimizerForm = (config) => {
  renderReadiness.push(context.__readinessTest.isCurrentStrategyConfigReady());
  elements.get('optimizerParamsContainer').innerHTML = `optimizer:${config.name}`;
  if (config.replaceDuringRender) {
    context.window.currentStrategyId = 'newer';
    context.window.currentStrategyConfig = config.replaceDuringRender;
    context.window.currentStrategyConfigId = 'newer';
    return;
  }
  context.__readinessTest.bindOptimizerInputs();
};

const preservedIds = [
  'dateFilter', 'startDate', 'startTime', 'endDate', 'endTime', 'warmupBars',
  'csvDirectory', 'dbTarget', 'gridBudget', 'gridSeed', 'gridV2PlanningPolicy',
  'gridAllocAuto', 'gridFastPrimaryObjective', 'wfIsPeriodDays',
];
preservedIds.forEach((id, index) => {
  const sentinel = `preserved-${index}-${id}`;
  elements.get(id).value = sentinel;
  elements.get(id).checked = index % 2 === 0;
});
const preservedBefore = Object.fromEntries(preservedIds.map((id) => [id, {
  value: elements.get(id).value,
  checked: elements.get(id).checked,
}]));
const config = (name, extras = {}) => ({
  name,
  parameters: {period: {type: 'int'}},
  grid_optimizer: {supported: true, available: true, profile: 'full_enumeration_v2', modes: []},
  ...extras,
});

async function load(strategyId, outcome) {
  context.window.currentStrategyId = strategyId;
  context.fetchStrategyConfig = async () => outcome;
  return context.__readinessTest.loadStrategyConfig(strategyId);
}

async function main() {
  context.window.currentStrategyConfig = null;
  context.__readinessTest.syncOptimizerModeUI();
  assert.equal(context.__readinessTest.getOptimizerMode(), 'grid');

  const gridBeforeReload = context.__readinessTest.getOptimizerMode();
  const configA = config('A');
  assert.equal(await load('a', configA), true);
  assert.equal(context.__readinessTest.isCurrentStrategyConfigReady(), true);
  assert.equal(context.window.currentStrategyConfigId, 'a');
  assert.equal(elements.get('backtestParamsContent').innerHTML, 'backtest:A');
  assert.equal(context.__readinessTest.getOptimizerMode(), 'grid');
  assert.equal(elements.get('optimizerModeGrid').disabled, false);
  assert.equal(coverageConfigs.includes(configA), true);
  assert.equal(renderReadiness.every((ready) => ready === false), true);

  elements.get('gridEnabledModes').innerHTML = 'stale-grid-modes';
  elements.get('gridEnabledModes').dataset.profileKey = 'stale-profile';
  elements.get('gridV2ManualAllocation').innerHTML = 'stale-manual-allocation';

  const failing = deferred();
  context.window.currentStrategyId = 'b';
  context.fetchStrategyConfig = () => failing.promise;
  const failingLoad = context.__readinessTest.loadStrategyConfig('b');
  assert.equal(context.__readinessTest.isCurrentStrategyConfigReady(), false);
  assert.equal(elements.get('backtestParamsContent').innerHTML, '');
  assert.equal(elements.get('optimizerParamsContainer').innerHTML, '');
  assert.equal(elements.get('gridEnabledModes').innerHTML, '');
  assert.equal(Object.hasOwn(elements.get('gridEnabledModes').dataset, 'profileKey'), false);
  assert.equal(elements.get('gridV2ManualAllocation').innerHTML, '');
  assert.equal(elements.get('strategyInfo').style.display, 'none');
  assert.equal(context.window.lastGridPreview, null);
  failing.reject(new Error('Profile is invalid.'));
  assert.equal(await failingLoad, false);
  assert.equal(alerts.pop(), 'Profile is invalid.');

  assert.equal(await load('c', config('C')), true);
  assert.equal(context.__readinessTest.isCurrentStrategyConfigReady(), true);

  const staleSuccess = deferred();
  context.window.currentStrategyId = 'b';
  context.fetchStrategyConfig = () => staleSuccess.promise;
  const obsoleteSuccessLoad = context.__readinessTest.loadStrategyConfig('b');
  assert.equal(await load('c', config('C current')), true);
  staleSuccess.resolve(config('B stale'));
  assert.equal(await obsoleteSuccessLoad, false);
  assert.equal(context.window.currentStrategyConfig.name, 'C current');
  assert.equal(elements.get('backtestParamsContent').innerHTML, 'backtest:C current');

  const staleFailure = deferred();
  context.window.currentStrategyId = 'b';
  context.fetchStrategyConfig = () => staleFailure.promise;
  const obsoleteFailureLoad = context.__readinessTest.loadStrategyConfig('b');
  assert.equal(await load('c', config('C newest')), true);
  const alertCount = alerts.length;
  staleFailure.reject(new Error('obsolete failure'));
  assert.equal(await obsoleteFailureLoad, false);
  assert.equal(context.window.currentStrategyConfig.name, 'C newest');
  assert.equal(alerts.length, alertCount);

  assert.equal(await load('render', config('Broken', {throwDuringRender: true})), false);
  assert.equal(context.__readinessTest.isCurrentStrategyConfigReady(), false);
  assert.equal(context.window.currentStrategyConfig, null);
  assert.equal(context.window.currentStrategyConfigId, null);
  assert.equal(elements.get('backtestParamsContent').innerHTML, '');
  assert.equal(alerts.pop(), 'render failed');

  const newerConfig = config('Newer during render');
  assert.equal(await load('old', config('Old', {replaceDuringRender: newerConfig})), false);
  assert.equal(context.window.currentStrategyId, 'newer');
  assert.equal(context.window.currentStrategyConfig, newerConfig);
  assert.equal(context.window.currentStrategyConfigId, 'newer');

  assert.equal(await load('warning', config('Warning', {
    validation_warnings: ['Review this setting.'],
    diagnostics: [{severity: 'info', message: 'secondary detail'}],
  })), true);
  assert.equal(context.__readinessTest.isCurrentStrategyConfigReady(), true);
  assert.equal(warnings.some((message) => message.includes('Review this setting.')), true);
  assert.equal(alerts.some((message) => message.includes('secondary detail')), false);

  assert.deepEqual(
    Object.fromEntries(preservedIds.map((id) => [id, {
      value: elements.get(id).value,
      checked: elements.get(id).checked,
    }])),
    preservedBefore,
  );

  console.log(`Grid before config reload: ${gridBeforeReload === 'grid' ? 'Grid' : 'Optuna'}`);
  console.log(`Grid after config reload:  ${context.__readinessTest.getOptimizerMode() === 'grid' ? 'Grid' : 'Optuna'}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

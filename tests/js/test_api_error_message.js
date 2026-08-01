'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repoRoot = path.resolve(__dirname, '..', '..');
const apiPath = path.join(repoRoot, 'src', 'ui', 'static', 'js', 'api.js');
const source = fs.readFileSync(apiPath, 'utf8');
const resultsFormatPath = path.join(
  repoRoot,
  'src',
  'ui',
  'static',
  'js',
  'results-format.js',
);
const resultsControllerPath = path.join(
  repoRoot,
  'src',
  'ui',
  'static',
  'js',
  'results-controller.js',
);
const context = {
  console,
  FormData,
  URLSearchParams,
};
vm.createContext(context);
vm.runInContext(
  `${source}\nthis.__apiTest = {fetchStrategyConfig, readApiErrorMessage, runBacktestRequest, downloadBacktestTradesRequest, runOptimizationRequest};`,
  context,
);

function textResponse(body, ok = false) {
  let reads = 0;
  return {
    ok,
    async text() {
      reads += 1;
      return body;
    },
    get reads() {
      return reads;
    },
  };
}

async function main() {
  const {readApiErrorMessage} = context.__apiTest;

  const structured = textResponse(JSON.stringify({
    error: 'Concise validation message.',
    diagnostics: [{code: 'V2_INVALID_RUNTIME_VALUE'}],
  }));
  assert.equal(
    await readApiErrorMessage(structured, 'fallback'),
    'Concise validation message.',
  );
  assert.equal(structured.reads, 1);

  assert.equal(
    await readApiErrorMessage(textResponse('Legacy plain-text failure.'), 'fallback'),
    'Legacy plain-text failure.',
  );
  assert.equal(
    await readApiErrorMessage(textResponse('{malformed json'), 'fallback'),
    '{malformed json',
  );
  assert.equal(await readApiErrorMessage(textResponse(''), 'fallback'), 'fallback');
  const jsonWithoutError = JSON.stringify({diagnostics: []});
  assert.equal(
    await readApiErrorMessage(textResponse(jsonWithoutError), 'fallback'),
    jsonWithoutError,
  );

  const clients = [
    ['runBacktestRequest', 'Backtest request failed.'],
    ['downloadBacktestTradesRequest', 'Backtest trade export failed.'],
    ['runOptimizationRequest', 'Optimization request failed.'],
  ];
  for (const [name, fallback] of clients) {
    const response = textResponse(JSON.stringify({
      error: `${name} concise`,
      diagnostics: [{code: 'V2_INVALID_RUNTIME_VALUE'}],
    }));
    context.fetch = async () => response;
    await assert.rejects(
      context.__apiTest[name]({}),
      (error) => error.message === `${name} concise`,
      `${name} must use the shared JSON-aware reader instead of ${fallback}`,
    );
    assert.equal(response.reads, 1);
  }

  for (const body of [
    JSON.stringify({error: 'Profile is invalid.', diagnostics: []}),
    'Unknown strategy.',
    '',
  ]) {
    const response = textResponse(body);
    response.status = body ? 422 : 404;
    response.statusText = 'Failure';
    context.fetch = async () => response;
    await assert.rejects(
      context.__apiTest.fetchStrategyConfig('broken'),
      (error) => error.message === (
        body === JSON.stringify({error: 'Profile is invalid.', diagnostics: []})
          ? 'Profile is invalid.'
          : body || 'Failed to load strategy configuration.'
      ),
    );
    assert.equal(response.reads, 1);
  }

  const successfulConfig = {name: 'Ready', parameters: {period: {type: 'int'}}};
  context.fetch = async () => ({
    ok: true,
    async json() { return successfulConfig; },
  });
  assert.equal(
    await context.__apiTest.fetchStrategyConfig('ready'),
    successfulConfig,
  );

  const intendedClients = [
    'fetchStrategyConfig',
    'runBacktestRequest',
    'downloadBacktestTradesRequest',
    'runOptimizationRequest',
  ];
  intendedClients.forEach((name) => {
    const start = source.indexOf(`async function ${name}(`);
    assert.notEqual(start, -1, `${name} must exist`);
    const nextFunction = source.indexOf('\nasync function ', start + 1);
    const functionSource = source.slice(start, nextFunction === -1 ? source.length : nextFunction);
    assert.match(functionSource, /await readApiErrorMessage\(/, `${name} must use the shared reader`);
  });

  const resultsControllerSource = fs.readFileSync(resultsControllerPath, 'utf8');
  assert.match(
    resultsControllerSource,
    /readApiErrorMessage\(response, 'Trade export failed\.'\)/,
  );
  assert.match(
    resultsControllerSource,
    /readApiErrorMessage\(response, 'Bundle export failed\.'\)/,
  );
  assert.equal(
    (resultsControllerSource.match(/const message = await response\.text\(\);/g) || []).length,
    0,
  );

  const resultsSource = fs.readFileSync(resultsFormatPath, 'utf8');
  const resultsContext = {};
  vm.createContext(resultsContext);
  vm.runInContext(
    `${resultsSource}\nthis.__paramIdTest = {canonicalizeParamIdParams, PARAM_ID_IGNORED_KEYS};`,
    resultsContext,
  );
  const ignored = resultsContext.__paramIdTest.PARAM_ID_IGNORED_KEYS;
  for (const name of ['dateFilter', 'start', 'end', 'warmupBars']) {
    assert.equal(ignored.has(name), true);
  }
  const raw = resultsContext.__paramIdTest.canonicalizeParamIdParams({
    maType: 'EMA',
    maLength: 50,
    dateFilter: true,
    start: '2025-06-01',
    end: '2025-06-30',
    warmupBars: 1000,
  });
  const canonical = resultsContext.__paramIdTest.canonicalizeParamIdParams({
    maType: 'EMA',
    maLength: 50,
    dateFilter: false,
    start: '2025-06-01T00:00:00Z',
    end: '2025-06-30T23:59:59.999999Z',
    warmupBars: 5000,
  });
  assert.equal(JSON.stringify(raw), JSON.stringify(canonical));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

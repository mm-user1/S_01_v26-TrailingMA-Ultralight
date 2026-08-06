(function () {
  const DAY_MS = 24 * 60 * 60 * 1000;
  const INSUFFICIENT_TEXT = 'Insufficient data for min 2 WFA windows';
  const GENERIC_ERROR_TEXT = 'Preview Error';

  function addDays(date, days) {
    const copy = new Date(date.getTime());
    copy.setUTCDate(copy.getUTCDate() + days);
    return copy;
  }

  function addMonths(date, months) {
    return new Date(Date.UTC(
      date.getUTCFullYear(),
      date.getUTCMonth() + months,
      1
    ));
  }

  function daysBetween(startDate, endDate) {
    return Math.round((endDate.getTime() - startDate.getTime()) / DAY_MS);
  }

  function fmtDate(date) {
    const month = String(date.getUTCMonth() + 1).padStart(2, '0');
    const day = String(date.getUTCDate()).padStart(2, '0');
    return `${month}.${day}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function parseDateInput(rawValue) {
    const trimmed = String(rawValue || '').trim();
    if (!trimmed) return null;

    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(trimmed);
    if (!match) return null;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const parsed = new Date(Date.UTC(year, month - 1, day));
    if (
      parsed.getUTCFullYear() !== year
      || parsed.getUTCMonth() !== (month - 1)
      || parsed.getUTCDate() !== day
    ) {
      return null;
    }

    return parsed;
  }

  function parsePositiveInt(rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) return null;
    const rounded = Math.round(parsed);
    return rounded > 0 ? rounded : null;
  }

  function readPositiveInt(id, fallback) {
    const element = document.getElementById(id);
    if (!element) return fallback;
    return parsePositiveInt(element.value);
  }

  function isCheckboxActive(element) {
    return Boolean(element && element.checked && !element.disabled);
  }

  function modeHasFt(mode) {
    return mode === 'optuna-ft' || mode === 'optuna-ft-oos' || mode === 'wfa-fixed-ft' || mode === 'wfa-adaptive-ft';
  }

  function modeHasOptunaOos(mode) {
    return mode === 'optuna-oos' || mode === 'optuna-ft-oos';
  }

  function modeIsWfa(mode) {
    return mode === 'wfa-fixed' || mode === 'wfa-fixed-ft' || mode === 'wfa-adaptive' || mode === 'wfa-adaptive-ft';
  }

  function modeIsAdaptive(mode) {
    return mode === 'wfa-adaptive' || mode === 'wfa-adaptive-ft';
  }

  function detectMode() {
    const wfEnabled = isCheckboxActive(document.getElementById('enableWF'));
    const adaptiveEnabled = wfEnabled && isCheckboxActive(document.getElementById('enableAdaptiveWF'));
    const ftEnabled = isCheckboxActive(document.getElementById('enablePostProcess'));
    const oosEnabled = isCheckboxActive(document.getElementById('enableOosTest'));

    if (wfEnabled) {
      if (adaptiveEnabled) {
        return ftEnabled ? 'wfa-adaptive-ft' : 'wfa-adaptive';
      }
      return ftEnabled ? 'wfa-fixed-ft' : 'wfa-fixed';
    }

    if (ftEnabled && oosEnabled) return 'optuna-ft-oos';
    if (ftEnabled) return 'optuna-ft';
    if (oosEnabled) return 'optuna-oos';
    return 'optuna';
  }

  function calcWFAWindows(startDate, endDate, isDays, oosDays) {
    if (!isDays || !oosDays) {
      throw new Error('invalid_wfa_periods');
    }

    const windows = [];
    let current = new Date(startDate.getTime());
    const endNorm = addDays(endDate, 1);

    while (true) {
      const isEnd = addDays(current, isDays);
      const oosEnd = addDays(isEnd, oosDays);
      if (oosEnd > endNorm) break;

      windows.push({
        isStart: new Date(current.getTime()),
        isEnd,
        oosStart: new Date(isEnd.getTime()),
        oosEnd
      });

      current = addDays(current, oosDays);
    }

    return windows;
  }

  function calcCalendarMonthWindows(startDate, endDate, isMonths, oosMonths) {
    if (!isMonths || !oosMonths || startDate.getUTCDate() !== 1) {
      throw new Error('invalid_calendar_month_periods');
    }
    const windows = [];
    const endExclusive = addDays(endDate, 1);
    for (let windowIndex = 0; ; windowIndex += 1) {
      const isStart = addMonths(startDate, windowIndex * oosMonths);
      const isEnd = addMonths(isStart, isMonths);
      const oosEnd = addMonths(isEnd, oosMonths);
      if (oosEnd > endExclusive) break;
      windows.push({
        isStart,
        isEnd,
        oosStart: new Date(isEnd.getTime()),
        oosEnd
      });
    }
    return windows;
  }

  function calcOptunaPeriods(startDate, endDate, ftEnabled, ftDays, oosEnabled, oosDays) {
    const totalDays = daysBetween(startDate, endDate);
    if (totalDays <= 0) {
      throw new Error('invalid_range');
    }

    let oosStart = null;
    let oosEnd = null;
    let ftStart = null;
    let ftEnd = null;

    if (oosEnabled) {
      if (!oosDays || oosDays >= totalDays) {
        throw new Error('invalid_oos_days');
      }
      oosStart = addDays(endDate, -oosDays);
      oosEnd = new Date(endDate.getTime());
    }

    if (ftEnabled) {
      if (!ftDays) {
        throw new Error('invalid_ft_days');
      }
      const remainingEnd = oosEnabled ? oosStart : endDate;
      const remainingDays = daysBetween(startDate, remainingEnd);
      if (ftDays >= remainingDays) {
        throw new Error('invalid_ft_range');
      }
      ftEnd = new Date(remainingEnd.getTime());
      ftStart = addDays(ftEnd, -ftDays);
    }

    const isEnd = ftEnabled ? ftStart : (oosEnabled ? oosStart : new Date(endDate.getTime()));
    if (!isEnd || daysBetween(startDate, isEnd) <= 0) {
      throw new Error('invalid_is_range');
    }

    return { isEnd, ftStart, ftEnd, oosStart, oosEnd };
  }

  function pushSegment(segments, type, days) {
    const safeDays = Math.round(days);
    if (safeDays <= 0) {
      return;
    }
    segments.push({ type, days: safeDays });
  }

  function carveFtDays(isPeriodDays, ftDays) {
    if (!ftDays || isPeriodDays <= 1) {
      return { isDays: isPeriodDays, ftDays: 0 };
    }
    const carvedFtDays = Math.min(ftDays, isPeriodDays - 1);
    return {
      isDays: isPeriodDays - carvedFtDays,
      ftDays: carvedFtDays
    };
  }

  function buildSegments(config) {
    const { mode, startDate, endDate, isDays, oosDays, ftDays, periodUnit } = config;
    const segments = [];

    if (!modeIsWfa(mode)) {
      const periods = calcOptunaPeriods(
        startDate,
        endDate,
        modeHasFt(mode),
        ftDays,
        modeHasOptunaOos(mode),
        oosDays
      );

      pushSegment(segments, 'is', daysBetween(startDate, periods.isEnd));
      if (modeHasFt(mode)) {
        pushSegment(segments, 'ft', daysBetween(periods.ftStart, periods.ftEnd));
      }
      if (modeHasOptunaOos(mode)) {
        pushSegment(segments, 'oos', daysBetween(periods.oosStart, periods.oosEnd));
      }

      return { segments, periods, windows: null };
    }

    if (!modeIsAdaptive(mode)) {
      const windows = periodUnit === 'months'
        ? calcCalendarMonthWindows(startDate, endDate, isDays, oosDays)
        : calcWFAWindows(startDate, endDate, isDays, oosDays);
      if (windows.length < 2) {
        return { segments: [], windows: [], error: 'insufficient' };
      }

      const firstWindow = windows[0];
      const lastWindow = windows[windows.length - 1];
      const firstIsDaysTotal = daysBetween(firstWindow.isStart, firstWindow.isEnd);
      const firstFtCarve = modeHasFt(mode) ? carveFtDays(firstIsDaysTotal, ftDays) : { isDays: firstIsDaysTotal, ftDays: 0 };
      pushSegment(segments, 'is', firstFtCarve.isDays);
      pushSegment(segments, 'ft', firstFtCarve.ftDays);
      pushSegment(segments, 'oos', daysBetween(firstWindow.oosStart, firstWindow.oosEnd));

      if (windows.length > 2) {
        pushSegment(segments, 'collapsed', daysBetween(firstWindow.oosEnd, lastWindow.isStart));
      }

      const lastIsDaysTotal = daysBetween(lastWindow.isStart, lastWindow.isEnd);
      const lastFtCarve = modeHasFt(mode) ? carveFtDays(lastIsDaysTotal, ftDays) : { isDays: lastIsDaysTotal, ftDays: 0 };
      pushSegment(segments, 'is', lastFtCarve.isDays);
      pushSegment(segments, 'ft', lastFtCarve.ftDays);
      pushSegment(segments, 'oos', daysBetween(lastWindow.oosStart, lastWindow.oosEnd));

      const endNorm = addDays(endDate, 1);
      pushSegment(segments, 'unused', daysBetween(lastWindow.oosEnd, endNorm));

      return { segments, windows, periods: null, firstFtCarve };
    }

    const totalDaysNorm = daysBetween(startDate, addDays(endDate, 1));
    if (isDays >= totalDaysNorm) {
      throw new Error('invalid_adaptive_is');
    }

    const w1IsEnd = addDays(startDate, isDays);
    const maxNominalOosDays = daysBetween(w1IsEnd, addDays(endDate, 1));
    if (!oosDays || oosDays > maxNominalOosDays) {
      throw new Error('invalid_adaptive_oos');
    }

    const w1FtCarve = modeHasFt(mode) ? carveFtDays(isDays, ftDays) : { isDays, ftDays: 0 };
    pushSegment(segments, 'is', w1FtCarve.isDays);
    pushSegment(segments, 'ft', w1FtCarve.ftDays);
    pushSegment(segments, 'oos', oosDays);

    const afterW1 = addDays(w1IsEnd, oosDays);
    const endNorm = addDays(endDate, 1);
    pushSegment(segments, 'collapsed', daysBetween(afterW1, endNorm));

    return { segments, windows: null, periods: null, adaptive: true };
  }

  function spanText(cssKey, text) {
    return `<span class="lbl-${cssKey}">${escapeHtml(text)}</span>`;
  }

  function buildLabels(config, result) {
    const { mode, startDate, endDate, isDays, oosDays, ftDays, periodUnit } = config;
    const arrow = '&rarr;';
    const sep = '<span class="lbl-sep">&middot;</span>';
    const dots = '<span class="lbl-dots">&middot;&middot;&middot;</span>';

    if (mode === 'optuna') {
      return `${spanText('is', 'IS')} ${spanText('is', fmtDate(startDate))} ${arrow} ${spanText('is', fmtDate(endDate))}`;
    }

    if (mode === 'optuna-ft') {
      const periods = result.periods;
      return `${spanText('is', 'IS')} ${spanText('is', fmtDate(startDate))} ${arrow} ${spanText('is', fmtDate(periods.isEnd))} ${sep} ${spanText('ft', 'FT')} ${arrow} ${spanText('ft', fmtDate(endDate))}`;
    }

    if (mode === 'optuna-oos') {
      const periods = result.periods;
      return `${spanText('is', 'IS')} ${spanText('is', fmtDate(startDate))} ${arrow} ${spanText('is', fmtDate(periods.isEnd))} ${sep} ${spanText('oos', 'OOS')} ${arrow} ${spanText('oos', fmtDate(endDate))}`;
    }

    if (mode === 'optuna-ft-oos') {
      const periods = result.periods;
      return `${spanText('is', 'IS')} ${spanText('is', fmtDate(startDate))} ${arrow} ${spanText('is', fmtDate(periods.isEnd))} ${sep} ${spanText('ft', 'FT')} ${arrow} ${spanText('ft', fmtDate(periods.ftEnd))} ${sep} ${spanText('oos', 'OOS')} ${arrow} ${spanText('oos', fmtDate(endDate))}`;
    }

    const withFt = modeHasFt(mode);
    const isAdaptive = modeIsAdaptive(mode);

    if (!isAdaptive && periodUnit === 'months') {
      const windows = result.windows || [];
      if (!windows.length) return '';
      const firstWindow = windows[0];
      const lastWindow = windows[windows.length - 1];
      const firstIsEnd = withFt
        ? addDays(firstWindow.isStart, result.firstFtCarve.isDays)
        : firstWindow.isEnd;
      let firstWindowLabel = `${spanText('dim', 'W1')} ${spanText('is', 'IS')} ${spanText('is', fmtDate(firstWindow.isStart))}${arrow}${spanText('is', fmtDate(firstIsEnd))}`;
      if (withFt) {
        firstWindowLabel += ` ${spanText('ft', 'FT')}${arrow}${spanText('ft', fmtDate(firstWindow.isEnd))}`;
      }
      firstWindowLabel += ` ${spanText('oos', 'OOS')}${arrow}${spanText('oos', fmtDate(firstWindow.oosEnd))}`;
      const endNorm = addDays(endDate, 1);
      const unusedDays = daysBetween(lastWindow.oosEnd, endNorm);
      const lastWindowLabel = `${spanText('dim', `W${windows.length}`)} ${spanText('oos', 'OOS')}${arrow}${spanText('oos', fmtDate(lastWindow.oosEnd))}`;
      const unusedLabel = unusedDays > 0 ? ` ${sep} ${spanText('dim', `${unusedDays}d`)}` : '';
      return `${firstWindowLabel} ${dots} ${lastWindowLabel}${unusedLabel}`;
    }

    const nominalIsEnd = addDays(startDate, isDays);
    const ftCarve = withFt ? carveFtDays(isDays, ftDays) : { isDays, ftDays: 0 };
    const w1IsEnd = addDays(startDate, ftCarve.isDays);
    const w1FtEnd = withFt && ftCarve.ftDays > 0 ? nominalIsEnd : null;
    const w1OosEnd = addDays(nominalIsEnd, oosDays);

    let firstWindowLabel = `${spanText('dim', 'W1')} ${spanText('is', 'IS')} ${spanText('is', fmtDate(startDate))}${arrow}${spanText('is', fmtDate(w1IsEnd))}`;
    if (withFt && w1FtEnd) {
      firstWindowLabel += ` ${spanText('ft', 'FT')}${arrow}${spanText('ft', fmtDate(w1FtEnd))}`;
    }

    if (isAdaptive) {
      const adaptiveOosStart = w1FtEnd || nominalIsEnd;
      firstWindowLabel += ` ${spanText('oos', 'OOS')} ${spanText('oos', fmtDate(adaptiveOosStart))}${arrow}${spanText('oos', '...')}`;
      return `${firstWindowLabel} ${dots} ${spanText('dim', 'End')} ${spanText('dim', fmtDate(endDate))}`;
    }

    firstWindowLabel += ` ${spanText('oos', 'OOS')}${arrow}${spanText('oos', fmtDate(w1OosEnd))}`;

    const windows = result.windows || [];
    if (!windows.length) {
      return firstWindowLabel;
    }

    const lastWindow = windows[windows.length - 1];
    const endNorm = addDays(endDate, 1);
    const unusedDays = daysBetween(lastWindow.oosEnd, endNorm);
    const lastWindowLabel = `${spanText('dim', `W${windows.length}`)} ${spanText('oos', 'OOS')}${arrow}${spanText('oos', fmtDate(lastWindow.oosEnd))}`;
    const unusedLabel = unusedDays > 0 ? ` ${sep} ${spanText('dim', `${unusedDays}d`)}` : '';

    return `${firstWindowLabel} ${dots} ${lastWindowLabel}${unusedLabel}`;
  }

  function renderWarning(container, message) {
    container.innerHTML = `<div class="dataset-preview-warn">${escapeHtml(message)}</div>`;
  }

  function renderPreview(container, config) {
    let result;
    try {
      result = buildSegments(config);
    } catch (error) {
      renderWarning(container, GENERIC_ERROR_TEXT);
      return;
    }

    if (result.error === 'insufficient') {
      renderWarning(container, INSUFFICIENT_TEXT);
      return;
    }

    if (!result.segments.length) {
      container.innerHTML = '';
      return;
    }

    const segmentHtml = result.segments
      .map((segment) => {
        const cssClass = {
          is: 'seg-is',
          ft: 'seg-ft',
          oos: 'seg-oos',
          unused: 'seg-unused',
          collapsed: 'seg-collapsed'
        }[segment.type] || 'seg-unused';

        return `<div class="seg ${cssClass}" style="flex-grow: ${segment.days}; flex-shrink: 0; flex-basis: 0;"></div>`;
      })
      .join('');

    const labelHtml = buildLabels(config, result);
    container.innerHTML = `<div class="dataset-preview-bar">${segmentHtml}</div><div class="dataset-preview-labels">${labelHtml}</div>`;
  }

  function updateDatasetPreview() {
    const container = document.getElementById('datasetPreview');
    if (!container) return;

    try {
      const dateFilterElement = document.getElementById('dateFilter');
      const startDateElement = document.getElementById('startDate');
      const endDateElement = document.getElementById('endDate');
      if (!dateFilterElement || !startDateElement || !endDateElement) {
        container.innerHTML = '';
        return;
      }

      if (!dateFilterElement.checked) {
        container.innerHTML = '';
        return;
      }

      const startDate = parseDateInput(startDateElement.value);
      const endDate = parseDateInput(endDateElement.value);
      if (!startDate || !endDate || startDate >= endDate) {
        container.innerHTML = '';
        return;
      }

      const mode = detectMode();
      const periodUnit = document.getElementById('wfCalendarMonths')?.checked ? 'months' : 'days';
      const isDays = readPositiveInt('wfIsPeriodDays', 90);
      const wfOosDays = readPositiveInt('wfOosPeriodDays', 30);
      const ftDays = readPositiveInt('ftPeriodDays', 30);
      const optunaOosDays = readPositiveInt('oosPeriodDays', 30);

      const config = {
        mode,
        startDate,
        endDate,
        periodUnit,
        isDays,
        oosDays: modeIsWfa(mode) ? wfOosDays : optunaOosDays,
        ftDays
      };

      if (modeIsWfa(mode) && (!config.isDays || !config.oosDays)) {
        renderWarning(container, GENERIC_ERROR_TEXT);
        return;
      }
      if (modeHasFt(mode) && !config.ftDays) {
        renderWarning(container, GENERIC_ERROR_TEXT);
        return;
      }
      if (modeHasOptunaOos(mode) && !config.oosDays) {
        renderWarning(container, GENERIC_ERROR_TEXT);
        return;
      }

      renderPreview(container, config);
    } catch (error) {
      console.warn('Dataset preview update failed:', error);
      renderWarning(container, GENERIC_ERROR_TEXT);
    }
  }

  window.updateDatasetPreview = updateDatasetPreview;
})();

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.walkforward_engine import build_calendar_month_windows
from tools.strategy_lab.config import complete_calendar_window_count


@pytest.mark.parametrize(
    ("start", "end", "is_months", "oos_months", "expected"),
    [
        (date(2025, 10, 1), date(2026, 8, 1), 2, 1, 8),
        (date(2025, 1, 1), date(2026, 1, 1), 4, 2, 4),
        (date(2025, 1, 1), date(2025, 5, 1), 3, 2, 0),
    ],
)
def test_complete_calendar_window_count_uses_oos_step(
    start,
    end,
    is_months,
    oos_months,
    expected,
):
    assert complete_calendar_window_count(start, end, is_months, oos_months) == expected


def test_phase0_calendar_facts_match_merlin_authoritative_builder():
    data_index = pd.date_range(
        "2025-08-01T00:00:00Z",
        "2026-08-01T23:30:00Z",
        freq="30min",
    )

    windows = build_calendar_month_windows(
        data_index,
        "2025-10-01",
        pd.Timestamp("2026-08-01T23:59:59.999999Z"),
        2,
        1,
    )

    assert len(windows) == complete_calendar_window_count(
        date(2025, 10, 1),
        date(2026, 8, 1),
        2,
        1,
    ) == 8
    first = windows[0]
    assert first.is_start == pd.Timestamp("2025-10-01T00:00:00Z")
    assert first.is_end == pd.Timestamp("2025-11-30T23:30:00Z")
    assert first.oos_start == pd.Timestamp("2025-12-01T00:00:00Z")
    assert first.oos_end == pd.Timestamp("2025-12-31T23:30:00Z")
    last = windows[-1]
    assert last.is_start == pd.Timestamp("2026-05-01T00:00:00Z")
    assert last.is_end == pd.Timestamp("2026-06-30T23:30:00Z")
    assert last.oos_start == pd.Timestamp("2026-07-01T00:00:00Z")
    assert last.oos_end == pd.Timestamp("2026-07-31T23:30:00Z")
    assert all(window.oos_end < pd.Timestamp("2026-08-01T00:00:00Z") for window in windows)

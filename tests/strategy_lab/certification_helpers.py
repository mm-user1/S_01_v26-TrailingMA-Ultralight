"""Semantic parameter comparison shared with explicit real WFA certification."""

import math

from tools.strategy_lab.certify import semantic_float_equal


def _normalized_params(params):
    normalized = {
        key: value
        for key, value in params.items()
        if key not in {"dateFilter", "start", "end"}
    }
    if "commissionRate" not in normalized:
        return normalized
    if "commissionPct" not in normalized:
        raise ValueError(
            "WFA best_params contains commissionRate without authoritative commissionPct."
        )
    commission_rate = normalized["commissionRate"]
    commission_pct = normalized["commissionPct"]
    if isinstance(commission_rate, bool) or isinstance(commission_pct, bool):
        raise ValueError(
            "WFA best_params commissionRate and commissionPct must be finite numeric values."
        )
    try:
        commission_rate_value = float(commission_rate)
        commission_pct_value = float(commission_pct)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "WFA best_params commissionRate and commissionPct must be finite numeric values."
        ) from exc
    if not math.isfinite(commission_rate_value) or not math.isfinite(
        commission_pct_value
    ):
        raise ValueError(
            "WFA best_params commissionRate and commissionPct must be finite numeric values."
        )
    if not semantic_float_equal(commission_rate_value, commission_pct_value):
        raise ValueError(
            "WFA best_params commissionRate conflicts with authoritative commissionPct."
        )
    normalized.pop("commissionRate")
    return normalized

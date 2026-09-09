"""Small S03 run-spec fixtures; no market dependency or post-change S06 pins."""
import json
from pathlib import Path

from core.grid_v2 import GridV2Settings, build_grid_v2_plan
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy
from tools.strategy_lab.config import semantic_key_digest

RUNSPEC = Path(__file__).resolve().parents[2] / "tools/strategy_lab/runspecs/s03_adaptive_ma_symmetric.json"


def small_raw():
    raw = json.loads(RUNSPEC.read_text(encoding="utf-8"))
    raw["generation"]["planning"].update(enabled_axes=[], axis_values={})
    # Equal fixed values let identity tests distinguish constraints from values.
    raw["generation"]["economics"]["base_params"]["closeCountShort"] = 7
    pin_small_plan(raw)
    return raw


def pin_small_plan(raw):
    generation = raw["generation"]
    planning = generation["planning"]
    plan = build_grid_v2_plan(strategy.load_config(), GridV2Settings(
        enabled_axes=tuple(planning["enabled_axes"]),
        enabled_tie_groups=tuple(planning.get("enabled_tie_groups", [])),
        slow_enrich_selected=False, compiled_workers=1,
        max_signal_cache_mb=generation["resources"]["grid_v2_max_cache_mb"],
    ), generation["economics"]["base_params"])
    planning.update(expected_candidate_count=plan.deduped_candidate_count,
                    expected_plan_fingerprint=plan.plan_fingerprint,
                    expected_semantic_key_digest=semantic_key_digest(plan))
    return plan


def write_spec(path, raw):
    path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path

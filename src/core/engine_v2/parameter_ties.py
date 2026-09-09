"""Bounded independent numeric signal equality declarations for Grid planning."""

from collections.abc import Mapping
import math


def parameter_tie_groups(config):
    rules = config.get("optimization_rules", {})
    groups = rules.get("parameter_tie_groups", []) if isinstance(rules, Mapping) else []
    if not isinstance(groups, list):
        raise ValueError("parameter_tie_groups must be a list.")
    if not groups:
        return ()
    parameters = config.get("parameters", {})
    dependencies = set()
    for name, spec in parameters.items():
        dependency = spec.get("depends_on")
        if dependency:
            dependencies.add(name)
            dependencies.update([dependency] if isinstance(dependency, str) else dependency)
    selector = config.get("execution", {}).get("variantSelector", {}).get("param")
    used, ids = set(), set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("Each parameter tie group must be an object.")
        group_id = group.get("id")
        if not isinstance(group_id, str) or not group_id.strip() or group_id != group_id.strip() or group_id in ids:
            raise ValueError("Parameter tie group IDs must be unique non-empty strings.")
        ids.add(group_id)
        pairs = group.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Parameter tie groups require non-empty pairs.")
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(not isinstance(n, str) for n in pair):
                raise ValueError("Parameter ties require two-member numeric pairs.")
            source, target = pair
            if source == target or any(n not in parameters or n in used or n in dependencies or n == selector for n in pair):
                raise ValueError("Parameter ties require known independent, always-active, non-overlapping parameters.")
            left, right = (parameters[n] for n in pair)
            if left.get("type") not in {"int", "float"} or left.get("type") != right.get("type") or any(s.get("role") != "signal" for s in (left, right)):
                raise ValueError("Parameter ties require matching numeric signal types.")
            for bound in ("min", "max"):
                values = [s.get(bound) for s in (left, right)]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values) or values[0] != values[1]:
                    raise ValueError("Parameter ties require compatible finite numeric domains.")
            if left["min"] > left["max"]:
                raise ValueError("Parameter tie numeric domain is reversed.")
            used.update(pair)
    return tuple(groups)


def enabled_parameter_ties(config, requested=()):
    groups = parameter_tie_groups(config)
    if not isinstance(requested, (list, tuple)) or any(not isinstance(n, str) or not n for n in requested):
        raise ValueError("grid_v2_enabled_tie_groups must be a list of group IDs.")
    if len(set(requested)) != len(requested):
        raise ValueError("grid_v2_enabled_tie_groups contains duplicate IDs.")
    unknown = set(requested) - {g["id"] for g in groups}
    if unknown:
        raise ValueError(f"Unknown parameter tie groups: {sorted(unknown)}.")
    return tuple(tuple(pair) for group in groups if group["id"] in requested for pair in group["pairs"])


def expand_parameter_ties(params, pairs):
    for source, target in pairs:
        params[target] = params[source]

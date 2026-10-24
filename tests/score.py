"""Fixed positive behavior weights; scientific failures reduce reward monotonically."""

from fractions import Fraction
import math


def aggregate(nodeids, reports, profile):
    tests = profile["tests"]
    if (
        not nodeids
        or len(set(nodeids)) != len(nodeids)
        or set(nodeids) != set(tests)
        or set(reports) != set(nodeids)
        or profile.get("gate_tests", [])
    ):
        raise ValueError("Scoring profile does not match the complete suite")
    weights = {}
    for node, item in tests.items():
        value = item["weight"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
            or not isinstance(item.get("rationale"), str)
            or not item["rationale"].strip()
        ):
            raise ValueError("Invalid scientific behavior weight: " + node)
        weights[node] = Fraction(str(value))
    total = sum(weights.values(), Fraction(0))
    if abs(float(total) - 1.0) > 1e-12:
        raise ValueError("Scientific behavior weights must sum to one")
    earned = Fraction(0)
    breakdown = []
    for node in nodeids:
        outcome = reports[node]["call"]["outcome"]
        if outcome not in ("passed", "failed"):
            raise ValueError("Scientific test did not complete: " + node)
        exact_weight = weights[node]
        weight = float(exact_weight)
        passed = outcome == "passed"
        contribution = weight if passed else 0.0
        earned += exact_weight if passed else 0
        breakdown.append(
            {"nodeid": node, "weight": weight, "passed": passed, "earned": contribution}
        )
    return float(earned / total), breakdown

"""Fixed positive behavior weights; scientific failures reduce reward monotonically."""

from collections import Counter
from fractions import Fraction


def aggregate(nodeids, reports, profile):
    functions = [node.split("::")[-1].split("[")[0] for node in nodeids]
    counts = Counter(functions)
    weights = profile["weights"]
    if set(counts) != set(weights) or sum(weights.values()) != profile["total_weight"]:
        raise ValueError("Scoring profile does not match the complete suite")
    earned = Fraction(0)
    breakdown = []
    for node, function in zip(nodeids, functions):
        exact_weight = Fraction(weights[function], counts[function])
        weight = float(exact_weight)
        passed = reports[node]["call"]["outcome"] == "passed"
        contribution = weight if passed else 0.0
        earned += exact_weight if passed else 0
        breakdown.append(
            {"nodeid": node, "weight": weight, "passed": passed, "earned": contribution}
        )
    return float(earned / profile["total_weight"]), breakdown

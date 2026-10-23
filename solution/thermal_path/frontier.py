"""Exact one-dimensional quadratic frontiers for globally smooth model paths.

Each discrete physical interpretation supplies an affine parameter family
v(t)=c+n*t. This module optimizes the whole path, not a greedy sequence of
independently projected representatives. It contains no scoring or task data.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import linalg, optimize


@dataclass
class Mode:
    frame: int
    mapping: tuple
    c: np.ndarray
    n: np.ndarray
    metadata: dict


@dataclass
class Piece:
    lo: float
    hi: float
    a: float
    b: float
    c: float
    mode: Mode
    parent: object
    parent_intercept: float
    parent_slope: float
    key: tuple

    def value(self, temperature):
        return (self.a * temperature + self.b) * temperature + self.c


def _roots(a, b, c, lo, hi):
    scale = max(abs(a), abs(b), abs(c), 1.0)
    if abs(a) <= 2e-14 * scale:
        if abs(b) <= 2e-14 * scale:
            return []
        root = -c / b
        return [root] if lo < root < hi else []
    aa, bb, cc = map(np.longdouble, (a, b, c))
    discriminant = bb * bb - 4 * aa * cc
    if discriminant <= 0:
        return []
    square_root = np.sqrt(discriminant)
    q = -np.longdouble(0.5) * (bb + np.copysign(square_root, bb))
    candidates = [q / aa]
    if q:
        candidates.append(cc / q)
    return sorted(set(float(x) for x in candidates if lo < x < hi))


def _same_lineage(left, right):
    return (left.mode is right.mode and left.parent is right.parent
            and left.parent_intercept == right.parent_intercept
            and left.parent_slope == right.parent_slope
            and (left.a, left.b, left.c, left.key) == (right.a, right.b, right.c, right.key))


def _append(result, source, lo, hi):
    if not hi > lo:
        return
    piece = replace(source, lo=float(lo), hi=float(hi))
    if result and result[-1].hi == piece.lo and _same_lineage(result[-1], piece):
        result[-1] = replace(result[-1], hi=piece.hi)
    else:
        result.append(piece)


def merge_envelopes(left, right):
    if not left:
        return list(right)
    if not right:
        return list(left)
    boundaries = sorted(set([v for p in left + right for v in (p.lo, p.hi)]))
    i = j = 0
    result = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        midpoint = lo + (hi - lo) / 2
        while i < len(left) and left[i].hi <= midpoint:
            i += 1
        while j < len(right) and right[j].hi <= midpoint:
            j += 1
        a = left[i] if i < len(left) and left[i].lo <= midpoint < left[i].hi else None
        b = right[j] if j < len(right) and right[j].lo <= midpoint < right[j].hi else None
        if a is None and b is None:
            continue
        if a is None or b is None:
            _append(result, b if a is None else a, lo, hi)
            continue
        intersections = _roots(a.a - b.a, a.b - b.b, a.c - b.c, lo, hi)
        cuts = [lo, *intersections, hi]
        for x, y in zip(cuts[:-1], cuts[1:]):
            sample = x + (y - x) / 2
            av, bv = a.value(sample), b.value(sample)
            roundoff = 5e-13 * (1 + abs(av) + abs(bv))
            winner = a if av < bv - roundoff else b if bv < av - roundoff else min((a, b), key=lambda p: p.key)
            _append(result, winner, x, y)
    return result


def lower_envelope(proposals):
    if not proposals:
        return []
    if len(proposals) == 1:
        return [proposals[0]]
    midpoint = len(proposals) // 2
    return merge_envelopes(lower_envelope(proposals[:midpoint]), lower_envelope(proposals[midpoint:]))


def initial_frontier(modes, anchor, interval):
    output = []
    for mode in modes:
        offset = mode.c - anchor
        output.append(Piece(interval[0], interval[1], float(mode.n @ mode.n),
                            float(2 * mode.n @ offset), float(offset @ offset),
                            mode, None, 0.0, 0.0, (mode.mapping,)))
    return output


def advance_frontier(frontier, modes, interval):
    result = []
    for mode in modes:
        proposals = []
        for old in frontier:
            offset = mode.c - old.mode.c
            curvature = old.a + old.mode.n @ old.mode.n
            b0 = old.b - 2 * old.mode.n @ offset
            b1 = -2 * old.mode.n @ mode.n
            intercept, slope = -b0 / (2 * curvature), -b1 / (2 * curvature)
            cuts = [interval[0], interval[1]]
            if slope:
                cuts.extend((bound - intercept) / slope for bound in (old.lo, old.hi)
                            if interval[0] < (bound - intercept) / slope < interval[1])
            cuts = sorted(set(cuts))
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                sample = lo + (hi - lo) / 2
                previous_temperature = intercept + slope * sample
                if old.lo < previous_temperature < old.hi:
                    a = mode.n @ mode.n - b1 * b1 / (4 * curvature)
                    b = 2 * mode.n @ offset - b0 * b1 / (2 * curvature)
                    c = old.c + offset @ offset - b0 * b0 / (4 * curvature)
                    parent_intercept, parent_slope = intercept, slope
                else:
                    fixed = float(np.clip(previous_temperature, old.lo, old.hi))
                    a = mode.n @ mode.n
                    b = 2 * mode.n @ offset + b1 * fixed
                    c = old.c + offset @ offset + curvature * fixed * fixed + b0 * fixed
                    parent_intercept, parent_slope = fixed, 0.0
                proposals.append(Piece(float(lo), float(hi), float(a), float(b), float(c),
                                       mode, old, float(parent_intercept), float(parent_slope),
                                       old.key + (mode.mapping,)))
        result.extend(lower_envelope(proposals))
    return result


def backtrack(piece, temperature):
    modes, temperatures = [], []
    while piece is not None:
        modes.append(piece.mode)
        temperatures.append(float(temperature))
        temperature = piece.parent_intercept + piece.parent_slope * temperature
        piece = piece.parent
    return list(reversed(modes)), np.array(list(reversed(temperatures)))


def path_system(modes, anchor):
    count = len(modes)
    dimension = len(anchor)
    matrix = np.zeros((count * dimension, count))
    offset = np.empty(count * dimension)
    for index, mode in enumerate(modes):
        rows = slice(index * dimension, (index + 1) * dimension)
        matrix[rows, index] = mode.n
        offset[rows] = mode.c - (anchor if index == 0 else modes[index - 1].c)
        if index:
            matrix[rows, index - 1] = -modes[index - 1].n
    return matrix, offset


def finish_frontier(frontier, anchor, interval):
    candidates = []
    for piece in frontier:
        if piece.a <= 0:
            raise RuntimeError('A physical path has lost strict terminal convexity')
        temperature = float(np.clip(-piece.b / (2 * piece.a), piece.lo, piece.hi))
        candidates.append((piece.value(temperature), piece.key, piece, temperature))
    if not candidates:
        raise ValueError('No admissible physical interpretations')
    _, _, piece, temperature = min(candidates, key=lambda row: (row[0], row[1]))
    modes, approximate = backtrack(piece, temperature)
    matrix, offset = path_system(modes, anchor)
    # Refine the identified discrete path using a different convex box solver;
    # this cannot change its selected correspondence or bypass the frontier.
    refined = optimize.lsq_linear(matrix, -offset, bounds=interval, method='bvls', tol=1e-13, max_iter=300)
    if not refined.success:
        raise RuntimeError('The selected physical path did not satisfy its convex KKT system')
    temperatures = refined.x
    parameters = np.array([mode.c + mode.n * t for mode, t in zip(modes, temperatures)])
    previous = np.vstack((anchor, parameters[:-1]))
    objective = float(np.sum((parameters - previous)**2))
    stationarity = matrix.T @ (matrix @ temperatures + offset)
    low = temperatures <= interval[0] + 1e-9
    high = temperatures >= interval[1] - 1e-9
    free = ~(low | high)
    residual = max(np.max(np.abs(stationarity[free]), initial=0),
                   np.max(-stationarity[low], initial=0), np.max(stationarity[high], initial=0))
    if residual > 1e-8:
        raise RuntimeError('The selected path has an invalid box-optimality certificate')
    return {'modes': modes, 'temperatures': temperatures, 'parameters': parameters,
            'objective': objective, 'box_stationarity': stationarity,
            'active_lower': low, 'active_upper': high,
            'frontier_polynomial_value': float(piece.value(temperature)),
            'backtracked_temperatures': approximate}


def global_path(mode_layers, anchor, interval):
    frontier = []
    for index, modes in enumerate(mode_layers):
        frontier = (initial_frontier(modes, anchor, interval) if index == 0
                    else advance_frontier(frontier, modes, interval))
    return finish_frontier(frontier, anchor, interval), frontier


def differentiate_path(solution, anchor, c_directions, n_directions, anchor_directions):
    """Differentiate the complete active-set KKT system of the winning path."""
    modes = solution['modes']
    temperatures = solution['temperatures']
    matrix, offset = path_system(modes, anchor)
    hessian = matrix.T @ matrix
    free = ~(solution['active_lower'] | solution['active_upper'])
    result, temperature_result = [], []
    for dc, dn, da in zip(c_directions, n_directions, anchor_directions):
        changed = [Mode(mode.frame, mode.mapping, dci, dni, {}) for mode, dci, dni in zip(modes, dc, dn)]
        derivative_matrix, derivative_offset = path_system(changed, da)
        derivative_hessian = derivative_matrix.T @ matrix + matrix.T @ derivative_matrix
        derivative_linear = derivative_matrix.T @ offset + matrix.T @ derivative_offset
        dt = np.zeros(len(modes))
        if np.any(free):
            dt[free] = -linalg.solve(hessian[np.ix_(free, free)],
                                     (derivative_hessian @ temperatures + derivative_linear)[free], assume_a='pos')
        derivative_parameters = np.array([dci + dni * t + mode.n * ti
                                          for mode, dci, dni, t, ti in zip(modes, dc, dn, temperatures, dt)])
        result.append(derivative_parameters)
        temperature_result.append(dt)
    return np.asarray(result), np.asarray(temperature_result)


def serialize_frontier(frontier):
    """Data-only frontier and ancestry; no executable/pickle state."""
    node_table, mode_table = [], []
    node_ids, mode_ids = {}, {}

    def visit(piece):
        if id(piece) in node_ids:
            return node_ids[id(piece)]
        parent = None if piece.parent is None else visit(piece.parent)
        if id(piece.mode) not in mode_ids:
            mode_ids[id(piece.mode)] = len(mode_table)
            mode_table.append({'frame': piece.mode.frame, 'mapping': list(piece.mode.mapping),
                               'c': piece.mode.c.tolist(), 'n': piece.mode.n.tolist(),
                               'metadata': piece.mode.metadata})
        index = len(node_table)
        node_ids[id(piece)] = index
        node_table.append({'lo': piece.lo, 'hi': piece.hi, 'a': piece.a, 'b': piece.b, 'c': piece.c,
                           'mode': mode_ids[id(piece.mode)], 'parent': parent,
                           'parent_intercept': piece.parent_intercept, 'parent_slope': piece.parent_slope,
                           'key': [list(x) for x in piece.key]})
        return index

    heads = [visit(piece) for piece in frontier]
    return {'modes': mode_table, 'nodes': node_table, 'heads': heads}


def deserialize_frontier(payload):
    modes = [Mode(x['frame'], tuple(x['mapping']), np.asarray(x['c'], float), np.asarray(x['n'], float), x['metadata'])
             for x in payload['modes']]
    nodes = []
    for index, row in enumerate(payload['nodes']):
        parent = row['parent']
        if parent is not None and (not isinstance(parent, int) or parent < 0 or parent >= index):
            raise ValueError('Invalid frontier ancestry')
        node = Piece(row['lo'], row['hi'], row['a'], row['b'], row['c'], modes[row['mode']],
                     None if parent is None else nodes[parent], row['parent_intercept'], row['parent_slope'],
                     tuple(tuple(x) for x in row['key']))
        nodes.append(node)
    return [nodes[index] for index in payload['heads']]

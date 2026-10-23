"""Complete independent certificate for a legal 12-frame complex GHF path."""

from pathlib import Path
import itertools
import numpy as np
from scipy import linalg
from pyscf import gto, scf
from independent import physical_modes


def certified_corner():
    raw = np.load(Path(__file__).parent / "data/native_global_corner.npz")
    spec = {"atom": "He 0 0 0; H 0 0 1.2", "charge": 1, "basis": "sto-3g", "verbose": 0}
    mol = gto.M(**spec)
    mf = scf.GHF(mol)
    s = mf.get_ovlp()
    root = np.asarray(linalg.sqrtm(s), complex)
    case = {
        "spec": spec,
        "mf": mf,
        "mol": mol,
        "s": s,
        "root": root,
        "inverse_root": np.linalg.inv(root),
        "generators": raw["generators"],
        "constraints": np.asarray([s]),
        "targets": np.array([2.0]),
        "unlabeled": np.array([3, 4]),
        "interval": (1.0, 2.0),
        "coupling": 0.85,
        "anchor": raw["anchor"],
        "anchor_directions": np.zeros((0, 16)),
        "records": [
            {"values": v, "entropy": float(e)}
            for v, e in zip(raw["values"], raw["entropy"])
        ],
    }
    layers = [physical_modes(case, record) for record in case["records"]]
    if any(len(modes) != 2 for modes in layers):
        raise AssertionError(
            "Certified fixture lost its physical channel correspondences"
        )
    for modes in layers:
        for mode in modes:
            mode["z"] = mode["c"] + case["interval"][0] * mode["n"]
    margin = np.inf
    for j, modes in enumerate(layers):
        for mode in modes:
            previous = [case["anchor"]] if j == 0 else [x["z"] for x in layers[j - 1]]
            following = (
                [None] if j == len(layers) - 1 else [x["z"] for x in layers[j + 1]]
            )
            for left, right in itertools.product(previous, following):
                gradient = 2 * mode["n"] @ (mode["z"] - left)
                if right is not None:
                    gradient += 2 * mode["n"] @ (mode["z"] - right)
                margin = min(margin, float(gradient))
    if margin <= 1e-4:
        raise AssertionError(
            "Universal temperature-box KKT certificate no longer holds"
        )
    solutions = []
    for indices in itertools.product((0, 1), repeat=len(layers)):
        z = np.array([layers[j][a]["z"] for j, a in enumerate(indices)])
        cost = float(np.sum((z - np.vstack((case["anchor"], z[:-1]))) ** 2))
        solutions.append((cost, indices, z))
    solutions.sort(key=lambda item: (item[0], item[1]))
    best = solutions[0]
    if solutions[1][0] - best[0] <= 1e-4:
        raise AssertionError("Certified global correspondence is not separated")
    return case, {
        "objective": best[0],
        "parameters": best[2],
        "mapping": [list(layers[j][a]["mapping"]) for j, a in enumerate(best[1])],
        "density": np.array([layers[j][a]["density"] for j, a in enumerate(best[1])]),
        "KKT_margin": margin,
        "enumerated": len(solutions),
        "gap": solutions[1][0] - best[0],
    }

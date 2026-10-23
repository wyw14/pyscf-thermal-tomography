"""Global unlabeled thermal-HF tomography using actual native PySCF kernels."""

import hashlib
import itertools
import json

import numpy as np
from scipy import linalg

from .linear_algebra import hermitian_basis, encode, decode, entropy_data
from .context import context_digest
from .frontier import (
    Mode,
    initial_frontier,
    advance_frontier,
    finish_frontier,
    differentiate_path,
    serialize_frontier,
    deserialize_frontier,
)


def seal(payload, context):
    content = {"schema_version": 1, "context": context, "payload": payload}
    data = json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return {**content, "sha256": hashlib.sha256(data).hexdigest()}


def open_checkpoint(checkpoint, context):
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema_version",
        "context",
        "payload",
        "sha256",
    }:
        raise ValueError("Invalid checkpoint envelope")
    if checkpoint["schema_version"] != 1 or checkpoint["context"] != context:
        raise ValueError("Checkpoint scientific context/version mismatch")
    if not isinstance(checkpoint["payload"], dict):
        raise ValueError("Checkpoint payload must be a JSON object")
    try:
        digest = seal(checkpoint["payload"], context)["sha256"]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Checkpoint payload is not finite JSON data") from error
    if digest != checkpoint["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    required = {
        "anchor",
        "anchor_directions",
        "records",
        "candidate_counts",
        "frontier",
    }
    if set(checkpoint["payload"]) != required:
        raise ValueError("Checkpoint payload is incomplete")
    return json.loads(json.dumps(checkpoint["payload"], allow_nan=False))


class NativeContext:
    def __init__(
        self, mf, generators, constraints, targets, interval, coupling, unlabeled
    ):
        self.mf = mf.copy()
        generators = np.asarray(generators, complex)
        constraints = np.asarray(constraints, complex)
        targets = np.asarray(targets, float)
        size = len(self.mf.get_ovlp())
        if generators.ndim != 3 or generators.shape[1:] != (size, size):
            raise ValueError("Invalid generator shape")
        if (
            constraints.ndim != 3
            or constraints.shape[1:] != (size, size)
            or targets.shape != (len(constraints),)
        ):
            raise ValueError("Invalid fixed constraint shape")
        if not all(np.all(np.isfinite(x)) for x in (generators, constraints, targets)):
            raise ValueError("Nonfinite scientific operator input")
        if (
            len(interval) != 2
            or not np.all(np.isfinite(interval))
            or not 0 < interval[0] < interval[1]
        ):
            raise ValueError("Invalid temperature interval")
        if not np.isfinite(coupling) or not 0 < coupling <= 1:
            raise ValueError("Invalid interaction coupling")
        if len(unlabeled) > 4 or any(
            not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
            for i in unlabeled
        ):
            raise ValueError("Invalid unlabeled indices")
        if len(set(unlabeled)) != len(unlabeled) or any(
            i < 0 or i >= len(generators) for i in unlabeled
        ):
            raise ValueError("Invalid unlabeled indices")
        self.interval = tuple(map(float, interval))
        self.coupling = float(coupling)
        self.generators = np.asarray(generators, complex)
        self.observables = np.asarray(constraints, complex)
        self.targets = np.asarray(targets, float)
        self.unlabeled = np.asarray(unlabeled, int)
        self.overlap = np.asarray(self.mf.get_ovlp(), complex)
        values, vectors = linalg.eigh(self.overlap)
        self.x = (vectors / np.sqrt(values)) @ vectors.conj().T
        self.basis = hermitian_basis(len(values))
        self.q = encode(
            self.basis,
            np.einsum("ij,ajk,kl->ail", self.x.conj().T, self.observables, self.x),
        )
        self.v = encode(
            self.basis,
            np.einsum("ij,ajk,kl->ail", self.x.conj().T, self.generators, self.x),
        )
        tomography = np.vstack((self.q, self.v))
        if np.linalg.matrix_rank(tomography, tol=1e-10) != len(self.basis):
            raise ValueError(
                "The labeled observables do not determine the one-body density"
            )
        if np.linalg.matrix_rank(self.q, tol=1e-10) != len(self.targets):
            raise ValueError("Dependent fixed constraints")
        self.tomography = tomography
        self.tomography_inverse = linalg.pinv(tomography, rtol=1e-10)
        self.z = linalg.null_space(self.q, rcond=1e-12)
        projected = self.z.T @ self.v.T
        self.projected_inverse = linalg.pinv(projected, rtol=1e-10)
        self.coefficient_gauge = (
            np.eye(len(self.generators)) - self.projected_inverse @ projected
        )
        self.h = encode(self.basis, self.x.conj().T @ self.mf.get_hcore() @ self.x)
        density_basis = np.einsum("ij,ajk,kl->ail", self.x, self.basis, self.x.conj().T)
        native = coupling * self.mf.get_veff(self.mf.mol, density_basis, hermi=1)
        transformed = np.einsum("ij,ajk,kl->ail", self.x.conj().T, native, self.x)
        self.interaction = encode(self.basis, transformed).T
        self.interaction = (self.interaction + self.interaction.T) / 2
        if 4 * self.interval[0] + linalg.eigvalsh(self.interaction)[0] <= 1e-6:
            raise ValueError("Outside the uniformly convex thermal domain")
        base = context_digest(
            self.mf, self.generators, self.observables, self.targets, interval, coupling
        )
        self.identity = hashlib.sha256(
            b"global-native-thermal-prototype-v1\x00"
            + base.encode("ascii")
            + json.dumps(self.unlabeled.tolist()).encode("ascii")
        ).hexdigest()
        self.nuclear_energy = float(self.mf.energy_nuc())

    def decoded(self, values, mapping):
        result = np.asarray(values, float).copy()
        result[self.unlabeled] = result[self.unlabeled[np.asarray(mapping, int)]]
        return result

    def modes(self, record, frame, anchor):
        result = []
        constant_gauge = self.coefficient_gauge @ anchor[:-1]
        for mapping in itertools.permutations(range(len(self.unlabeled))):
            moments = self.decoded(record["values"], mapping)
            rhs = np.r_[self.targets, moments]
            p = self.tomography_inverse @ rhs
            if np.max(np.abs(self.tomography @ p - rhs)) > 2e-9:
                continue
            try:
                entropy, logit, _, _, _ = entropy_data(
                    decode(self.basis, p), self.basis, False
                )
            except ValueError:
                continue
            if abs(entropy - record["entropy"]) > 2e-9:
                continue
            c = np.r_[
                -self.projected_inverse @ self.z.T @ (self.h + self.interaction @ p)
                + constant_gauge,
                0.0,
            ]
            n = np.r_[-self.projected_inverse @ self.z.T @ logit, 1.0]
            result.append(
                Mode(
                    frame,
                    tuple(mapping),
                    c,
                    n,
                    {
                        "p": p.tolist(),
                        "logit": logit.tolist(),
                        "entropy": float(entropy),
                        "decoded_moments": moments.tolist(),
                    },
                )
            )
        if not result:
            raise ValueError(
                "No channel correspondence gives compatible thermal density data"
            )
        return result

    def mode_derivatives(self, mode, record, anchor_directions):
        count = len(anchor_directions)
        p = np.asarray(mode.metadata["p"], float)
        entropy_hessian = entropy_data(decode(self.basis, p), self.basis)[-1]
        dc, dn, dp_result = [], [], []
        for direction, anchor_direction in zip(record["directions"], anchor_directions):
            moments = self.decoded(direction[:-1], mode.mapping)
            rhs = np.r_[np.zeros(len(self.targets)), moments]
            dp = self.tomography_inverse @ rhs
            if np.max(np.abs(self.tomography @ dp - rhs)) > 2e-8:
                raise ValueError("Incompatible tomographic observation direction")
            if abs(-np.asarray(mode.metadata["logit"]) @ dp - direction[-1]) > 2e-8:
                raise ValueError("Incompatible entropy direction")
            dc.append(
                np.r_[
                    -self.projected_inverse @ self.z.T @ self.interaction @ dp
                    + self.coefficient_gauge @ anchor_direction[:-1],
                    0.0,
                ]
            )
            dn.append(
                np.r_[-self.projected_inverse @ self.z.T @ entropy_hessian @ dp, 0.0]
            )
            dp_result.append(dp)
        dimension = len(self.generators) + 1
        return (
            np.asarray(dc).reshape((count, dimension)),
            np.asarray(dn).reshape((count, dimension)),
            np.asarray(dp_result).reshape((count, len(self.basis))),
        )


def normalize_record(row, generators, queries):
    if not isinstance(row, dict) or "values" not in row or "entropy" not in row:
        raise ValueError("Observation record requires values and entropy")
    values = np.asarray(row["values"], float)
    entropy = float(row["entropy"])
    directions = np.asarray(
        row.get("directions", np.zeros((queries, generators + 1))), float
    )
    if values.shape != (generators,) or directions.shape != (queries, generators + 1):
        raise ValueError("Observation record has an invalid shape")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(directions))
        or not np.isfinite(entropy)
    ):
        raise ValueError("Nonfinite observation")
    return {
        "values": values.tolist(),
        "entropy": entropy,
        "directions": directions.tolist(),
    }


def reconstruct(
    mf,
    generators,
    constraints,
    targets,
    records,
    unlabeled_indices=(),
    initial_anchor=None,
    initial_directions=None,
    checkpoint=None,
    temperature_interval=(0.4, 1.6),
    coupling=1.0,
):
    context = NativeContext(
        mf,
        generators,
        constraints,
        targets,
        temperature_interval,
        coupling,
        unlabeled_indices,
    )
    dimension = len(generators) + 1
    if checkpoint is None:
        anchor = (
            np.r_[np.zeros(dimension - 1), 1.0]
            if initial_anchor is None
            else np.array(initial_anchor, dtype=float, copy=True)
        )
        if initial_directions is None:
            queries = len(records[0].get("directions", [])) if records else 0
            anchor_directions = np.zeros((queries, dimension))
        else:
            anchor_directions = np.asarray(initial_directions, float).copy()
            queries = len(anchor_directions)
        previous_records, frontier, candidate_counts = [], [], []
    else:
        if initial_anchor is not None or initial_directions is not None:
            raise ValueError("A resumed inverse path already has an initial prior")
        previous = open_checkpoint(checkpoint, context.identity)
        try:
            anchor = np.asarray(previous["anchor"], float)
            anchor_directions = np.asarray(
                previous["anchor_directions"], float
            ).reshape((-1, dimension))
            queries = len(anchor_directions)
            previous_records = previous["records"]
            frontier = deserialize_frontier(previous["frontier"])
            candidate_counts = previous["candidate_counts"]
            if (
                not isinstance(previous_records, list)
                or not isinstance(candidate_counts, list)
                or len(previous_records) != len(candidate_counts)
            ):
                raise ValueError("Invalid checkpoint prefix")
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("Invalid checkpoint payload") from error
    if anchor.shape != (dimension,) or anchor_directions.shape != (queries, dimension):
        raise ValueError("Invalid prior shape")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(anchor_directions)):
        raise ValueError("Nonfinite prior")
    all_records = list(previous_records)
    for original in records:
        row = normalize_record(original, dimension - 1, queries)
        modes = context.modes(row, len(all_records), anchor)
        frontier = (
            initial_frontier(modes, anchor, temperature_interval)
            if not all_records
            else advance_frontier(frontier, modes, temperature_interval)
        )
        all_records.append(row)
        candidate_counts.append(len(modes))
    payload = {
        "anchor": anchor.tolist(),
        "anchor_directions": anchor_directions.tolist(),
        "records": all_records,
        "candidate_counts": candidate_counts,
        "frontier": serialize_frontier(frontier),
    }
    saved = seal(payload, context.identity)
    if not all_records:
        return {
            "frames": [],
            "objective": 0.0,
            "objective_response": np.zeros(queries),
            "candidate_counts": [],
            "checkpoint": saved,
        }
    solution = finish_frontier(frontier, anchor, temperature_interval)
    c_derivatives, n_derivatives, p_derivatives = [], [], []
    for mode, row in zip(solution["modes"], all_records):
        dc, dn, dp = context.mode_derivatives(mode, row, anchor_directions)
        c_derivatives.append(dc)
        n_derivatives.append(dn)
        p_derivatives.append(dp)
    if queries:
        dc = np.asarray(c_derivatives).transpose(1, 0, 2)
        dn = np.asarray(n_derivatives).transpose(1, 0, 2)
        parameter_response, temperature_response = differentiate_path(
            solution, anchor, dc, dn, anchor_directions
        )
    else:
        parameter_response = np.zeros((0, len(all_records), dimension))
    output = []
    coefficient_gauge = np.zeros((dimension, dimension))
    coefficient_gauge[:-1, :-1] = context.coefficient_gauge
    gram = context.q @ context.q.T
    for index, (mode, parameters) in enumerate(
        zip(solution["modes"], solution["parameters"])
    ):
        p = np.asarray(mode.metadata["p"])
        logit = np.asarray(mode.metadata["logit"])
        density = context.x @ decode(context.basis, p) @ context.x.conj().T
        temperature = float(parameters[-1])
        gradient = (
            context.h
            + context.interaction @ p
            + context.v.T @ parameters[:-1]
            + temperature * logit
        )
        dual = linalg.solve(gram, context.q @ gradient, assume_a="pos")
        energy = float(
            context.nuclear_energy
            + context.h @ p
            + parameters[:-1] @ np.asarray(mode.metadata["decoded_moments"])
            + 0.5 * p @ context.interaction @ p
        )
        projector = coefficient_gauge + np.outer(mode.n, mode.n) / (mode.n @ mode.n)
        density_response = np.einsum(
            "ij,ajk,kl->ail",
            context.x,
            decode(context.basis, p_derivatives[index]),
            context.x.conj().T,
        )
        output.append(
            {
                "mapping": list(mode.mapping),
                "density": density,
                "coefficients": parameters[:-1].copy(),
                "temperature": temperature,
                "constraint_multipliers": dual,
                "joint_gauge_projector": projector,
                "entropy": mode.metadata["entropy"],
                "energy": energy,
                "free_energy": energy - temperature * mode.metadata["entropy"],
                "parameter_response": parameter_response[:, index],
                "density_response": density_response,
                "temperature_bound_multiplier": float(
                    2 * solution["box_stationarity"][index]
                ),
            }
        )
    previous_parameters = np.vstack((anchor, solution["parameters"][:-1]))
    objective_response = np.zeros(queries)
    for query in range(queries):
        previous_direction = np.vstack(
            (anchor_directions[query], parameter_response[query, :-1])
        )
        objective_response[query] = 2 * np.sum(
            (solution["parameters"] - previous_parameters)
            * (parameter_response[query] - previous_direction)
        )
    return {
        "frames": output,
        "objective": solution["objective"],
        "objective_response": objective_response,
        "candidate_counts": candidate_counts,
        "checkpoint": saved,
    }

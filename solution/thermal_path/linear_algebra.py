"""Hermitian coordinates and fermionic entropy derivatives."""

import numpy as np
from scipy import linalg


def hermitian_basis(size):
    basis = []
    for i in range(size):
        value = np.zeros((size, size), complex)
        value[i, i] = 1
        basis.append(value)
    for i in range(size):
        for j in range(i + 1, size):
            value = np.zeros((size, size), complex)
            value[i, j] = value[j, i] = 1 / np.sqrt(2)
            basis.append(value)
            value = np.zeros((size, size), complex)
            value[i, j] = 1j / np.sqrt(2)
            value[j, i] = -1j / np.sqrt(2)
            basis.append(value)
    return np.asarray(basis)


def encode(basis, matrix):
    return np.einsum("aij,...ij->...a", basis.conj(), matrix).real


def decode(basis, vector):
    return np.einsum("...a,aij->...ij", vector, basis)


def entropy_data(matrix, basis, with_hessian=True):
    occupations, vectors = linalg.eigh(matrix)
    if occupations.min() <= 0 or occupations.max() >= 1:
        raise ValueError("Thermal occupation outside the open physical interval")
    logarithms = np.log(occupations) - np.log1p(-occupations)
    entropy = -np.sum(
        occupations * np.log(occupations) + (1 - occupations) * np.log1p(-occupations)
    )
    logit_matrix = (vectors * logarithms) @ vectors.conj().T
    gradient = encode(basis, logit_matrix)
    if not with_hessian:
        return entropy, gradient, occupations, vectors, None
    differences = occupations[:, None] - occupations[None, :]
    scale = np.minimum(occupations, 1 - occupations)
    close = np.abs(differences) <= 1e-7 * np.minimum(scale[:, None], scale[None, :])
    weights = np.empty_like(differences)
    np.divide(
        logarithms[:, None] - logarithms[None, :],
        differences,
        out=weights,
        where=~close,
    )
    means = (occupations[:, None] + occupations[None, :]) / 2
    weights[close] = 1 / (means[close] * (1 - means[close]))
    transformed = np.einsum("ij,ajk,kl->ail", vectors.conj().T, basis, vectors)
    hessian = np.einsum("aij,ij,bij->ab", transformed.conj(), weights, transformed).real
    return entropy, gradient, occupations, vectors, (hessian + hessian.T) / 2

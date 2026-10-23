"""Author extension boundary for native PySCF thermal path tomography."""

def reconstruct(mf, generators, constraints, targets, records,
                unlabeled_indices=(), initial_anchor=None,
                initial_directions=None, checkpoint=None,
                temperature_interval=(0.4, 1.6), coupling=1.0):
    """See the Harbor instruction for the scientific and interface contract."""
    raise NotImplementedError("Implement coupled native thermal path reconstruction")

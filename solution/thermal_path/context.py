"""Hash native scientific inputs for data-only continuation."""

import hashlib
import json
import numpy as np


def _hash_array(digest, name, value):
    data = np.asarray(value)
    data = np.asarray(
        data, dtype="<c16" if np.iscomplexobj(data) else "<f8", order="C"
    ).copy()
    if not np.all(np.isfinite(data)):
        raise ValueError("Non-finite scientific context")
    data[data == 0] = 0
    digest.update(name.encode("utf-8") + b"\x00")
    digest.update(
        json.dumps(list(data.shape), separators=(",", ":")).encode("ascii") + b"\x00"
    )
    digest.update(data.dtype.str.encode("ascii") + b"\x00" + data.tobytes())


def context_digest(
    mf, generators, observables, targets, temperature_interval, coupling
):
    digest = hashlib.sha256(b"GHF-learning-prototype-v1\x00")
    for name, value in [
        ("overlap", mf.get_ovlp()),
        ("one_body", mf.get_hcore()),
        ("two_electron", mf.mol.intor("int2e", aosym="s8")),
        ("nuclear_energy", mf.energy_nuc()),
        ("generators", generators),
        ("observables", observables),
        ("targets", targets),
        ("temperature_interval", temperature_interval),
        ("coupling", coupling),
    ]:
        _hash_array(digest, name, value)
    return digest.hexdigest()

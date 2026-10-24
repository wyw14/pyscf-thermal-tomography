"""Restricted-process adapter: build a genuine native GHF, then call candidate API."""

import contextlib, copy, importlib, json, sys, traceback

sys.path.insert(0, "/runner")
from codec import encode, decode


def scientific_state(mf):
    """Snapshot physical SCF results without forbidding lazy integral caches."""
    return {
        field: copy.deepcopy(getattr(mf, field))
        for field in ("mo_coeff", "mo_occ", "mo_energy", "converged")
    }


def unchanged_scientific_state(before, mf):
    """Preserve None, scalar types, and array types, dtypes, shapes and values."""
    import numpy as np

    for field, previous in before.items():
        current = getattr(mf, field)
        if type(current) is not type(previous):
            return False
        if isinstance(previous, np.ndarray):
            if current.dtype != previous.dtype or not np.array_equal(
                current, previous, equal_nan=True
            ):
                return False
        elif current != previous:
            return False
    return True


sys.path.insert(0, "/app/src")
request = decode(json.loads(sys.stdin.read()))
try:
    with contextlib.redirect_stdout(sys.stderr):
        from pyscf import gto, scf, lib

        lib.num_threads(1)
        molecule = gto.M(**request["args"][0])
        mf = scf.GHF(molecule)
        # Cache the native AO integrals; the supported molecules have at most
        # four spatial orbitals, so all Coulomb/exchange paths fit in memory.
        mf._eri = molecule.intor("int2e", aosym="s8")
        native_state = request.get("native_state", "unrun")
        if native_state == "converged":
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("Native GHF preparation did not converge")
        elif native_state != "unrun":
            raise ValueError("Unknown native GHF preparation state")
        module = importlib.import_module("thermal_path")
        before = json.dumps(encode(request), sort_keys=True, allow_nan=False)
        native_eri = mf._eri
        native_before = (mf.get_ovlp().copy(), mf.get_hcore().copy(), mf._eri.copy())
        state_before = scientific_state(mf)
        failure = None
        try:
            result = getattr(module, request["function"])(
                mf, *request["args"][1:], **request["kwargs"]
            )
        except Exception as error:
            failure = error
        import numpy as np

        unchanged = (
            before == json.dumps(encode(request), sort_keys=True, allow_nan=False)
            and np.array_equal(native_before[0], mf.get_ovlp())
            and np.array_equal(native_before[1], mf.get_hcore())
            and native_eri is mf._eri
            and np.array_equal(native_before[2], mf._eri)
            and unchanged_scientific_state(state_before, mf)
        )
        if not unchanged:
            raise AssertionError(
                "Candidate changed an input or native mean-field object"
            )
        if failure is not None:
            raise failure
    response = {"ok": True, "value": encode(result)}
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    response = {"ok": False, "error": type(error).__name__, "message": str(error)}
sys.stdout.write(json.dumps(response, allow_nan=False))

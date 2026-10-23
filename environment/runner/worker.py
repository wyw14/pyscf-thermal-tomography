"""Restricted-process adapter: build a genuine native GHF, then call candidate API."""

import contextlib, importlib, json, sys, traceback

sys.path.insert(0, "/runner")
from codec import encode, decode

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
        module = importlib.import_module("thermal_path")
        before = json.dumps(encode(request), sort_keys=True, allow_nan=False)
        native_eri = mf._eri
        native_before = (mf.get_ovlp().copy(), mf.get_hcore().copy(), mf._eri.copy())
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

"""Trusted test-side proxy. Candidate code never imports into the pytest process."""

import builtins
import ctypes
import time
import json
import os
import signal
import subprocess
import sys
import shutil
from contextlib import contextmanager
from runtime_sandbox import new_scratch, restrict
from pathlib import Path
from codec import encode, decode
from trusted_events import record_runtime_fault


def _cleanup_candidate_uid(candidate_uid):
    """This dedicated verifier runs one candidate call at a time."""
    if os.geteuid() != 0:
        return
    for attempt in range(30):
        remaining = []
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1)
                    for line in status.read_text().splitlines()
                    if ":" in line
                )
                uid = int(fields["Uid"].split()[0])
                state = fields["State"].strip().split()[0]
                if uid == candidate_uid:
                    remaining.append((int(status.parent.name), state))
            except (OSError, ValueError, KeyError):
                continue
        if not remaining:
            return
        for pid, state in remaining:
            if state == "Z":
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(0.05)
    raise RuntimeError("Candidate processes were not fully reaped")


@contextmanager
def _scratch():
    try:
        folder, uid = new_scratch()
    except (OSError, RuntimeError) as error:
        record_runtime_fault("sandbox_preparation", error_type=type(error).__name__)
        raise
    try:
        yield folder, uid
    finally:
        try:
            shutil.rmtree(folder)
        except OSError as error:
            record_runtime_fault("scratch_cleanup", error_type=type(error).__name__)
            raise


def call(function, *args, _native_state="unrun", **kwargs):
    timeout = float(os.environ.get("CANDIDATE_CALL_TIMEOUT", "180"))
    payload = json.dumps(
        encode(
            dict(function=function, args=args, kwargs=kwargs, native_state=_native_state)
        ),
        allow_nan=False,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        record_runtime_fault("subreaper_preparation", errno=ctypes.get_errno())
        raise OSError(ctypes.get_errno(), "Candidate descendant reaping unavailable")
    with _scratch() as (folder, uid):
        temp = "/tmp/" + Path(folder).name
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "/runner/worker.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/",
                start_new_session=True,
                preexec_fn=lambda: restrict(folder, uid),
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": temp,
                    "MPLCONFIGDIR": temp,
                    "TMPDIR": temp,
                    "LANG": "C.UTF-8",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        except (OSError, subprocess.SubprocessError) as error:
            record_runtime_fault("worker_start", error_type=type(error).__name__)
            raise
        try:
            stdout, stderr = process.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            record_runtime_fault("candidate_deadline", timeout_sec=timeout)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=5)
            raise TimeoutError(f"candidate exceeded {timeout}s")
        except (OSError, UnicodeError) as error:
            record_runtime_fault("worker_transport", error_type=type(error).__name__)
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                record_runtime_fault("worker_cleanup", error_type=type(error).__name__)
                raise
            try:
                _cleanup_candidate_uid(uid)
            except (OSError, RuntimeError) as error:
                record_runtime_fault("worker_cleanup", error_type=type(error).__name__)
                raise
        if process.returncode:
            record_runtime_fault("worker_exit", returncode=process.returncode)
            raise RuntimeError(
                f"candidate process failed {process.returncode}: {stderr[-3000:]}"
            )
        try:
            response = json.loads(stdout)
            if not isinstance(response, dict) or type(response.get("ok")) is not bool:
                raise ValueError("Invalid worker response envelope")
            if response["ok"]:
                return decode(response["value"])
            if not isinstance(response.get("error"), str) or not isinstance(
                response.get("message"), str
            ):
                raise ValueError("Invalid worker exception response")
        except (KeyError, TypeError, ValueError) as error:
            record_runtime_fault("worker_protocol", error_type=type(error).__name__)
            raise RuntimeError("Invalid candidate worker protocol") from error
        if not response["ok"]:
            kind = response["error"]
            cls = getattr(builtins, kind, RuntimeError)
            if not isinstance(cls, type) or not issubclass(cls, Exception):
                cls = RuntimeError
            raise cls(response["message"])

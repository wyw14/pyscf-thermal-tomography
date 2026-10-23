"""Dependency-only candidate filesystem, independent of verifier mount permissions."""

import fcntl
import os
from pathlib import Path
import shutil
import stat

ROOT = Path("/opt/candidate-root")


def _worker_identity():
    """Allocate a non-reused UID so workers cannot write each other's scratch."""
    state = Path("/opt/candidate-worker-state")
    state.mkdir(mode=0o700, exist_ok=True)
    if state.is_symlink() or state.stat().st_uid != 0 or state.stat().st_mode & 0o077:
        raise RuntimeError("Unsafe worker identity state")
    descriptor = os.open(state / "next-uid", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        value = handle.read()
        uid = int(value) if value else 10000
        if not 10000 <= uid < 60000:
            raise RuntimeError("Candidate identity range exhausted")
        handle.seek(0)
        handle.write(str(uid + 1))
        handle.truncate()
        handle.flush()
    return uid


def prepare_root(source):
    """Stage only public candidate code; never copy trusted test material."""
    metadata = ROOT.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0:
        raise RuntimeError("Missing trusted dependency filesystem")
    if metadata.st_mode & 0o022:
        raise RuntimeError("Dependency filesystem is writable by candidate")
    source = Path(source)
    if source.is_symlink() or str(source.resolve()) != str(source):
        raise RuntimeError("Candidate source must be a real directory")
    target = ROOT / "app/src"
    if target.is_symlink():
        raise RuntimeError("Candidate filesystem must not be a link")
    if target.exists():
        shutil.rmtree(target)
    # Validate special files without following candidate-controlled symlinks.
    for parent, dirs, files in os.walk(source, followlinks=False):
        for name in dirs + files:
            mode = (Path(parent) / name).lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                raise RuntimeError("Unsupported special file in candidate source")
    shutil.copytree(source, target, symlinks=True)
    for parent, dirs, files in os.walk(target, followlinks=False):
        Path(parent).chmod(0o755)
        for name in files:
            path = Path(parent) / name
            if not path.is_symlink():
                path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
    # Inaccessible empty roots make forbidden reads/writes unambiguous even
    # though the actual verifier assets are outside this filesystem entirely.
    for name in ("tests", "solution", "logs"):
        path = ROOT / name
        if path.is_symlink():
            raise RuntimeError("Protected filesystem root must not be a link")
        path.mkdir(mode=0o000, exist_ok=True)
        path.chmod(0o000)
    for name, minor in (("null", 3), ("urandom", 9)):
        device = ROOT / "dev" / name
        if not device.exists():
            os.mknod(device, stat.S_IFCHR | 0o666, os.makedev(1, minor))
        metadata = device.lstat()
        if not stat.S_ISCHR(metadata.st_mode) or metadata.st_rdev != os.makedev(1, minor):
            raise RuntimeError("Unexpected device in candidate filesystem")
        device.chmod(0o666 if name == "null" else 0o444)


import ctypes
import errno
import platform
import subprocess
import sys
import tempfile
import json

def _seccomp(libc):
    """Block networking, tracing, signalling and privilege escapes."""

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    denied = [
        41,
        42,
        43,
        49,
        50,
        53,
        62,
        101,
        109,
        112,
        155,
        161,
        165,
        166,
        200,
        234,
        246,
        248,
        249,
        250,
        272,
        297,
        298,
        303,
        304,
        308,
        310,
        311,
        321,
        323,
        424,
        425,
        426,
        427,
        438,
    ]
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, 0xC000003E),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
        (0x35, 0, 1, 0x40000000),
        (0x06, 0, 0, 0x80000000),
    ]
    for number in denied:
        instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x50000 | errno.EPERM)])
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    filters = (Filter * len(instructions))(*(Filter(*x) for x in instructions))
    program = Program(len(instructions), filters)
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Mandatory PR_SET_SECCOMP failed")


def new_scratch():
    """Allocate an isolated identity and writable directory for one call."""
    if os.geteuid() != 0 or platform.machine() != "x86_64":
        raise RuntimeError("Verifier needs Linux/amd64 with root bootstrap")
    uid = _worker_identity()
    folder = tempfile.mkdtemp(prefix="candidate-", dir=ROOT / "tmp")
    os.chmod(folder, 0o700)
    os.chown(folder, uid, uid)
    return folder, uid


def restrict(folder, uid):
    """Enter the dependency filesystem before executing any candidate code."""
    scratch = "/tmp/" + Path(folder).name
    libc = ctypes.CDLL(None, use_errno=True)
    os.chdir(ROOT)
    os.chroot(".")
    os.chdir(scratch)
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Mandatory no_new_privs unavailable")
    _seccomp(libc)


def prepare():
    """Prepare the current source and fail closed if isolation is unavailable."""
    if os.geteuid() != 0 or platform.machine() != "x86_64":
        raise RuntimeError("Verifier needs Linux/amd64 with root bootstrap")
    prepare_root("/app/src")
    folder, uid = new_scratch()
    probe = r"""
import os, socket
from pathlib import Path
for path in ('/tests', '/solution', '/logs'):
    try:
        list(Path(path).iterdir())
    except PermissionError:
        pass
    else:
        raise RuntimeError('Hidden directory accessible: ' + path)
for path in ('/logs/verifier/reward.txt', '/tmp/outside-scratch', '/app/src/forbidden-write'):
    try:
        with open(path, 'w') as stream: stream.write('probe')
    except PermissionError:
        pass
    else:
        raise RuntimeError('Forbidden write succeeded: ' + path)
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError:
    pass
else:
    raise RuntimeError('Network socket accessible')
Path('scratch-probe').write_text('ok')
Path('scratch-probe').unlink()
assert Path('/runner/worker.py').is_file()
assert Path('/app/src').is_dir()
print('filesystem=chroot; seccomp=enabled; candidate probes passed')
"""
    try:
        result = subprocess.run(
            [sys.executable, '-I', '-c', probe],
            preexec_fn=lambda: restrict(folder, uid),
            cwd='/', capture_output=True, text=True, timeout=15,
            env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8'},
        )
        if result.returncode:
            raise RuntimeError('Isolation preflight failed: ' + result.stderr[-3000:])
        print(result.stdout.strip())
    finally:
        shutil.rmtree(folder)


if __name__ == '__main__':
    prepare()

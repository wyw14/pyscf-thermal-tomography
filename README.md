# Verifier environment compatibility

The verifier starts as root on Linux/amd64. It accepts read-only `/tests`
and `/solution` mounts without changing their ownership or permissions.
Candidates run in a dependency-only chroot with a distinct unprivileged UID
and private scratch per call. Only current `/app/src` and the public native
object adapter enter that filesystem; tests, answers and reward files do not.
Mandatory no_new_privs and seccomp prohibit privilege recovery and networking.
Preparation probes filesystem and network restrictions before scientific tests.
Unavailable isolation fails closed. Landlock support is not required.

The container needs the standard Docker SYS_CHROOT, MKNOD, SETUID and SETGID
capabilities; privileged mode is not required. Custom runtimes that remove
these capabilities need compatible configuration. CPU, memory, time budgets,
scientific contracts, assertions and reference implementations are unchanged.

This compatibility revision was prepared without executing builds, tests,
Oracle/NOP, audit tools or model evaluation, as requested. Earlier scores and
trajectories remain evidence for their original commits only.

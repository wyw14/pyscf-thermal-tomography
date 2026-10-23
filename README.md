# Thermal Hamiltonian reconstruction

This application extends PySCF 2.14.0 GHF to synthetic observation paths.
Source: https://github.com/pyscf/pyscf, tag `v2.14.0`, commit
`c63a953ba603a5ad8c1d65d88da72aaf05ede4d8`; its Apache-2.0 license is in
`environment/src/PYSCF-LICENSE`.

Implement `thermal_path.reconstruct` under `/app/src`. The scientific contract
is in `instruction.md`. Only `/app/src` is transferred to the separate verifier.
The native adapter supplies genuine molecular GHF objects with cached AO
integrals. Runtime and verifier dependencies are installed from pinned wheels.

Run with Harbor and Docker on Linux/amd64. The verifier starts as root and runs
each candidate call under a distinct unprivileged UID in a dependency-only
chroot. It needs Docker's standard SYS_CHROOT, MKNOD, SETUID and SETGID
capabilities. Candidate scratch is private; tests, solutions and reward files
are inaccessible. Each scientific call has 180 seconds, 2 CPU and 4 GiB.

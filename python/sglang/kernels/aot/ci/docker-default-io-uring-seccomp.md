# io_uring test seccomp profile

`docker-default-io-uring-seccomp.json` is copied from the Moby default profile at `moby/profiles` tag `seccomp/v0.2.3`, file `seccomp/default.json`. The final rule adds `io_uring_setup`, `io_uring_enter`, and `io_uring_register` for the native PLE test.

To refresh it, download `seccomp/default.json` from a named `moby/profiles` release tag, append the existing three-call rule, and validate the result with `jq empty docker-default-io-uring-seccomp.json`. Review the upstream diff and record the new tag in this note before replacing the JSON file.

#!/usr/bin/env python3
"""
Pre-commit hook: validate CI registry calls under test/registered/.

1. Every test file must contain a CI registry call (register_cuda_ci,
   register_amd_ci, etc.).
2. A CUDA test must register its suite via the modern
   `stage=`/`runner_config=` form. The legacy single-string `suite=` is reserved
   for the stress family (and for AMD/CPU/NPU suites); any other CUDA `suite=`
   resolves to a name no workflow invokes, so the test silently never runs.
   Two shapes are rejected:
     a. `{stage}-test-{runner_config}` -- the modern name stuffed back into the
        legacy form. Reported with the exact stage/runner split to use.
     b. an older `{stage}-{runner_config}` PR-test name (e.g. the pre-migration
        `base-b-kernel-unit-1-gpu-large`) -- no longer matches any workflow
        suite at all.
   The modern form resolves to the identical suite (CIRegistry.effective_suite
   is f"{stage}-test-{runner_config}") and is /rerun-test-able.
3. Every enabled registered file must have an executable `__main__` test entry
   because the CI runner invokes files directly. The block must call pytest or
   unittest, except for registered benchmarks, the named multi-process runners,
   and the explicit direct-runner file lists below.

Reuses ut_parse_one_file() from ci_register.py (AST-based parsing)
to match the same logic used by run_suite.py's collect_tests().
"""

import ast
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

# Suite names of the form `{stage}-test-{runner_config}` are exactly what the
# modern stage=/runner_config= form produces, so a legacy suite= carrying this
# shape is always expressible (and should be expressed) the modern way.
_MODERN_SHAPE = re.compile(r"^(.+)-test-(.+)$")

# The only CUDA suite family still allowed on the legacy single-string `suite=`
# form. Anything else needs stage=/runner_config=, or its effective_suite matches
# no suite any workflow invokes and the test silently never runs.
_LEGACY_CUDA_PREFIXES = ("stress",)
_ALTERNATE_TEST_RUNNERS = {
    "multigpu_bench_main",
    "multigpu_pytest_main",
    "multiprocess_main",
}
_DIRECT_MAIN_RUNNER_FILES = {
    "test/registered/kernels/ops/communication/test_amd_deterministic_custom_allreduce.py",
    "test/registered/kernels/ops/communication/test_amd_nccl_allreduce_determinism.py",
}


def _main_entrypoint_status(tree: ast.Module, filename: str) -> tuple[bool, bool, bool]:
    has_main = False
    runs_framework = False
    runs_alternate = False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            continue
        has_main = True
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if not isinstance(child, ast.Call) or not isinstance(
                child.func, ast.Attribute
            ):
                continue
            if (
                child.func.attr == "main"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id in ("pytest", "unittest")
            ):
                runs_framework = True
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name) and (
                child.func.id in _ALTERNATE_TEST_RUNNERS
                or (child.func.id == "main" and filename in _DIRECT_MAIN_RUNNER_FILES)
            ):
                runs_alternate = True
    return has_main, runs_framework, runs_alternate


def main() -> int:
    # Import ci_register directly to avoid pulling in all of sglang
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join("python", "sglang", "test", "ci", "ci_register.py"),
    )
    ci_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_register)
    cuda = ci_register.HWBackend.CUDA

    # Same exclusion as run_suite.py: pytest+package structure files.
    files = sorted(
        f
        for f in glob.glob("test/registered/**/*.py", recursive=True)
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    )
    if not files:
        return 0

    missing = []
    legacy_shape = []  # (file, suite, stage, runner_config) -- has a -test- split
    non_dispatchable = []  # (file, suite) -- legacy CUDA suite no workflow invokes
    dead_tests = []  # (file) -- enabled registered files with no test entry point
    for f in files:
        try:
            registries, _ = ci_register.ut_parse_one_file(f)
            tree = ast.parse(Path(f).read_text(encoding="utf-8"), filename=f)
            has_main_block, runs_framework, runs_alternate = _main_entrypoint_status(
                tree, f
            )
        except Exception:
            # Skip files that can't be parsed (syntax errors, etc.)
            continue
        if len(registries) == 0:
            missing.append(f)
            continue
        is_registered_benchmark = "/benchmark/" in f
        if any(r.disabled is None for r in registries) and (
            not has_main_block
            or not (runs_framework or runs_alternate or is_registered_benchmark)
        ):
            dead_tests.append(f)
        for r in registries:
            # Pure legacy form on a CUDA registry: suite set, stage/runner unset.
            if not (
                r.backend == cuda
                and r.suite is not None
                and r.stage is None
                and r.runner_config is None
            ):
                continue
            if r.suite.split("-", 1)[0] in _LEGACY_CUDA_PREFIXES:
                continue
            m = _MODERN_SHAPE.match(r.suite)
            if m:
                legacy_shape.append((f, r.suite, m.group(1), m.group(2)))
            else:
                non_dispatchable.append((f, r.suite))

    exit_code = 0
    if missing:
        print("ERROR: Files in test/registered/ missing CI registry call:")
        print("  Move manual-only tests to test/manual/.\n")
        for f in missing:
            print(f"  {f}")
        print()
        exit_code = 1
    if legacy_shape:
        print(
            "ERROR: CUDA test(s) register a `{stage}-test-{runner_config}`-shaped "
            'suite via the legacy `suite="..."` form, which is not dispatchable '
            "via /rerun-test. Switch to the modern `stage=`/`runner_config=` form "
            "(same stage, same runner):\n"
        )
        for f, suite, stage, runner_config in legacy_shape:
            print(
                f"  {f}\n"
                f'    suite="{suite}"'
                f'  ->  stage="{stage}", runner_config="{runner_config}"'
            )
        print()
        exit_code = 1
    if non_dispatchable:
        print(
            'ERROR: CUDA test(s) register a legacy `suite="..."` that is neither a '
            "nightly/stress/weekly suite nor the modern `stage=`/`runner_config=` "
            "form. This name matches no suite the PR-test workflows invoke, so the "
            "test silently never runs. Switch to the modern form:\n"
        )
        for f, suite in non_dispatchable:
            print(
                f"  {f}\n"
                f'    suite="{suite}"'
                f'  ->  stage="...", runner_config="..."'
            )
        print()
        exit_code = 1
    if dead_tests:
        print(
            "ERROR: Enabled registered test file(s) have no test entry point: "
            "the registered file is executed as `python3 file.py`, but its "
            '`if __name__ == "__main__"` block is missing or does not call '
            "unittest.main() or pytest.main(), and does not use a registered "
            "benchmark, named multi-process runner, or allowlisted direct runner. "
            "The tests are skipped while the file reports success. Make __main__ "
            "run the tests (put any CLI "
            "entry point behind an explicit flag):\n"
        )
        for f in dead_tests:
            print(f"  {f}")
        print()
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

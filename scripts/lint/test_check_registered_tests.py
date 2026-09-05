import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CheckRegisteredTestsTest(unittest.TestCase):
    def _run_lint(
        self,
        test_source,
        *,
        relative_path="test/registered/unit/test_cli.py",
        disabled=None,
    ):
        script = Path(__file__).with_name("check_registered_tests.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ci_dir = root / "python/sglang/test/ci"
            ci_dir.mkdir(parents=True)
            (ci_dir / "ci_register.py").write_text(
                "from types import SimpleNamespace\n"
                "class HWBackend:\n"
                "    CUDA = 'cuda'\n"
                "def ut_parse_one_file(path):\n"
                "    return [SimpleNamespace(backend='cpu', "
                "suite='base-a-test-cpu', stage=None, runner_config=None, "
                f"disabled={disabled!r})], True\n"
            )
            test_file = root / relative_path
            test_file.parent.mkdir(parents=True)
            test_file.write_text(test_source)

            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
        return result

    def test_missing_main_block_fails_lint(self):
        result = self._run_lint(
            "import unittest\n"
            "class Broken(unittest.TestCase):\n"
            "    def test_failure(self):\n"
            "        self.fail('should have run')\n"
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("block is missing", result.stdout)

    def test_cli_only_main_block_fails_lint(self):
        result = self._run_lint(
            "import unittest\n"
            "class Broken(unittest.TestCase):\n"
            "    def test_failure(self):\n"
            "        self.fail('should have run')\n"
            "if __name__ == '__main__':\n"
            "    print('cli only')\n"
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("does not call unittest.main() or pytest.main()", result.stdout)

    def test_pytest_function_with_empty_main_block_fails_lint(self):
        result = self._run_lint(
            "def test_failure():\n"
            "    assert False\n"
            "if __name__ == '__main__':\n"
            "    pass\n"
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("does not call unittest.main() or pytest.main()", result.stdout)

    def test_framework_main_block_passes_lint(self):
        result = self._run_lint(
            "import unittest\n"
            "class Passing(unittest.TestCase):\n"
            "    def test_success(self):\n"
            "        pass\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_direct_test_function_call_fails_lint(self):
        result = self._run_lint(
            "def test_one():\n"
            "    pass\n"
            "def test_two():\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    test_one()\n"
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("does not call unittest.main() or pytest.main()", result.stdout)

    def test_named_alternate_runner_passes_lint(self):
        result = self._run_lint(
            "def multiprocess_main():\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    multiprocess_main()\n"
        )

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_named_direct_main_runner_passes_lint(self):
        result = self._run_lint(
            "def main():\n" "    pass\n" "if __name__ == '__main__':\n" "    main()\n",
            relative_path=(
                "test/registered/kernels/ops/communication/"
                "test_amd_deterministic_custom_allreduce.py"
            ),
        )

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_registered_benchmark_main_passes_lint(self):
        result = self._run_lint(
            "if __name__ == '__main__':\n" "    print('run benchmark')\n",
            relative_path="test/registered/benchmark/test_latency.py",
        )

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_disabled_registration_does_not_need_a_main_block(self):
        result = self._run_lint(
            "def test_disabled():\n" "    pass\n",
            disabled="tracked issue",
        )

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_former_direct_test_function_runner_fails_lint(self):
        result = self._run_lint(
            "def test_one():\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    test_one()\n",
            relative_path="test/registered/kernels/ops/moe/test_fp4_moe.py",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("does not call unittest.main() or pytest.main()", result.stdout)


if __name__ == "__main__":
    unittest.main()

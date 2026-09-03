import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CheckRegisteredTestsTest(unittest.TestCase):
    def _run_lint(self, test_source):
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
                "disabled=None)], True\n"
            )
            test_dir = root / "test/registered/unit"
            test_dir.mkdir(parents=True)
            (test_dir / "test_cli.py").write_text(test_source)

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


if __name__ == "__main__":
    unittest.main()

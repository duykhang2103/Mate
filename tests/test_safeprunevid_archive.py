import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bash_executable():
    candidates = []
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            candidates.append(Path(program_files) / "Git" / "bin" / "bash.exe")
    discovered = shutil.which("bash")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("bash is required to test the archived runner")


def _write_fake_command(bin_dir, name):
    command_path = bin_dir / name
    command_path.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'CALLED %s\\n' {name!r} >&2\n",
        encoding="utf-8",
    )


def _runner_environment(temp_dir, *, allow_archived=False):
    bin_dir = temp_dir / "bin"
    bin_dir.mkdir()
    for name in ("hf", "pip", "python"):
        _write_fake_command(bin_dir, name)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
            "HF_TOKEN": "redacted-test-token",
            "SAFEPRUNEVID_STAGE": "baseline-timing",
            "SAFEPRUNEVID_RESULT_ROOT": str(temp_dir / "results"),
        }
    )
    if allow_archived:
        environment["SAFEPRUNEVID_ALLOW_ARCHIVED"] = "1"
    else:
        environment.pop("SAFEPRUNEVID_ALLOW_ARCHIVED", None)
    return environment


def _run(script, environment):
    return subprocess.run(
        [_bash_executable(), str(script)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


class SafePruneVidArchiveTests(unittest.TestCase):
    def test_root_modal_wrapper_refuses_archived_evaluations_locally(self):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "run_evaluation.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("archived", completed.stderr.lower())

    def test_root_runner_refuses_archived_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = _runner_environment(Path(directory))

            completed = _run(REPO_ROOT / "safeprunevid-eval.sh", environment)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("archived", completed.stderr.lower())
        self.assertNotIn("CALLED", completed.stderr)

    def test_preserved_runner_requires_explicit_archive_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = _runner_environment(Path(directory))

            completed = _run(
                REPO_ROOT
                / "archive"
                / "safeprunevid"
                / "safeprunevid-eval.sh",
                environment,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("SAFEPRUNEVID_ALLOW_ARCHIVED=1", completed.stderr)
        self.assertNotIn("CALLED", completed.stderr)

    def test_preserved_runner_remains_reproducible_with_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = _runner_environment(
                Path(directory), allow_archived=True
            )

            completed = _run(
                REPO_ROOT
                / "archive"
                / "safeprunevid"
                / "safeprunevid-eval.sh",
                environment,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CALLED python", completed.stderr)


if __name__ == "__main__":
    unittest.main()

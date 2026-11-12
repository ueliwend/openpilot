"""Bytecode of the new version, compiled by finalize_update() (updated.precompile), so that no process compiles its
modules at its first import onroad on the first drive after an update.

Real compileall on small trees; the updater loop with its commands answered by a stand-in where only the order or
the outcome matters."""
import json
import os
import py_compile
import signal
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated
from openpilot.system.updated.tests.helpers import FakeCommands, GIT_ENV, git, reset_process_state, run_main, running, tmp_root, use_dirs

FIRST_WAIT, AFTER_SUCCESS = 60, 1.5 * 60 * 60
MANAGER_KILLS_AFTER = 5.0   # s, system/manager/process.py
STOPPED = 42
CACHE_TAG = sys.implementation.cache_tag
CHECKED_HASH = 0b11         # PEP 552: flags word of a hash-based pyc whose source is checked on import

BAD_SOURCE = "def broken(:\n"
COMPILE_ERROR = """*** Error compiling 'openpilot/broken.py'...
  File "openpilot/broken.py", line 1
    def broken(:
               ^
SyntaxError: invalid syntax
"""

# a tree laid out as an install: compiled parts, and parts precompile() must skip
TREE = {
  ".gitignore": "*.pyc\n.overlay_init\n.overlay_consistent\n",
  "openpilot/__init__.py": "",
  "openpilot/common/__init__.py": "",
  "openpilot/common/basics.py": "VALUE = 1\n\ndef where():\n  return __file__\n",
  "openpilot/selfdrive/test/helpers.py": "X = 1\n",
  "openpilot/system/tests/test_x.py": "X = 1\n",
  "openpilot/third_party/vendored.py": "print 'python 2'\n",
  "opendbc_repo/opendbc/__init__.py": "",
  "opendbc_repo/opendbc/car.py": "CAR = 'inster'\n",
  "opendbc_repo/opendbc/brands/tests/__init__.py": "",                # opendbc.car.tests is imported onroad (brand scan)
  "opendbc_repo/opendbc/brands/tests/test_car_interfaces.py": "X = 1\n",
  "opendbc_repo/opendbc/brands/tests/helpers/__init__.py": "",
  "tinygrad_repo/tinygrad/__init__.py": "",
  "tinygrad_repo/tinygrad/tensor.py": "class Tensor:\n  pass\n",
  "tinygrad_repo/extra/tool.py": "X = 1\n",
  "tools/dev_only.py": "X = 1\n",
}
COMPILED = ["openpilot/__init__.py", "openpilot/common/__init__.py", "openpilot/common/basics.py", "opendbc_repo/opendbc/__init__.py",
            "opendbc_repo/opendbc/car.py", "opendbc_repo/opendbc/brands/tests/__init__.py", "tinygrad_repo/tinygrad/__init__.py",
            "tinygrad_repo/tinygrad/tensor.py"]


def pyc(path: Path) -> Path:
  return path.parent / "__pycache__" / f"{path.stem}.{CACHE_TAG}.pyc"


def pyc_flags(path: Path) -> int:
  return int.from_bytes(pyc(path).read_bytes()[4:8], "little")


def all_pycs(root: Path) -> list[str]:
  return sorted(str(p.relative_to(root)) for p in root.rglob("*.pyc"))


def children(pid: int) -> list[int]:
  found = []
  for entry in filter(str.isdigit, os.listdir("/proc")):
    try:
      with open(f"/proc/{entry}/stat") as f:
        if int(f.read().rsplit(")", 1)[1].split()[1]) == pid:
          found.append(int(entry))
    except (OSError, ValueError, IndexError):
      pass  # gone meanwhile
  return found


def cmdline(pid: int) -> bytes:
  try:
    return Path(f"/proc/{pid}/cmdline").read_bytes()
  except OSError:
    return b""


def write_tree(root: Path, files: dict[str, str]) -> Path:
  for name, content in files.items():
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(content)
  return root


def git_repo(root: Path, files: dict[str, str]) -> Path:
  root.mkdir(parents=True)
  git(root, "init", "-q", "-b", "inster")
  write_tree(root, files)
  git(root, "add", "-A")
  git(root, "commit", "-q", "-m", "tree")
  return root


class TestPrecompileStep(OpenpilotTestCase):
  """precompile() and finalize_update() with real git and the real compileall"""

  def setup_method(self):
    os.environ.update(GIT_ENV)
    reset_process_state(self)
    self.root = tmp_root(self)

  def _finalize(self, mocker, files=TREE):
    self.dirs = use_dirs(mocker, self.root)
    git_repo(self.dirs.merged, files)                                 # stands for the merged view after the download
    # a current timestamp pyc, as Python leaves in a submodule that the update does not change (git clean does not
    # reach into submodules): copied into the finalized tree with the rest
    py_compile.compile(str(self.dirs.merged / "opendbc_repo/opendbc/car.py"), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    return updated.finalize_update()

  def test_pycs_in_checked_hash_mode_after_finalize(self, mocker):
    assert self._finalize(mocker) is None
    f = self.dirs.finalized
    assert (f / ".overlay_consistent").is_file()
    assert all_pycs(f) == sorted(str(pyc(Path(name))) for name in COMPILED)   # and nothing of tests, third_party, tools, extra
    assert all(pyc_flags(f / name) == CHECKED_HASH for name in COMPILED)                # the timestamp pyc too
    assert git(f, "status", "--porcelain", "--ignored=no") == ""             # the install's git state is untouched

  def test_compiled_code_does_not_depend_on_the_staging_folder(self, mocker):
    self._finalize(mocker)
    data = pyc(self.dirs.finalized / "openpilot/common/basics.py").read_bytes()
    assert str(self.root).encode() not in data                       # relative paths only

  def test_pycs_are_used_after_the_swap(self, mocker):
    # the launch script renames the finalized folder to the install folder; Python must take the pycs from there
    self._finalize(mocker)
    install = self.root / "data_openpilot"
    os.rename(self.dirs.finalized, install)
    os.symlink("tinygrad_repo/tinygrad", install / "tinygrad")       # as launch_chffrplus.sh links the submodule packages
    os.symlink("opendbc_repo/opendbc", install / "opendbc")
    probe = """
import importlib.machinery as m, json
compiled = []
real = m.SourceFileLoader.source_to_code
def source_to_code(self, data, path, **kw):
  compiled.append(path)
  return real(self, data, path, **kw)
m.SourceFileLoader.source_to_code = source_to_code
import openpilot.common.basics as b, opendbc.car, tinygrad.tensor
print(json.dumps([compiled, b.where(), b.where.__code__.co_filename]))
"""
    env = {**os.environ, "PYTHONPATH": str(install)}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    before = {p: p.stat().st_mtime_ns for p in install.rglob("*.pyc")}
    out = subprocess.run([sys.executable, "-c", probe], env=env, cwd=self.root, check=True, capture_output=True, text=True).stdout
    compiled, file, co_filename = json.loads(out)
    assert compiled == []                                            # every module came from its pyc
    assert file == co_filename == str(install / "openpilot/common/basics.py")   # tracebacks name the install
    assert {p: p.stat().st_mtime_ns for p in install.rglob("*.pyc")} == before  # nothing rewritten

  def test_changed_source_is_never_run_from_a_stale_pyc(self, mocker):
    # what checked-hash buys: same size, same mtime, other content (a timestamp pyc would be taken as current)
    self._finalize(mocker)
    source = self.dirs.finalized / "openpilot/common/basics.py"
    stat = source.stat()
    source.write_text(source.read_text().replace("VALUE = 1", "VALUE = 2"))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    env = {**os.environ, "PYTHONPATH": str(self.dirs.finalized), "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", "import openpilot.common.basics as b; print(b.VALUE)"], env=env, cwd=self.root,
                         check=True, capture_output=True, text=True).stdout
    assert out.strip() == "2"

  def test_file_that_does_not_compile_does_not_fail_finalize(self, mocker):
    record = self._finalize(mocker, {**TREE, "openpilot/broken.py": BAD_SOURCE})
    f = self.dirs.finalized
    assert (f / ".overlay_consistent").is_file()                     # the update is ready all the same
    assert all(pyc(f / name).is_file() for name in COMPILED)         # everything else is compiled
    assert not pyc(f / "openpilot/broken.py").exists()
    assert (record["stage"], record["rc"]) == ("precompile", 1)
    assert record["reason"] == "update continues; 1 file(s) not compiled: openpilot/broken.py"
    assert record["output"].startswith("*** Error compiling 'openpilot/broken.py'...")    # errors only
    assert "SyntaxError" in record["output"] and str(self.root) not in record["output"]

  def test_nothing_to_compile(self, mocker):
    run = mocker.patch.object(updated, "run")
    assert updated.precompile(str(self.root)) is None
    run.assert_not_called()

  def test_command_that_cannot_start_does_not_fail_finalize(self, mocker):
    mocker.patch.object(updated.sys, "executable", str(self.root / "no-such-python"))
    record = self._finalize(mocker)
    assert (self.dirs.finalized / ".overlay_consistent").is_file()
    assert record["stage"] == "precompile" and record["reason"].startswith("update continues; FileNotFoundError")

  def test_next_updates_git_clean_removes_them(self, mocker):
    self._finalize(mocker)
    git(self.dirs.finalized, "clean", "-xdff")                       # what fetch_update() runs in the next update's staging area
    assert all_pycs(self.dirs.finalized) == []


# the updater process for the stop test: finalize_update() with the real commands. One source appears in the finalized
# copy after the resets, and the compileall it starts stops at that file until it is ended (usercustomize below).
UPDATER_PROCESS = r'''
import signal, sys
from pathlib import Path
from openpilot.system.updated import updated
staging = Path(sys.argv[1])
updated.OVERLAY_MERGED, updated.FINALIZED = str(staging / "merged"), str(staging / "finalized")
updated.OVERLAY_INIT = Path(sys.argv[2])
run = updated.run
def run_then_add_slow_file(cmd, cwd=None):
  out = run(cmd, cwd)
  if cmd[:3] == ["git", "submodule", "foreach"]:
    (staging / "finalized/openpilot/zz_slow.py").write_text("X = 1\n")
  return out
updated.run = run_then_add_slow_file
signal.signal(signal.SIGINT, updated.request_stop)
try:
  updated.finalize_update()
except KeyboardInterrupt:
  sys.exit(42)
'''
# loaded by every Python started with PYTHONUSERBASE set to its folder, so also by the real compileall
SLOW_COMPILE = r'''
import os, py_compile, time
compile_file = py_compile.compile
def compile_slowly(file, *args, **kwargs):
  if file.endswith("zz_slow.py"):
    open(os.environ["SLOW_COMPILE_MARK"], "w").close()
    time.sleep(600)
  return compile_file(file, *args, **kwargs)
py_compile.compile = compile_slowly
'''


class TestStopDuringPrecompile(OpenpilotTestCase):
  def setup_method(self):
    os.environ.update(GIT_ENV)
    reset_process_state(self)
    self.root = tmp_root(self)

  def test_going_onroad_stops_it_and_the_update_is_redone(self, mocker):
    staging = self.root / "safe_staging"
    git_repo(staging / "merged", TREE)
    overlay_init = self.root / ".overlay_init"
    overlay_init.touch()
    user_base = self.root / "userbase"
    user_site = Path(sysconfig.get_path("purelib", f"{os.name}_user", vars={"userbase": str(user_base)}))
    user_site.mkdir(parents=True)
    (user_site / "usercustomize.py").write_text(SLOW_COMPILE)
    mark = self.root / "compileall_reached_the_slow_file"
    env = {**os.environ, "PYTHONUSERBASE": str(user_base), "SLOW_COMPILE_MARK": str(mark)}
    process = subprocess.Popen([sys.executable, "-c", UPDATER_PROCESS, str(staging), str(overlay_init)], env=env)
    self.addCleanup(process.wait)
    self.addCleanup(lambda: process.poll() is None and process.kill())

    finalized = staging / "finalized"
    end = time.monotonic() + 60
    while not mark.exists():                                         # compileall is in the middle of its work
      assert process.poll() is None and time.monotonic() < end, "compileall did not reach the slow file"
      time.sleep(0.02)
    compileall = [pid for pid in children(process.pid) if b"compileall" in cmdline(pid)]
    assert len(compileall) == 1 and os.getsid(compileall[0]) == compileall[0], compileall   # one, in its own session
    assert pyc(finalized / "openpilot/common/basics.py").exists() and not pyc(finalized / "openpilot/zz_slow.py").exists()
    assert not (finalized / ".overlay_consistent").exists()

    t = time.monotonic()
    os.kill(process.pid, signal.SIGINT)                              # the manager's stop when the car goes onroad
    process.wait(timeout=MANAGER_KILLS_AFTER)
    seconds = time.monotonic() - t

    assert process.returncode == STOPPED and seconds < updated.STOP_TERM_WAIT, (process.returncode, seconds)
    assert not running(compileall[0])                                # ended with its process group
    assert not (finalized / ".overlay_consistent").exists()          # no update offered for install
    assert overlay_init.is_file()                                    # nothing had to be killed

    # the next attempt does finalize again from the start and completes it
    dirs = use_dirs(mocker, self.root)
    assert dirs.finalized == finalized
    assert updated.finalize_update() is None
    assert (finalized / ".overlay_consistent").is_file()
    assert not (finalized / "openpilot/zz_slow.py").exists()          # a fresh copy
    assert all(pyc_flags(finalized / name) == CHECKED_HASH for name in COMPILED)


class TestPrecompileInTheUpdater(OpenpilotTestCase):
  """the updater loop; its commands answered by the stand-in"""

  def setup_method(self):
    reset_process_state(self)
    self.params = Params()

  def _new_process(self, mocker) -> FakeCommands:
    self.dirs = use_dirs(mocker, tmp_root(self))
    self.dirs.basedir.mkdir()
    self.commands = FakeCommands(self.dirs).install(mocker)
    set_flag = updated.set_consistent_flag

    def set_flag_in_order(consistent: bool) -> None:  # the flag, in order with the commands
      self.commands.calls.append((["<flag>", consistent], None))
      set_flag(consistent)
    mocker.patch.object(updated, "set_consistent_flag", set_flag_in_order)
    return self.commands

  def test_runs_after_both_resets_and_before_the_flag(self, mocker):
    commands = self._new_process(mocker)
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_SUCCESS]
    finalized = str(self.dirs.finalized)
    flag_set = [cmd for cmd, _ in commands.calls].index(["<flag>", True])
    assert [cmd for cmd, cwd in commands.calls[flag_set - 3:flag_set + 1]] == [
      ["git", "reset", "--hard"],
      ["git", "submodule", "foreach", "--recursive", "git", "reset", "--hard"],
      [sys.executable, "-m", "compileall", "-q", "-f", "--invalidation-mode", "checked-hash", "-x", updated.PRECOMPILE_SKIP, "openpilot"],
      ["<flag>", True],
    ]
    assert all(cwd == finalized for _, cwd in commands.calls[flag_set - 3:flag_set])
    assert [cwd for cmd, cwd in commands.calls if "compileall" in cmd] == [finalized]   # once, in the finalized copy
    assert "unchecked-hash" not in updated.precompile_command(["openpilot"])

  def test_file_that_does_not_compile_is_kept_as_information_only(self, mocker):
    commands = self._new_process(mocker)
    commands.fail_on(sys.executable, "-m", "compileall", output=COMPILE_ERROR)
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_SUCCESS]  # a successful attempt: the long wait
    assert self.params.get("UpdateFailedCount") == 0
    assert self.params.get("LastUpdateException") is None and self.params.get("LastUpdateTime") is not None
    assert (self.dirs.finalized / ".overlay_consistent").is_file()
    assert len(commands.commands("git", "fetch")) == 1               # not retried
    record = self.params.get("UpdaterLastFailure")
    assert (record["stage"], record["rc"], record["reason"]) == ("precompile", 1, "update continues; 1 file(s) not compiled: openpilot/broken.py")
    assert "retry" not in record

  def test_software_screen_shows_no_failure_for_it(self, mocker):
    from openpilot.selfdrive.ui.tests.test_software_update_failure import load_layout
    commands = self._new_process(mocker)
    commands.fail_on(sys.executable, "-m", "compileall", output=COMPILE_ERROR)
    run_main(self, mocker, 2)
    software, params = load_layout()
    layout = software.SoftwareLayout()
    params.update({key: self.params.get(key) for key in ("UpdaterState", "UpdateFailedCount", "UpdaterFetchAvailable", "LastUpdateTime",
                                                         "UpdaterLastFailure")})
    layout._update_state()
    assert layout._download_btn.action_item.value.startswith("up to date, last checked")
    assert layout._download_btn.description == ""

  def test_a_real_failure_after_it_replaces_it(self, mocker):
    commands = self._new_process(mocker)
    commands.fail_on(sys.executable, "-m", "compileall", output=COMPILE_ERROR)
    commands.fail_on("git", "ls-remote", "--heads", after=1, times=1)
    assert run_main(self, mocker, 3) == [FIRST_WAIT, AFTER_SUCCESS, 5 * 60]
    assert self.params.get("UpdaterLastFailure")["stage"] == "check" and self.params.get("UpdateFailedCount") == 1


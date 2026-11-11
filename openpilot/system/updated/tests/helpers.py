"""Shared parts of the updater tests: local git repositories (file:// only, no network), a stand-in for the updater's
commands, and a way to run updated.main() for a set number of loops."""
import datetime
import shutil
import signal
import subprocess
import tempfile
import types
from pathlib import Path

from openpilot.common.params import Params
from openpilot.system.updated import updated

BRANCH = "updater-test"

# git as the tests use it: no user or system configuration, a fixed identity, and the file transport allowed for
# submodules (git >= 2.38 refuses it by default). Environment only: the updater's own commands stay as they are.
GIT_ENV = {
  "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
  "GIT_AUTHOR_NAME": "updater test", "GIT_AUTHOR_EMAIL": "updater-test@example.invalid",
  "GIT_COMMITTER_NAME": "updater test", "GIT_COMMITTER_EMAIL": "updater-test@example.invalid",
  "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "protocol.file.allow", "GIT_CONFIG_VALUE_0": "always",
}


def git(cwd, *args: str) -> str:
  return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def locks(repo) -> list[str]:
  """every *.lock under the repository's .git"""
  return sorted(str(p.relative_to(repo)) for p in Path(repo, ".git").rglob("*.lock"))


def running(pid: int) -> bool:
  """the process exists and has not exited (an exited process stays listed until its parent has collected it)"""
  try:
    with open(f"/proc/{pid}/stat") as f:
      return f.read().rsplit(")", 1)[1].split()[0] != "Z"
  except FileNotFoundError:
    return False


def tmp_root(test) -> Path:
  root = Path(tempfile.mkdtemp(prefix="updater_test_"))
  test.addCleanup(shutil.rmtree, root, ignore_errors=True)
  return root


def reset_process_state(test) -> None:
  """each test starts as a new updated process does, and leaves no signal handler behind"""
  def reset():
    state = getattr(updated, "ProcessState", None)  # not there in an updated.py from before the fix (test_success_trace.py)
    if state is not None:
      state.overlay_built = state.in_command = state.stop_requested = False
  reset()
  test.addCleanup(reset)
  for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
    test.addCleanup(signal.signal, sig, signal.getsignal(sig))


def updater_dirs(root: Path, basedir: Path | None = None) -> types.SimpleNamespace:
  basedir = basedir or root / "openpilot"
  staging = root / "safe_staging"
  return types.SimpleNamespace(root=root, basedir=basedir, staging=staging, upper=staging / "upper", metadata=staging / "metadata",
                               merged=staging / "merged", finalized=staging / "finalized", overlay_init=basedir / ".overlay_init")


def updater_settings(d: types.SimpleNamespace) -> dict:
  """the updater's module constants (its paths are read from the environment at import) and its view of the build, for a
  device whose install is d.basedir and whose staging area is d.staging"""
  return {
    "BASEDIR": str(d.basedir), "STAGING_ROOT": str(d.staging), "OVERLAY_UPPER": str(d.upper), "OVERLAY_METADATA": str(d.metadata),
    "OVERLAY_MERGED": str(d.merged), "FINALIZED": str(d.finalized), "OVERLAY_INIT": d.overlay_init, "LOCK_FILE": str(d.root / "overlay.lock"),
    "get_build_metadata": lambda: types.SimpleNamespace(tested_channel=False),
  }


def use_dirs(mocker, root: Path, basedir: Path | None = None) -> types.SimpleNamespace:
  """point the updater at folders in root"""
  d = updater_dirs(root, basedir)
  for name, value in updater_settings(d).items():
    mocker.patch.object(updated, name, value)
  mocker.patch("os.sync")  # set_consistent_flag() syncs all disks, twice
  return d


class Remotes:
  """A remote with one submodule, as bare repositories reached by file:// URLs.

  commit_sub() / commit_main() add a pushed commit; missing_sub_commit() gives a commit id that names no commit
  anywhere, which is what the submodule pointer recorded by 4186aea in the car repository is (a typing error)."""

  def __init__(self, root: Path):
    self.root = root
    self.sub_url = f"file://{root}/remote/sub.git"
    self.main_url = f"file://{root}/remote/main.git"
    self._sub = root / "work/sub"
    self._main = root / "work/main"
    self._n = 0
    for bare, work, url in ((root / "remote/sub.git", self._sub, self.sub_url), (root / "remote/main.git", self._main, self.main_url)):
      bare.mkdir(parents=True)
      git(bare, "init", "-q", "--bare", "-b", BRANCH)
      work.mkdir(parents=True)
      git(work, "init", "-q", "-b", BRANCH)
      git(work, "remote", "add", "origin", url)
    first = self.commit_sub()
    (self._main / "openpilot/sunnypilot/common").mkdir(parents=True)
    (self._main / "openpilot/sunnypilot/common/version.h").write_text('#define SUNNYPILOT_VERSION "0.0.1"\n')
    (self._main / "CHANGELOG.md").write_text("updater test\n============\n\n* nothing\n")
    (self._main / ".gitignore").write_text(".overlay_init\n.overlay_consistent\n")  # as in openpilot's .gitignore
    git(self._main, "submodule", "add", "-q", self.sub_url, "sub")
    git(self._main, "add", "-A")
    self.commit_main(first)

  def _commit(self, work: Path, push: bool) -> str:
    self._n += 1
    (work / "file.txt").write_text(f"{self._n}\n")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", f"commit {self._n}")
    if push:
      git(work, "push", "-q", "origin", BRANCH)
    return git(work, "rev-parse", "HEAD")

  def commit_sub(self) -> str:
    return self._commit(self._sub, push=True)

  def missing_sub_commit(self) -> str:
    """the id of a pushed submodule commit with its last ten digits changed"""
    sha = git(self._sub, "rev-parse", "HEAD")
    return sha[:30] + "".join("0" if c != "0" else "1" for c in sha[30:])

  def commit_main(self, sub_sha: str, files: dict[str, str] | None = None) -> str:
    """a pushed commit of the main repository whose submodule pointer is sub_sha (which need not be fetchable)"""
    for name, content in (files or {}).items():
      (self._main / name).write_text(content)
      git(self._main, "add", "--", name)
    git(self._main, "update-index", "--add", "--cacheinfo", f"160000,{sub_sha},sub")
    return self._commit_index(self._main)

  def _commit_index(self, work: Path) -> str:
    self._n += 1
    git(work, "commit", "-q", "--allow-empty", "-m", f"commit {self._n}")
    git(work, "push", "-q", "origin", BRANCH)
    return git(work, "rev-parse", "HEAD")

  def install(self, path: Path) -> Path:
    """a device install, as the installer makes it: depth 1, submodules initialised"""
    git(self.root, "clone", "-q", "--depth=1", "--recurse-submodules", "-b", BRANCH, self.main_url, str(path))
    return path


class LoopExit(BaseException):
  """ends updated.main() from inside its wait"""


def run_main(test, mocker, waits: int, during_wait=None) -> list[float]:
  """updated.main() until its wait number `waits` (1 = the first) would start; returns the lengths of all waits asked for.
  during_wait(n, wait_helper) runs in place of wait n, e.g. to press a button."""
  asked: list[float] = []

  def sleep(self, t):
    asked.append(t)
    if len(asked) >= waits:
      raise LoopExit
    if during_wait is not None:
      during_wait(len(asked), self)

  mocker.patch.object(updated.WaitTimeHelper, "sleep", sleep)
  mocker.patch.object(updated, "system_time_valid", lambda: True)
  with test.assertRaises(LoopExit):
    updated.main()
  return asked


class FakeCommands:
  """Stand-in for updated.run(): answers the updater's commands the way git, sudo and find would for a device on
  commit `installed` whose remote branch is at `remote`, without running anything. The staging area is a plain
  folder; "mounting" copies BASEDIR into it. fail_on() makes commands fail."""

  def __init__(self, dirs: types.SimpleNamespace, installed: str = "a" * 40, remote: str = "b" * 40, branch: str = "inster"):
    self.dirs = dirs
    self.remote = remote
    self.branch = branch
    self.heads = {str(dirs.basedir): installed}
    self.mounted = False
    self.calls: list[tuple[list[str], str | None]] = []
    self._failures: dict[tuple[str, ...], list] = {}
    (dirs.basedir / "openpilot/sunnypilot/common").mkdir(parents=True)
    (dirs.basedir / "openpilot/sunnypilot/common/version.h").write_text('#define SUNNYPILOT_VERSION "0.0.1"\n')
    (dirs.basedir / "CHANGELOG.md").write_text("updater test\n============\n\n* nothing\n")
    (dirs.basedir / ".git").mkdir()

  def install(self, mocker) -> "FakeCommands":
    mocker.patch.object(updated, "run", self)
    mocker.patch("os.path.ismount", lambda p: self.mounted and str(p) == str(self.dirs.merged))
    return self

  def fail_on(self, *prefix: str, times: int | None = None, after: int = 0, output: str = "fatal: scripted failure\n", cwd=None) -> None:
    """commands that start with prefix (and run in cwd, if given) fail with exit code 1 and this output:
    `times` times (None = always), after `after` of them have passed"""
    self._failures[prefix] = [after, times, output, None if cwd is None else str(cwd)]

  def commands(self, *prefix: str) -> list[list[str]]:
    return [cmd for cmd, _ in self.calls if tuple(cmd[:len(prefix)]) == prefix]

  def __call__(self, cmd: list[str], cwd: str | None = None) -> str:
    self.calls.append((list(cmd), cwd))
    for prefix, failure in self._failures.items():
      if tuple(cmd[:len(prefix)]) == prefix and failure[3] in (None, cwd):
        if failure[0] > 0:
          failure[0] -= 1
        elif failure[1] is None or failure[1] > 0:
          failure[1] = None if failure[1] is None else failure[1] - 1
          raise subprocess.CalledProcessError(1, cmd, output=failure[2])

    d = self.dirs
    if cmd[:2] == ["sudo", "umount"]:
      self.mounted = False
    elif cmd[:3] == ["sudo", "rm", "-rf"]:
      shutil.rmtree(cmd[3], ignore_errors=True)
      self.heads = {str(d.basedir): self.heads[str(d.basedir)]}
      self.mounted = False
    elif cmd[:2] == ["sudo", "mount"]:
      shutil.copytree(d.basedir, d.merged, dirs_exist_ok=True)
      self.heads[str(d.merged)] = self.heads[str(d.basedir)]
      self.mounted = True
    elif cmd[:2] == ["git", "rev-parse"]:
      return (self.branch if "--abbrev-ref" in cmd else self.heads.get(cwd, "0" * 40)) + "\n"
    elif cmd[:2] == ["git", "ls-remote"]:
      return f"{self.remote}\t{'refs/heads/' + self.branch if '--heads' in cmd else 'HEAD'}\n"
    elif cmd[:2] == ["git", "show"]:
      return "1762776000\n"  # 10 Nov 2025 12:00 UTC: the same day in every time zone
    elif cmd[:2] == ["git", "checkout"]:
      self.heads[cwd] = self.remote
    elif cmd[:3] == ["git", "reset", "--hard"] and cwd == str(d.finalized):
      self.heads[cwd] = self.heads[str(d.merged)]
    return ""


class Recorder:
  """Every command the updater runs and every param it writes or removes, in order, without the parts that differ
  from run to run (the folder, times)."""

  def __init__(self, root: Path):
    self.root = str(root)
    self.trace: list[list] = []

  def _plain(self, value):
    if isinstance(value, (bytes, bytearray)):
      value = value.decode("utf-8", "replace")
    if isinstance(value, datetime.datetime):
      return "<time>"
    if isinstance(value, str):
      return value.replace(self.root, "<root>")
    if isinstance(value, (list, tuple)):
      return [self._plain(v) for v in value]
    if isinstance(value, dict):
      return {k: self._plain(v) for k, v in value.items()}
    return value

  def install(self, mocker, commands) -> "Recorder":
    def run(cmd, cwd=None):
      self.trace.append(["run", self._plain(cmd), self._plain(cwd)])
      return commands(cmd, cwd)

    def wrap(name):
      original = getattr(Params, name)

      def method(params, key, *args, **kwargs):
        key_text = key.decode() if isinstance(key, bytes) else key
        self.trace.append([name, key_text, *self._plain(args)])
        return original(params, key, *args, **kwargs)
      mocker.patch.object(Params, name, method)

    mocker.patch.object(updated, "run", run)
    for name in ("put", "put_bool", "remove"):
      wrap(name)
    return self

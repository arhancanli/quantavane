"""Run an untrusted candidate program in a separate process with no network, no writes outside its own temp folder,
and hard CPU, wall-clock, memory and disk limits. The program must define construct(); its return value comes back as
JSON."""
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

MARKER = "@@QV-RESULT@@"
MAX_OUTPUT = 4_000_000
MAX_RSS_MB = 2048   # macOS ignores RLIMIT_AS, so a watchdog polls the candidate's resident memory instead
MAX_DISK_MB = 256   # RLIMIT_FSIZE caps each file; this caps the whole temp folder
HARNESS = (
    "import json, runpy, sys\n"
    "g = runpy.run_path(sys.argv[1], run_name='candidate')\n"
    "out = g['construct']()\n"
    "enc = lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)\n"
    f"sys.stdout.write('\\n{MARKER}' + json.dumps(out, default=enc))\n"
)
# macOS Seatbelt profile, deny by default. The candidate may run only this interpreter (no other programs, no forking),
# read only the Python install, system libraries and its own temp folder, and write only inside that folder.
PROFILE = """(version 1)
(deny default)
(allow process-exec (literal "{exe}") (literal "{real_exe}"))
(allow file-read* (subpath "/System") (subpath "/usr/lib") (subpath "/usr/share") (subpath "/private/var/db/dyld")
  (subpath "{prefix}") (subpath "{base}") (subpath "{tmp}")
  (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random") (literal "/private/etc/localtime"))
(allow file-read-metadata)
(allow file-write* (subpath "{tmp}") (literal "/dev/null"))
(allow sysctl-read)
(allow signal (target self))
(allow process-info* (target self))
"""


@dataclass
class RunResult:
    ok: bool
    output: object = None
    error: str = ""
    seconds: float = 0.0


def _limits(cpu_seconds):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT + 1024, MAX_OUTPUT + 1024))
    return apply


def _rss_mb(pid):
    """Resident memory of one process in MB (the sandbox denies forking, so the candidate is a single process)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip() or 0) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _folder_mb(path, max_entries=5000):
    """Total size of the files under path in MB; inf when there are suspiciously many entries."""
    total, seen, stack = 0, 0, [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for e in entries:
            seen += 1
            if seen > max_entries:
                return float("inf")
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                else:
                    total += e.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total / 2 ** 20


def run(source, timeout=60, python=sys.executable, files=None, max_rss_mb=MAX_RSS_MB, max_disk_mb=MAX_DISK_MB):
    import time
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="qv-"))
    path = os.path.join(tmp, "candidate.py")
    with open(path, "w") as f:
        f.write(source)
    for name, content in (files or {}).items():  # inputs the candidate may read, e.g. a covering to start from
        with open(os.path.join(tmp, os.path.basename(name)), "w") as f:
            f.write(content)
    env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    profile = PROFILE.format(tmp=tmp, exe=python, real_exe=os.path.realpath(python), prefix=os.path.realpath(sys.prefix),
                             base=os.path.realpath(sys.base_prefix))
    cmd = ["sandbox-exec", "-p", profile, python, "-I", "-c", HARNESS, path]
    start = time.monotonic()
    # Output goes to files that RLIMIT_FSIZE caps, so a flood can't fill memory. They are read back through the same open
    # descriptors, never reopened by name, so a candidate that swaps them for links to other files gains nothing.
    try:
        with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
            proc = subprocess.Popen(cmd, cwd=tmp, env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                                    preexec_fn=_limits(int(timeout)))
            # The CPU limit is the real cap. The wall clock gets 3x room, because on a busy machine a run gets less
            # than a full core; a 5 s margin made every run time out on 2026-10-22 under load ~40. Memory and the temp
            # folder are polled twice a second, since macOS enforces neither for us.
            deadline, timed_out, over = start + timeout * 3 + 30, False, ""
            try:
                while True:
                    try:
                        proc.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if time.monotonic() > deadline:
                        timed_out = True
                        break
                    if _rss_mb(proc.pid) > max_rss_mb:
                        over = f"used more than {max_rss_mb} MB of memory"
                        break
                    if _folder_mb(tmp) > max_disk_mb:
                        over = f"wrote more than {max_disk_mb} MB to its folder"
                        break
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            secs = time.monotonic() - start
            if over:
                return RunResult(False, error=over, seconds=secs)
            if timed_out:
                return RunResult(False, error=f"timed out after {secs:.0f}s wall clock ({timeout}s CPU allowed)",
                                 seconds=secs)
            fo.seek(0)
            out = fo.read(MAX_OUTPUT + 1)
            fe.seek(max(0, os.fstat(fe.fileno()).st_size - 20_000))
            err = fe.read(20_000)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if len(out) > MAX_OUTPUT:
        return RunResult(False, error="output too large", seconds=secs)
    text = out.decode(errors="replace")
    if proc.returncode != 0 or MARKER not in text:
        # The error line itself first, then the candidate's own frames; the runner's frames carry no information.
        lines = [l for l in err.decode(errors="replace").strip().splitlines() if "<frozen runpy>" not in l and 'File "<string>"' not in l]
        tail = lines[-1:] + [l for l in lines[-6:-1] if "candidate.py" in l or l.startswith("    ")]
        tail = [l.replace(path, "candidate.py") for l in tail]
        reason = "CPU limit reached" if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL) else "\n".join(tail) or f"exit {proc.returncode}"
        return RunResult(False, error=reason[:1500], seconds=secs)
    try:
        return RunResult(True, json.loads(text.rsplit(MARKER, 1)[1]), seconds=secs)
    except json.JSONDecodeError as e:
        return RunResult(False, error=f"output is not JSON: {e}", seconds=secs)

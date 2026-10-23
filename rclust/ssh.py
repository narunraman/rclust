"""OpenSSH transport with reusable authentication and non-interactive commands."""

import hashlib
import logging
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

# Printed before a command run in a login shell: anything a site's profile prints before it
# (banners, module messages) is dropped from the command's output.
OUTPUT_MARKER = "RCLUST_OUTPUT_START"
# A shared connection whose network is gone (laptop asleep, Wi-Fi changed) otherwise lingers and
# makes every command hang; with these, ssh closes it after about a minute of silence.
KEEPALIVE = ["-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]


class NotConnectedError(RuntimeError):
    """No usable SSH connection to a cluster."""


def quote_remote_path(path: str) -> str:
    """Quote one path for a remote *shell command* (ssh), allowing a leading ~/ to mean remote home."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def rsync_remote_path(path: str) -> str:
    """A remote path for an rsync destination, never relying on remote-shell expansion.

    rsync >= 3.2.4 protects its arguments from the remote shell, so neither `~` nor `$HOME` is
    expanded there. Both rsync and ssh resolve relative paths against the remote home directory,
    so a leading `~/` is simply dropped ("~" alone becomes ".").
    """
    if path in ("~", "~/"):
        return "."
    if path.startswith("~/"):
        path = path[2:].lstrip("/") or "."
    if path.startswith("-"):
        path = "./" + path  # never let a remote path look like an option
    return path


def _rsync_env() -> dict:
    """Environment for rsync so one quoting rule works with every version.

    With RSYNC_OLD_ARGS=1, rsync >= 3.2.4 hands the remote path to the remote shell as older
    versions (and openrsync) always do, so a shell-quoted path means the same thing everywhere.
    RSYNC_PROTECT_ARGS would switch older versions to the opposite rule, so it is dropped.
    """
    env = {k: v for k, v in os.environ.items() if k != "RSYNC_PROTECT_ARGS"}
    env["RSYNC_OLD_ARGS"] = "1"
    return env


class SSHClient:
    # What to tell the user to run to log in; tools built on rclust can name their own command.
    connect_command = "rclust connect"

    def __init__(self, host: str, user: Optional[str] = None,
                 key_path: Optional[str] = None, name: Optional[str] = None):
        if not isinstance(host, str) or not host or host.startswith("-") or any(c.isspace() for c in host):
            raise ValueError("SSH host must be a hostname or SSH alias")
        if user is not None and (not isinstance(user, str) or not user or user.startswith("-")
                                 or any(c.isspace() for c in user)):
            raise ValueError("SSH user must be a username")
        self.host = host
        self.user = user
        self.key_path = str(Path(key_path).expanduser()) if key_path else None
        self.name = name
        self._private_socket_dir = None
        self.last_rsync_error: Optional[str] = None  # rsync's stderr (last lines) after a failed copy
        self.socket_path = Path(self._get_control_master_path())

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def _get_control_master_path(self) -> str:
        """Use the actual destination's SSH configuration, including user and port."""
        config = {}
        try:
            result = subprocess.run(self._get_base_flags() + ["-G", self.destination],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                config = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
        except (OSError, subprocess.TimeoutExpired):
            pass
        # ssh -G may leave tokens in ControlPath on some OpenSSH versions.
        host = config.get("hostname", self.host)
        user = config.get("user", self.user or "")
        port = config.get("port", "22")
        identity = socket.gethostname() + host + port + user
        digest = hashlib.sha1(identity.encode()).hexdigest()
        path = config.get("controlpath")
        if path and path.lower() != "none":
            tokens = {"%h": host, "%p": port, "%r": user, "%n": self.host,
                      "%l": socket.gethostname(), "%L": socket.gethostname().split(".")[0],
                      "%C": digest, "%i": str(os.getuid())}
            path = path.replace("%%", "\0")
            for token, value in tokens.items():
                path = path.replace(token, value)
            return str(Path(path.replace("\0", "%")).expanduser())
        self._private_socket_dir = Path(tempfile.gettempdir()) / f"rclust-ssh-{os.getuid()}"
        return str(self._private_socket_dir / digest)

    def _get_base_flags(self) -> List[str]:
        # Keep OpenSSH's host-key verification and the user's SSH configuration.
        args = ["ssh", "-o", "ConnectTimeout=10"]
        if self.key_path:
            args.extend(["-i", self.key_path])
        return args

    def is_connected(self) -> bool:
        """Whether a shared (master) connection is open, checked locally."""
        if not self.socket_path.exists():
            return False
        try:
            ret = subprocess.run(self._get_base_flags() + ["-O", "check", "-S",
                                  str(self.socket_path), self.destination], capture_output=True, timeout=10)
            return ret.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    _is_connected = is_connected  # older name

    @property
    def _label(self) -> str:
        return self.name or self.host

    def _not_connected(self, detail: str) -> str:
        return f"not connected to {self._label} ({detail}); run `{self.connect_command} {self._label}`"

    def connect(self, persist: Optional[str] = None, interactive: bool = True):
        """Authenticate once; reuse the master for subsequent commands.

        Interactive: the terminal is passed through so password and two-factor prompts can be
        answered. Otherwise the login must succeed without prompting (e.g. SSH keys), and
        NotConnectedError says to connect interactively.

        `persist` sets ControlPersist (how long an idle master stays open). Without it the user's
        ssh config decides; if that sets nothing, the backgrounded master stays open until it is
        closed (`ssh -O exit`), the machine sleeps or reboots, or the network drops (noticed
        within about a minute: KEEPALIVE).
        """
        if self.is_connected():
            return
        self.socket_path.unlink(missing_ok=True)  # left behind by a master that has exited
        if self._private_socket_dir:
            self._private_socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self._private_socket_dir.is_symlink() or self._private_socket_dir.stat().st_uid != os.getuid():
                raise RuntimeError("Unsafe SSH socket directory")
            self._private_socket_dir.chmod(0o700)
        cmd = self._get_base_flags() + KEEPALIVE + ["-M", "-S", str(self.socket_path)]
        if persist is not None:  # otherwise defer to ControlPersist in the user's ssh config
            cmd.extend(["-o", f"ControlPersist={persist}"])
        cmd.extend(["-f", "-N"])
        if interactive:
            print(f"Connecting to {self._label} ({self.destination}); "
                  "answer any password or two-factor prompts...", file=sys.stderr)
            try:
                ret = subprocess.run(cmd + [self.destination], stdin=sys.stdin, stdout=sys.stdout,
                                     stderr=sys.stderr, timeout=300)
            except subprocess.TimeoutExpired:
                ret, detail = None, "timed out"
            else:
                detail = f"ssh exit code {ret.returncode}"
        else:
            # stderr goes to a file, not a pipe: a pipe could stay open in the backgrounded master
            with tempfile.TemporaryFile("w+") as err:
                try:
                    ret = subprocess.run(cmd + ["-o", "BatchMode=yes", self.destination],
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=err, timeout=60)
                except subprocess.TimeoutExpired:
                    ret = None
                err.seek(0)
                lines = err.read().strip().splitlines()
            detail = "timed out" if ret is None else lines[-1] if lines else f"ssh exit code {ret.returncode}"
        if ret is None or ret.returncode != 0 or not self.is_connected():
            hint = (f"Check that `ssh {self.destination}` works." if interactive else
                    f"Run `{self.connect_command} {self._label}` to log in (it can answer password or "
                    "two-factor prompts).")
            raise NotConnectedError(f"Could not connect to {self._label} ({self.destination}): {detail}. {hint}")

    def execute_command(self, cmd: str, timeout: int = 15, use_login_shell: bool = False,
                        input: Optional[str] = None) -> Tuple[int, str, str]:
        """Run a command on the login node. Returns (exit code, stdout, stderr); never prompts.

        `use_login_shell` runs it under `bash -l`, so site profiles are loaded (on some clusters
        only they put Slurm on the PATH); anything they print is dropped from stdout. `input` is
        sent on stdin (use it for secrets, which must not appear in command lines).

        A lost connection gives exit code 255 and a "not connected" message, including one that
        stopped responding: a command that times out is followed by a quick check, and if the
        connection does not answer either it is closed, so the next `connect` starts afresh.
        """
        real_cmd = f"bash -l -c {shlex.quote(f'echo {OUTPUT_MARKER}; {cmd}')}" if use_login_shell else cmd
        try:
            result = subprocess.run(self._command(real_cmd), input=input, capture_output=True,
                                    text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if self._responsive():
                return -1, "", "SSH command timed out"
            self._close()
            return 255, "", self._not_connected("the connection stopped responding and was closed")
        except OSError as e:
            return -1, "", str(e)
        stdout, stderr = result.stdout, result.stderr.strip()
        if use_login_shell:
            _, found, rest = stdout.partition(OUTPUT_MARKER + "\n")
            stdout = rest if found else stdout
        if result.returncode == 255 and not self.is_connected():
            # BatchMode ssh fails rather than prompting when there is no open connection
            stderr = self._not_connected(stderr.splitlines()[-1] if stderr else "ssh failed")
        return result.returncode, stdout.strip(), stderr

    def _command(self, remote_cmd: str, *options: str) -> List[str]:
        # BatchMode: never prompt here (prompts belong in `connect`, not behind a spinner).
        # ControlMaster=no: only use the shared connection, never start one (ssh configs with
        # `ControlMaster auto` would otherwise make a command try to become the master).
        return self._get_base_flags() + ["-o", "BatchMode=yes", "-o", "ControlMaster=no", *options,
                                         "-S", str(self.socket_path), self.destination, remote_cmd]

    def _responsive(self) -> bool:
        """Whether the shared connection still answers (a trivial command, briefly)."""
        try:
            return subprocess.run(self._command("true"), capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _close(self) -> None:
        """Close a shared connection that no longer answers, so the next `connect` opens a new one."""
        try:
            subprocess.run(self._get_base_flags() + ["-O", "exit", "-S", str(self.socket_path), self.destination],
                           capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.socket_path.unlink(missing_ok=True)  # if the master itself is stuck

    def execute_streaming(self, cmd: str, timeout: int = 300) -> int:
        """Run a command with its output shown live in the terminal. Returns the exit code.

        LogLevel=ERROR hides ssh's "Shared connection to ... closed." line (printed at the INFO
        level after every session with a terminal) but still shows connection errors.
        """
        try:
            return subprocess.run(self._command(cmd, "-o", "LogLevel=ERROR", "-tt"), timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"\nCommand timed out after {timeout}s", file=sys.stderr)
            return -1

    def rsync(self, local_path: str, remote_path: str) -> bool:
        """Copy to the cluster. On failure returns False and sets `last_rsync_error`."""
        # The remote path is relative to the remote home (see rsync_remote_path) and shell-quoted
        # for the remote side (see _rsync_env), so spaces and metacharacters stay one path. The
        # local path is absolute so filenames beginning with '-' cannot become options.
        self.last_rsync_error = None
        target = f"{self.destination}:{shlex.quote(rsync_remote_path(remote_path))}"
        ssh_opt = shlex.join(self._get_base_flags() + ["-o", "BatchMode=yes", "-S", str(self.socket_path)])
        source = str(Path(local_path).absolute())
        if local_path.endswith(os.sep):
            source += os.sep  # rsync distinguishes a directory from its contents.
        rsync_cmd = ["rsync", "-az", "--exclude", ".git", "-e", ssh_opt,
                     source, target]
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600,
                                    env=_rsync_env())
        except subprocess.TimeoutExpired:
            self.last_rsync_error = "rsync timed out"
            return False
        except OSError as e:
            self.last_rsync_error = f"could not run rsync: {e}"
            return False
        if result.returncode != 0:
            lines = [l for l in (result.stderr or "").strip().splitlines() if l.strip()]
            self.last_rsync_error = "\n".join(lines[-3:]) or f"rsync exit code {result.returncode}"
            return False
        return True

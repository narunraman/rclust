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


def quote_remote_path(path: str) -> str:
    """Quote one remote shell path, allowing a leading ~/ to refer to remote home."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


class SSHClient:
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

    def _is_connected(self) -> bool:
        if not self.socket_path.exists():
            return False
        try:
            ret = subprocess.run(self._get_base_flags() + ["-O", "check", "-S",
                                  str(self.socket_path), self.destination], capture_output=True, timeout=10)
            return ret.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def connect(self, persist: Optional[str] = None):
        """Authenticate interactively once; reuse the master for subsequent commands."""
        if self._is_connected():
            return
        if self._private_socket_dir:
            self._private_socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self._private_socket_dir.is_symlink() or self._private_socket_dir.stat().st_uid != os.getuid():
                raise RuntimeError("Unsafe SSH socket directory")
            self._private_socket_dir.chmod(0o700)
        print(f"Connecting to {self.name or self.host} ({self.destination}); "
              "answer any password or two-factor prompts...")
        cmd = self._get_base_flags() + ["-M", "-S", str(self.socket_path)]
        # Bound the background master when the user has not configured a duration.
        cmd.extend(["-o", f"ControlPersist={persist if persist is not None else '1h'}"])
        cmd.extend(["-f", "-N", self.destination])
        ret = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        if ret.returncode != 0:
            raise RuntimeError(f"could not connect to {self.destination} (ssh exit code {ret.returncode})")

    def execute_command(self, cmd: str, timeout: int = 15,
                        use_login_shell: bool = False) -> Tuple[int, str, str]:
        real_cmd = f"bash -l -c {shlex.quote(cmd)}" if use_login_shell else cmd
        # BatchMode: never prompt here (prompts belong in `connect`, not behind a spinner)
        ssh_args = self._get_base_flags() + ["-o", "BatchMode=yes", "-S", str(self.socket_path),
                                           self.destination, real_cmd]
        try:
            result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return -1, "", "SSH command timed out"
        except OSError as e:
            return -1, "", str(e)
        stderr = result.stderr.strip()
        if result.returncode == 255 and not self._is_connected():
            # BatchMode ssh fails rather than prompting when there is no open connection
            detail = stderr.splitlines()[-1] if stderr else "ssh failed"
            name = self.name or self.host
            stderr = f"not connected to {name} ({detail}); run `rclust connect {name}`"
        return result.returncode, result.stdout.strip(), stderr

    def rsync(self, local_path: str, remote_path: str) -> bool:
        # Quote the remote-shell path, including spaces and metacharacters; use an
        # absolute local path so filenames beginning with '-' cannot become options.
        target = f"{self.destination}:{quote_remote_path(remote_path)}"
        ssh_opt = shlex.join(self._get_base_flags() + ["-o", "BatchMode=yes", "-S", str(self.socket_path)])
        source = str(Path(local_path).absolute())
        if local_path.endswith(os.sep):
            source += os.sep  # rsync distinguishes a directory from its contents.
        rsync_cmd = ["rsync", "-az", "--exclude", ".git", "-e", ssh_opt,
                     source, target]
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, timeout=600)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

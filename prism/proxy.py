"""Per-session LiteLLM proxy supervision.

Prism spawns litellm as a **child** on a free ephemeral port, waits for readiness,
then (see cli.py) runs ``claude`` and tears the proxy down when claude exits. No
persistent daemon, so there is no reuse/ownership/config-drift to get wrong.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from . import config as cfgmod

MIN_LITELLM = (1, 83, 0)
READINESS_TIMEOUT_S = 90.0  # cold import of litellm[proxy] + provider SDKs is slow


class ProxyError(Exception):
    """Proxy lifecycle failure (surfaced to the user without a traceback)."""


def pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def litellm_bin() -> Path:
    """The litellm console script in Prism's OWN environment (never bare PATH).

    Under pipx, `litellm` is not on PATH and a foreign one couldn't import `prism`.
    """
    candidate = Path(sys.executable).parent / "litellm"
    if not candidate.exists():
        raise ProxyError(
            f"litellm executable not found at {candidate} — reinstall Prism "
            "(litellm is a dependency and must live in the same environment)."
        )
    return candidate


def installed_litellm_version() -> str:
    from importlib.metadata import version  # `litellm.__version__` raises — use metadata

    return version("litellm")


def _parse_version(v: str) -> tuple[int, int, int]:
    parts = (v.split("+")[0].split("-")[0]).split(".")
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])  # type: ignore[return-value]


def assert_version_ok() -> str:
    v = installed_litellm_version()
    if _parse_version(v) < MIN_LITELLM:
        raise ProxyError(
            f"litellm {v} is below the minimum {'.'.join(map(str, MIN_LITELLM))} "
            "(1.82.7/1.82.8 were a supply-chain compromise). Upgrade litellm."
        )
    return v


class Proxy:
    """A supervised litellm child process."""

    def __init__(self, port: int, gen_config: Path, master_key: str, log_fd: int, popen) -> None:
        self.port = port
        self.gen_config = gen_config
        self.master_key = master_key
        self._log_fd = log_fd
        self._proc = popen

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
        try:
            os.close(self._log_fd)
        except OSError:
            pass


def _wait_ready(proc, port: int, timeout: float, log_file: Path) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health/readiness"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ProxyError(_boot_failure_msg(log_file, "the proxy exited during startup"))
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout, httpx.ReadTimeout):
            pass
        time.sleep(0.25)
    raise ProxyError(_boot_failure_msg(log_file, f"proxy not ready after {timeout:.0f}s"))


def _boot_failure_msg(log_file: Path, why: str) -> str:
    tail = ""
    try:
        lines = log_file.read_text(errors="replace").splitlines()[-15:]
        tail = "\n".join(lines)
    except OSError:
        pass
    return f"{why}. Proxy log tail ({log_file}):\n{tail}"


def _assert_loopback_only(port: int) -> None:
    """Belt-and-suspenders: confirm the listener is bound to loopback only."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # lsof unavailable — --host 127.0.0.1 literal is still the guarantee
    for line in out.splitlines():
        if "LISTEN" not in line:
            continue
        addr = line.split()[8] if len(line.split()) > 8 else ""
        host = addr.rsplit(":", 1)[0]
        if host not in ("127.0.0.1", "[::1]", "localhost"):
            raise ProxyError(
                f"proxy bound a non-loopback address ({addr}) — refusing to expose it. "
                "Check for a HOST env override."
            )


def start(cfg: dict, master_key: str) -> Proxy:
    """Spawn the litellm proxy as a supervised child and block until it is ready."""
    assert_version_ok()
    home = cfgmod.prism_home()
    gen = cfgmod.gen_config_path()
    import yaml

    gen.write_text(yaml.safe_dump(cfgmod.to_litellm_config(cfg)))
    os.chmod(gen, 0o600)

    port = pick_free_port()
    log_file = cfgmod.log_path()
    # Truncate + 0600 so a long-lived user's log can't grow forever / be world-read.
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)

    env = {**os.environ, cfgmod.MASTER_KEY_ENV: master_key}
    proc = subprocess.Popen(
        [str(litellm_bin()), "--config", str(gen), "--host", "127.0.0.1", "--port", str(port)],
        stdout=log_fd, stderr=log_fd, env=env, cwd=str(home), start_new_session=True,
    )
    proxy = Proxy(port, gen, master_key, log_fd, proc)
    try:
        _wait_ready(proc, port, READINESS_TIMEOUT_S, log_file)
        _assert_loopback_only(port)
    except Exception:
        proxy.stop()
        raise
    return proxy

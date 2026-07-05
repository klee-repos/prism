"""Unit tests for proxy helpers (version guard, port, binary resolution)."""
import pytest

from prism import proxy as p


def test_pick_free_port_in_range():
    port = p.pick_free_port()
    assert 1024 <= port <= 65535


def test_parse_version():
    assert p._parse_version("1.91.0") == (1, 91, 0)
    assert p._parse_version("1.83.0") == (1, 83, 0)
    assert p._parse_version("1.82.6") == (1, 82, 6)
    assert p._parse_version("2.0.0rc1".replace("rc1", "-rc1")) == (2, 0, 0)


def test_version_guard_accepts_current(monkeypatch):
    monkeypatch.setattr(p, "installed_litellm_version", lambda: "1.91.0")
    assert p.assert_version_ok() == "1.91.0"


def test_version_guard_rejects_below_minimum(monkeypatch):
    monkeypatch.setattr(p, "installed_litellm_version", lambda: "1.82.6")
    with pytest.raises(p.ProxyError):
        p.assert_version_ok()


def test_version_guard_rejects_compromised_versions(monkeypatch):
    # 1.82.7 / 1.82.8 were the supply-chain compromise; both are < 1.83.0.
    for bad in ("1.82.7", "1.82.8"):
        monkeypatch.setattr(p, "installed_litellm_version", lambda v=bad: v)
        with pytest.raises(p.ProxyError):
            p.assert_version_ok()


def test_litellm_bin_resolves_next_to_interpreter(monkeypatch, tmp_path):
    (tmp_path / "litellm").write_text("#!/bin/sh\n")
    monkeypatch.setattr(p.sys, "executable", str(tmp_path / "python"))
    assert p.litellm_bin() == tmp_path / "litellm"


def test_litellm_bin_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(p.sys, "executable", str(tmp_path / "python"))
    with pytest.raises(p.ProxyError):
        p.litellm_bin()

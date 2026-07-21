"""Unit tests for CLI arg partitioning + Claude Code passthrough fidelity."""
import os
import stat

import pytest

from prism import cli
from prism import config as cfgmod


def test_no_args_is_passthrough():
    assert cli.partition_args([]) == ("passthrough", None, [])


def test_reserved_bare_verbs():
    assert cli.partition_args(["setup"]) == ("mgmt", "setup", [])
    assert cli.partition_args(["status", "x"]) == ("mgmt", "status", ["x"])
    assert cli.partition_args(["doctor"]) == ("mgmt", "doctor", [])


def test_double_dash_escape_forces_passthrough():
    assert cli.partition_args(["--", "setup"]) == ("passthrough", None, ["setup"])


def test_flags_pass_through_verbatim():
    a = ["--dangerously-skip-permissions", "--model", "opus", "-p", "hi there"]
    assert cli.partition_args(a) == ("passthrough", None, a)


def test_informational_only_skips_proxy():
    assert cli.partition_args(["--help"]) == ("passthrough_no_proxy", None, ["--help"])
    assert cli.partition_args(["-v"]) == ("passthrough_no_proxy", None, ["-v"])


def test_help_with_subcommand_is_normal_passthrough():
    assert cli.partition_args(["mcp", "--help"])[0] == "passthrough"


def test_requires_completion():
    assert cli.requires_completion([]) is True
    assert cli.requires_completion(["-p", "hi"]) is True
    assert cli.requires_completion(["mcp", "list"]) is False
    assert cli.requires_completion(["doctor"]) is False
    assert cli.requires_completion(["auth", "login"]) is False
    assert cli.requires_completion(["--help"]) is False


def test_build_claude_env_sets_all_expected():
    cfg = {"mapping": {"opus": "coder", "sonnet": "coder", "haiku": "background"}}
    env = cli.build_claude_env({"PATH": "/x"}, "http://127.0.0.1:5000", cfg)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5000"
    assert env["ANTHROPIC_AUTH_TOKEN"] == cli.LOCAL_AUTH_TOKEN
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "coder"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "coder"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "background"
    assert env["PATH"] == "/x"  # existing env preserved
    assert "PRISM_SEARCH" not in env  # no search section → flag unset


def test_build_claude_env_sets_prism_search_when_configured():
    cfg = {"mapping": {"opus": "coder"}, "search": {"provider": "tavily", "api_key_env": "TAVILY_API_KEY"}}
    env = cli.build_claude_env({}, "http://127.0.0.1:5000", cfg)
    assert env["PRISM_SEARCH"] == "1"


def test_exec_claude_forwards_args_verbatim(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, argv, env=None):
            captured["argv"] = argv
            captured["env"] = env

        def wait(self):
            return 0

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli, "_find_claude", lambda: "/usr/bin/claude")
    args = ["--dangerously-skip-permissions", "--model", "opus", "-p", "hi"]
    rc = cli._exec_claude(args)
    assert rc == 0
    # C5: every flag reaches claude in order, unmodified.
    assert captured["argv"] == ["/usr/bin/claude", *args]


def test_exec_claude_propagates_nonzero_exit(monkeypatch):
    class FakePopen:
        def __init__(self, argv, env=None):
            pass

        def wait(self):
            return 42

    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli, "_find_claude", lambda: "/usr/bin/claude")
    assert cli._exec_claude(["-p", "x"]) == 42


def test_main_prism_error_uses_reserved_exit_code(monkeypatch):
    def boom(_args):
        raise cli.PrismError("nope")

    monkeypatch.setitem(cli._MGMT, "doctor", boom)
    assert cli.main(["doctor"]) == cli.EXIT_PRISM_ERROR


# ── `prism profile` (list + one-word switch) ────────────────────────────────────────

def test_partition_args_profile_is_mgmt():
    assert cli.partition_args(["profile"]) == ("mgmt", "profile", [])
    assert cli.partition_args(["profile", "k3"]) == ("mgmt", "profile", ["k3"])
    # the escape hatch still forwards the word to claude
    assert cli.partition_args(["--", "profile"]) == ("passthrough", None, ["profile"])


def test_set_active_profile_text_guards():
    f = cli._set_active_profile_text
    assert f("active_profile: glm  # x\n", "k3") == "active_profile: k3  # x\n"  # inline comment kept
    assert f('active_profile: "glm"\n', "k3") == 'active_profile: "k3"\n'        # quotes kept
    assert f("active_profile:\n", "k3") is None                                 # empty value -> guard
    assert f("routes:\n  coder: {}\n", "k3") is None                            # no line -> guard (no clobber)
    assert f("active_profile: a\nactive_profile: b\n", "z") is None             # duplicate -> refuse
    # a comment line that merely mentions active_profile is never edited
    r = f("# edit active_profile: here\nactive_profile: glm\n", "k3")
    assert r is not None
    assert "# edit active_profile: here" in r and "active_profile: k3" in r
    # a name with regex-special chars is inserted literally (no backref corruption)
    assert f("active_profile: glm\n", r"a\1b") == "active_profile: a\\1b\n"


@pytest.fixture()
def prism_home_default(tmp_path, monkeypatch):
    """An isolated PRISM_HOME seeded with the SHIPPED default config (so comment-preservation
    assertions are meaningful and the dev's real ~/.prism is never touched)."""
    home = tmp_path / ".prism"
    monkeypatch.setenv("PRISM_HOME", str(home))
    home.mkdir(parents=True)
    cfgmod.config_path().write_text(cfgmod._default_config_text())
    return home


def test_profile_list_marks_active(prism_home_default, capsys):
    rc = cli.cmd_profile([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "active profile: glm" in out
    assert "* glm" in out and "k3" in out
    assert "moonshotai/kimi-k3" in out  # k3's coder + multimodal model is shown


def test_profile_switch_flips_and_preserves_comments(prism_home_default):
    assert cli.cmd_profile(["k3"]) == 0
    text = cfgmod.config_path().read_text()
    assert any(l.strip().startswith("active_profile: k3") for l in text.splitlines())
    # the shipped default's comments survived the in-place rewrite (a whole-file rewrite fails here)
    assert "# Prism config — edit freely" in text
    assert "the one word that switches everything" in text
    # and the config now resolves to k3 for text AND vision
    cfg = cfgmod.load_config()
    assert cfgmod._route_model_string(cfg, "coder") == "openrouter/moonshotai/kimi-k3"
    assert cfgmod._route_model_string(cfg, "multimodal") == "openrouter/moonshotai/kimi-k3"


def test_profile_switch_preserves_0600_perms(prism_home_default):
    cli.cmd_profile(["k3"])
    assert stat.S_IMODE(os.stat(cfgmod.config_path()).st_mode) == 0o600


def test_profile_switch_idempotent_message(prism_home_default, capsys):
    cli.cmd_profile(["glm"])
    assert "already on profile 'glm'." in capsys.readouterr().out


def test_profile_switch_unknown_name_errors(prism_home_default):
    with pytest.raises(cli.PrismError):
        cli.cmd_profile(["nope"])


def test_profile_switch_ignores_flags(prism_home_default, capsys):
    # A stray flag must not be treated as a profile name (would give a confusing 'unknown profile').
    rc = cli.cmd_profile(["--help"])  # no positional name -> falls back to list
    assert rc == 0 and "active profile:" in capsys.readouterr().out


def test_profile_on_legacy_config_errors_clearly(tmp_path, monkeypatch):
    home = tmp_path / ".prism"
    monkeypatch.setenv("PRISM_HOME", str(home))
    home.mkdir(parents=True)
    cfgmod.config_path().write_text(
        "schema_version: 1\n"
        "providers:\n  openrouter: {type: openrouter, api_key_env: OPENROUTER_API_KEY}\n"
        "routes:\n"
        "  coder: {provider: openrouter, model: z-ai/glm-5.2}\n"
        "  background: {provider: openrouter, model: z-ai/glm-4.7-flash}\n"
        "  multimodal: {provider: openrouter, model: google/gemini-2.5-flash}\n"
        "mapping: {opus: coder, sonnet: coder, haiku: background}\n"
    )
    with pytest.raises(cli.PrismError):
        cli.cmd_profile(["k3"])   # switch on a no-profiles config
    with pytest.raises(cli.PrismError):
        cli.cmd_profile([])       # list on a no-profiles config


def test_status_on_bad_active_profile_prints_config_error_not_traceback(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".prism"
    monkeypatch.setenv("PRISM_HOME", str(home))
    home.mkdir(parents=True)
    text = cfgmod._default_config_text().replace("active_profile: glm", "active_profile: ghost")
    cfgmod.config_path().write_text(text)
    rc = cli.cmd_status([])  # must NOT raise a KeyError/traceback
    out = capsys.readouterr().out
    assert rc == 0 and "config error" in out


def test_status_shows_active_profile_line(prism_home_default, capsys):
    cli.cmd_status([])
    out = capsys.readouterr().out
    assert "active profile -> glm" in out
    assert "coder" in out and "z-ai/glm-5.2" in out

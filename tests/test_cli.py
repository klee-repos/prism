"""Unit tests for CLI arg partitioning + Claude Code passthrough fidelity."""
from prism import cli


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

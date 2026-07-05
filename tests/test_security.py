"""Security invariants that are cheap to assert as unit tests."""
import re
from pathlib import Path

# A real-looking key: sk- followed by 20+ base64-ish chars. Our fixtures use short
# dummies ("sk-dummy-not-real") which must NOT match.
KEY_RE = re.compile(r"sk-[A-Za-z0-9]{20,}")


def test_no_real_api_keys_committed_in_tests():
    tests_dir = Path(__file__).parent
    for f in tests_dir.rglob("*.py"):
        text = f.read_text()
        hits = KEY_RE.findall(text)
        assert not hits, f"possible real key literal in {f.name}: {hits}"


def test_to_litellm_config_never_inlines_a_literal_key():
    # Keys must be env references (os.environ/VAR), never literal values on disk.
    from prism import config as c

    out = c.to_litellm_config(c.default_config())
    for m in out["model_list"]:
        api_key = m["litellm_params"]["api_key"]
        assert api_key.startswith("os.environ/"), api_key
    assert out["general_settings"]["master_key"].startswith("os.environ/")

"""Tests for the config profile overlay system (config.load_config / _deep_merge).

Profiles are thin overlays in profiles/ that change only per-agent
provider/model/host; everything else (prompts, tool allowlists, most settings)
is inherited from agents.yaml. These tests pin that contract.
"""
import os

import pytest

import config


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

def test_deep_merge_overrides_scalar_keeps_siblings():
    base = {"agents": {"recon": {"provider": "anthropic", "model": "haiku", "tools": ["nmap*"]}}}
    overlay = {"agents": {"recon": {"model": "opus"}}}
    out = config._deep_merge(base, overlay)
    # Overridden field changes...
    assert out["agents"]["recon"]["model"] == "opus"
    # ...siblings (prompt/tools/provider) are preserved.
    assert out["agents"]["recon"]["provider"] == "anthropic"
    assert out["agents"]["recon"]["tools"] == ["nmap*"]


def test_deep_merge_does_not_mutate_inputs():
    base = {"settings": {"a": 1, "b": 2}}
    overlay = {"settings": {"b": 3}}
    out = config._deep_merge(base, overlay)
    assert out["settings"] == {"a": 1, "b": 3}
    assert base["settings"] == {"a": 1, "b": 2}  # base untouched


def test_deep_merge_list_replaces_not_appends():
    base = {"agents": {"recon": {"tools": ["nmap*", "ffuf*"]}}}
    overlay = {"agents": {"recon": {"tools": ["curl*"]}}}
    out = config._deep_merge(base, overlay)
    assert out["agents"]["recon"]["tools"] == ["curl*"]


# ---------------------------------------------------------------------------
# load_config + shipped profiles
# ---------------------------------------------------------------------------

def test_base_config_loads():
    cfg = config.load_config("agents.yaml")
    assert "supervisor" in cfg["agents"]
    assert cfg["agents"]["recon"]["tools"]  # base carries tool allowlists


def test_profiles_are_discoverable():
    profiles = config.list_profiles()
    # opus/cheap flip the Ollama base to hosted Claude for frontier comparison.
    # There is no "ollama" profile — the base agents.yaml is already Ollama-only.
    assert {"opus", "cheap"} <= set(profiles)
    assert "ollama" not in profiles


def test_base_config_is_ollama_only_with_host_expanded():
    cfg = config.load_config("agents.yaml")  # OLLAMA_HOST defaulted in conftest
    providers = {c["provider"] for c in cfg["agents"].values()}
    assert providers == {"ollama"}
    # host: "${OLLAMA_HOST}" is interpolated from the env, not left literal.
    assert cfg["agents"]["supervisor"]["host"] == os.environ["OLLAMA_HOST"]
    assert "${" not in cfg["agents"]["supervisor"]["host"]


def test_only_refusal_specialist_may_use_an_abliterated_model():
    cfg = config.load_config("agents.yaml")
    for name, c in cfg["agents"].items():
        if name == "refusal_specialist":
            continue
        assert "abliterated" not in c["model"].lower(), name


def test_opus_profile_sets_every_worker_to_opus_but_keeps_prompts():
    base = config.load_config("agents.yaml")
    cfg = config.load_config("agents.yaml", profile="opus")
    workers = ["supervisor", "recon", "exploit", "privesc", "webexplorer", "vulnsearch", "summarizer"]
    for name in workers:
        assert cfg["agents"][name]["provider"] == "anthropic"
        assert "opus" in cfg["agents"][name]["model"]
    # Inherited, not duplicated: prompts and tool lists still come from the base.
    assert cfg["agents"]["recon"]["tools"] == base["agents"]["recon"]["tools"]
    assert cfg["agents"]["recon"]["system_prompt"] == base["agents"]["recon"]["system_prompt"]


def test_cheap_profile_uses_haiku():
    cfg = config.load_config("agents.yaml", profile="cheap")
    assert "haiku" in cfg["agents"]["exploit"]["model"]


def test_refusal_specialist_stays_ollama_across_profiles():
    """refusal_specialist stays on its local (abliterated) model even when a
    profile flips the other agents to a hosted provider."""
    for profile in (None, "opus", "cheap"):
        cfg = config.load_config("agents.yaml", profile=profile)
        assert cfg["agents"]["refusal_specialist"]["provider"] == "ollama"


def test_unknown_profile_raises_with_available_list():
    with pytest.raises(FileNotFoundError) as exc:
        config.load_config("agents.yaml", profile="does-not-exist")
    assert "opus" in str(exc.value)  # error lists what IS available


# ---------------------------------------------------------------------------
# Environment interpolation (${VAR} / ${VAR:-default})
# ---------------------------------------------------------------------------

def test_env_expansion_uses_value_then_default(monkeypatch):
    monkeypatch.setenv("FC_TEST_VAR", "xyz")
    missing = set()
    assert config._expand_env("a-${FC_TEST_VAR}-b", missing) == "a-xyz-b"
    monkeypatch.delenv("FC_TEST_VAR", raising=False)
    assert config._expand_env("${FC_TEST_VAR:-fallback}", missing) == "fallback"
    assert missing == set()  # default satisfied the unset var


def test_env_expansion_collects_missing_unset_vars():
    missing = set()
    out = config._expand_env({"host": "${FC_DEFINITELY_UNSET}"}, missing)
    assert out == {"host": "${FC_DEFINITELY_UNSET}"}  # left untouched
    assert "FC_DEFINITELY_UNSET" in missing


def test_load_config_raises_when_ollama_host_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    with pytest.raises(RuntimeError) as exc:
        config.load_config("agents.yaml")
    assert "OLLAMA_HOST" in str(exc.value)

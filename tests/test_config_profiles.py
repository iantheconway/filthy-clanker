"""Tests for the config profile overlay system (config.load_config / _deep_merge).

Profiles are thin overlays in profiles/ that change only per-agent
provider/model/host; everything else (prompts, tool allowlists, most settings)
is inherited from agents.yaml. These tests pin that contract.
"""
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
    assert {"opus", "cheap", "ollama"} <= set(profiles)


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


def test_ollama_profile_switches_provider_and_lowers_thresholds():
    base = config.load_config("agents.yaml")
    cfg = config.load_config("agents.yaml", profile="ollama")
    assert cfg["agents"]["exploit"]["provider"] == "ollama"
    assert cfg["agents"]["exploit"]["host"]  # host supplied for the local server
    # Settings overlay narrows the context window for small local models.
    assert cfg["settings"]["context_limit_threshold"] < base["settings"]["context_limit_threshold"]


def test_refusal_specialist_stays_local_across_profiles():
    """refusal_specialist must remain on the abliterated local model in every
    profile — its role is to answer where a safety-tuned model refused."""
    for profile in ("opus", "cheap", "ollama"):
        cfg = config.load_config("agents.yaml", profile=profile)
        assert cfg["agents"]["refusal_specialist"]["provider"] == "ollama"


def test_unknown_profile_raises_with_available_list():
    import pytest
    with pytest.raises(FileNotFoundError) as exc:
        config.load_config("agents.yaml", profile="does-not-exist")
    assert "opus" in str(exc.value)  # error lists what IS available

import os

import yaml

_GUIDANCE = (
    "Keep in mind there is a 30 second timeout on tool calls, so "
    "let's avoid scanning too deeply at first. Let's make sure to pause and discuss "
    "findings; try not to chain together too many tool calls without waiting for human "
    "input. Avoid looking up write ups of the specific challenge."
)

# Directory (relative to project root) holding thin per-run config overlays.
_PROFILES_DIR = "profiles"


def _project_root() -> str:
    """Absolute path to the project root (one level above src/)."""
    return os.path.dirname(os.path.dirname(__file__))


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict: overlay merged onto base.

    Nested dicts are merged key-by-key so an overlay can set just one field of an
    agent (e.g. ``model``) without dropping the rest (prompts, tool allowlists).
    Scalars and lists in the overlay replace the base value outright.
    """
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def list_profiles(root: str | None = None) -> list[str]:
    """Names (without extension) of the overlays available in profiles/."""
    root = root or _project_root()
    profiles_dir = os.path.join(root, _PROFILES_DIR)
    if not os.path.isdir(profiles_dir):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(profiles_dir)
        if name.endswith((".yaml", ".yml"))
    )


def _resolve_profile_path(profile: str, root: str) -> str:
    """Map a profile reference to a file path.

    Accepts a bare name ("opus"), a filename ("opus.yaml"), or an explicit path.
    """
    candidates = []
    if os.path.isabs(profile) or os.sep in profile or "/" in profile:
        candidates.append(profile)
    filename = profile if profile.endswith((".yaml", ".yml")) else f"{profile}.yaml"
    candidates.append(os.path.join(root, _PROFILES_DIR, filename))
    candidates.append(os.path.join(root, filename))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    available = ", ".join(list_profiles(root)) or "(none)"
    raise FileNotFoundError(
        f"Config profile '{profile}' not found. Available profiles: {available}"
    )


def load_config(path: str = "agents.yaml", profile: str | None = None) -> dict:
    """Load the agents config, optionally overlaying a profile.

    ``path`` is the base config (agents.yaml). ``profile`` names a thin overlay in
    profiles/ (e.g. "opus", "cheap", "ollama") that only specifies the fields it
    changes — typically each agent's provider/model/host — with everything else
    inherited from the base. Paths are resolved relative to the project root.
    """
    root = _project_root()
    base_path = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"{base_path} not found. Copy agents.yaml.example or check the path."
        )
    with open(base_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if profile:
        overlay_path = _resolve_profile_path(profile, root)
        with open(overlay_path, "r", encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh) or {}
        config = _deep_merge(config, overlay)

    return config


def build_system_prompt(
    machine_name: str | None = None,
    machine_os: str | None = None,
    machine_difficulty: str | None = None,
    machine_ip: str | None = None,
) -> str:
    if machine_name:
        hostname = f"{machine_name.lower()}.htb"
        parts = [
            f"We're going to work on a CTF on the hackthebox platform using available tools. "
            f"The name of the challenge is {machine_name}. We are connected to the VPN and the ip "
            f"has been added to /etc/hosts as {hostname}.",
        ]
        if machine_os or machine_difficulty:
            desc_parts = []
            if machine_difficulty:
                desc_parts.append(f"rated as {machine_difficulty.lower()}")
            if machine_os:
                desc_parts.append(f"{machine_os.lower()} machine")
            else:
                desc_parts.append("machine")
            parts.append(f"The challenge is a {' '.join(desc_parts)}, so adjust strategy accordingly.")
        if machine_ip:
            parts.append(f"The target IP is {machine_ip}.")
        parts.append(f"Let's start with some scanning. {_GUIDANCE}")
        return " ".join(parts)

    return (
        "We're going to work on a CTF on the hackthebox platform using available tools. "
        "No specific machine is targeted yet — use /spawn <name> to start one. "
        f"{_GUIDANCE}"
    )


# Backwards-compatible default for existing code that imports this directly.
DEFAULT_SYSTEM_PROMPT = build_system_prompt()

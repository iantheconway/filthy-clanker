_GUIDANCE = (
    "Keep in mind there is a 30 second timeout on tool calls, so "
    "let's avoid scanning too deeply at first. Let's make sure to pause and discuss "
    "findings; try not to chain together too many tool calls without waiting for human "
    "input. Avoid looking up write ups of the specific challenge."
)


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

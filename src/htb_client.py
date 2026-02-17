"""Thin wrapper around pyhackthebox for filthy-clanker integration."""

from hackthebox import HTBClient


class HTB:
    def __init__(self, app_token: str):
        self._client = HTBClient(app_token=app_token)

    def get_active_machine(self) -> dict | None:
        """Return dict with active machine info, or None."""
        try:
            instance = self._client.get_active_machine(release_arena=False)
        except Exception:
            return None
        if instance is None:
            return None
        m = instance.machine
        return {
            "name": m.name,
            "os": m.os,
            "difficulty": m.difficulty,
            "ip": instance.ip,
            "id": m.id,
        }

    def get_machine_info(self, name_or_id) -> dict | str:
        """Return dict with machine details, or error string."""
        try:
            m = self._client.get_machine(name_or_id)
            return {
                "name": m.name,
                "os": m.os,
                "difficulty": m.difficulty,
                "points": m.points,
                "id": m.id,
                "active": m.active,
                "retired": m.retired,
                "user_owned": m.user_owned,
                "root_owned": m.root_owned,
                "stars": m.stars,
            }
        except Exception as e:
            return f"Error looking up machine: {e}"

    def spawn_machine(self, name_or_id) -> str:
        """Spawn a machine and return its IP, or an error string."""
        try:
            m = self._client.get_machine(name_or_id)
            instance = m.spawn(release_arena=False)
            return instance.ip
        except Exception as e:
            return f"Error spawning machine: {e}"

    def stop_machine(self) -> str:
        """Stop the active machine."""
        try:
            instance = self._client.get_active_machine(release_arena=False)
            if instance is None:
                return "No active machine to stop."
            name = instance.machine.name
            instance.stop()
            return f"Stopped machine '{name}'."
        except Exception as e:
            return f"Error stopping machine: {e}"

    def reset_machine(self) -> str:
        """Reset the active machine."""
        try:
            instance = self._client.get_active_machine(release_arena=False)
            if instance is None:
                return "No active machine to reset."
            name = instance.machine.name
            instance.reset()
            return f"Reset requested for '{name}' (takes ~1 minute)."
        except Exception as e:
            return f"Error resetting machine: {e}"

    def submit_flag(self, flag: str, difficulty: int = 50) -> str:
        """Submit a flag for the active machine."""
        try:
            instance = self._client.get_active_machine(release_arena=False)
            if instance is None:
                return "No active machine to submit a flag for."
            result = instance.machine.submit(flag=flag, difficulty=difficulty)
            return str(result)
        except Exception as e:
            return f"Error submitting flag: {e}"

    def search(self, term: str) -> str:
        """Search HTB and return formatted results."""
        try:
            results = self._client.search(term)
            lines = []
            if results.machines:
                lines.append("Machines:")
                for m in results.machines:
                    lines.append(f"  - {m.name} ({m.os}, {m.difficulty})")
            if results.challenges:
                lines.append("Challenges:")
                for c in results.challenges:
                    lines.append(f"  - {c.name}")
            if not lines:
                return "No results found."
            return "\n".join(lines)
        except Exception as e:
            return f"Error searching: {e}"

    def get_vpn_status(self) -> str:
        """Return current VPN server info."""
        try:
            vpn = self._client.get_current_vpn_server(release_arena=False)
            if vpn is None:
                return "No VPN server assigned."
            return (
                f"VPN Server: {vpn.friendly_name}\n"
                f"Location: {vpn.location}\n"
                f"Active clients: {vpn.current_clients}"
            )
        except Exception as e:
            return f"Error getting VPN status: {e}"

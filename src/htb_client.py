"""HTB integration via the HackTheBox API v4 (direct requests).

The `pyhackthebox` lib went stale against HTB's current API (its `machine/list` /
name-based `machine/profile/{name}` endpoints 404), so this talks to the API directly with
the app token (a JWT). The *action* endpoints are still current: `vm/spawn`, `vm/terminate`,
`vm/reset`, `machine/own`, `machine/active`, and `machine/paginated` for listing. Machine
lookups use a numeric **id** (names only resolve for machines on the paginated page — pass the
id for anything older). Interface preserved for generate_htb_trajectories.py.
"""
from __future__ import annotations

import time

import requests

_BASE = "https://labs.hackthebox.com/api/v4"
_UA = "Mozilla/5.0 (filthy-clanker HTB client)"


class HTB:
    def __init__(self, app_token: str):
        self._token = (app_token or "").strip()
        self._s = requests.Session()
        self._s.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "User-Agent": _UA,
        })

    # ── low-level ────────────────────────────────────────────────────────────────
    def _get(self, path: str, **kw):
        r = self._s.get(f"{_BASE}/{path}", timeout=20, **kw)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json_data: dict | None = None):
        r = self._s.post(f"{_BASE}/{path}", json=json_data or {}, timeout=30)
        # HTB returns 400/200 with a JSON {message}; surface the body either way
        try:
            body = r.json()
        except Exception:
            body = {"message": r.text[:200]}
        return r.status_code, body

    def _resolve_id(self, name_or_id) -> int | None:
        """Numeric id passes through; a name is matched against the paginated list
        (recent machines only — pass the numeric id for older/retired boxes)."""
        s = str(name_or_id).strip()
        if s.isdigit():
            return int(s)
        try:
            data = self._get("machine/paginated?per_page=100").get("data", [])
        except Exception:
            return None
        for m in data:
            if str(m.get("name", "")).lower() == s.lower():
                return int(m["id"])
        for m in data:  # substring fallback
            if s.lower() in str(m.get("name", "")).lower():
                return int(m["id"])
        return None

    # ── machine info / state ─────────────────────────────────────────────────────
    def get_active_machine(self) -> dict | None:
        try:
            info = (self._get("machine/active") or {}).get("info")
        except Exception:
            return None
        if not info:
            return None
        return {"name": info.get("name"), "ip": info.get("ip"),
                "id": info.get("id"), "type": info.get("type")}

    def get_machine_info(self, name_or_id) -> dict | str:
        mid = self._resolve_id(name_or_id)
        if mid is None:
            return f"Error looking up machine: '{name_or_id}' not found (pass a numeric id)."
        try:
            info = self._get(f"machine/profile/{mid}").get("info", {})
            return {"name": info.get("name"), "os": info.get("os"),
                    "difficulty": info.get("difficultyText"), "points": info.get("points"),
                    "id": info.get("id"), "active": info.get("active"),
                    "retired": info.get("retired"), "ip": info.get("ip"),
                    "user_owned": info.get("authUserInUserOwns"),
                    "root_owned": info.get("authUserInRootOwns")}
        except Exception as e:
            return f"Error looking up machine: {e}"

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def spawn_machine(self, name_or_id, wait_sec: int = 240) -> str:
        """Spawn (VIP on-demand) and poll until an IP is assigned; return the IP or an error."""
        mid = self._resolve_id(name_or_id)
        if mid is None:
            return f"Error spawning machine: '{name_or_id}' not found (pass a numeric id)."
        active = self.get_active_machine()
        if active and active.get("id") == mid and active.get("ip"):
            return active["ip"]
        if active and active.get("id") != mid:
            return (f"Error spawning machine: another machine is active "
                    f"('{active.get('name')}' id={active.get('id')}). Stop it first.")
        code, body = self._post("vm/spawn", {"machine_id": mid})
        msg = body.get("message", "")
        if code >= 400 and "deploy" not in msg.lower() and "spawn" not in msg.lower():
            return f"Error spawning machine: HTTP {code}: {msg}"
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            a = self.get_active_machine()
            if a and a.get("id") == mid and a.get("ip"):
                return a["ip"]
            time.sleep(8)
        return f"Error spawning machine: spawned but no IP after {wait_sec}s (msg: {msg})"

    def stop_machine(self) -> str:
        a = self.get_active_machine()
        if not a:
            return "No active machine to stop."
        code, body = self._post("vm/terminate", {"machine_id": a["id"]})
        return f"Stopped '{a.get('name')}': {body.get('message', code)}"

    def reset_machine(self) -> str:
        a = self.get_active_machine()
        if not a:
            return "No active machine to reset."
        code, body = self._post("vm/reset", {"machine_id": a["id"]})
        return f"Reset '{a.get('name')}': {body.get('message', code)}"

    def submit_flag(self, flag: str, difficulty: int = 50) -> str:
        a = self.get_active_machine()
        if not a:
            return "No active machine to submit a flag for."
        code, body = self._post("machine/own",
                                {"flag": flag, "id": a["id"], "difficulty": difficulty})
        return f"HTTP {code}: {body.get('message', body)}"

    # ── misc ─────────────────────────────────────────────────────────────────────
    def search(self, term: str) -> str:
        try:
            data = self._get("machine/paginated?per_page=100").get("data", [])
        except Exception as e:
            return f"Error searching: {e}"
        hits = [m for m in data if term.lower() in str(m.get("name", "")).lower()]
        if not hits:
            return "No results found (paginated list only covers recent machines)."
        return "\n".join(f"  - {m['name']} (id={m['id']}, {m.get('os')}, "
                         f"{m.get('difficultyText')})" for m in hits)

    def get_vpn_status(self) -> str:
        try:
            info = self._get("connection/status")
            return str(info)
        except Exception as e:
            return f"Error getting VPN status: {e}"

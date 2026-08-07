#!/usr/bin/env bash
#
# Eval sandbox entrypoint (SPEC: filthy-clanker-agent-solve-quality — internet
# isolation). Blocks the agent from reaching arbitrary public hosts while keeping
# the things the harness legitimately needs:
#
#   ALLOWED  • loopback + established connections
#            • the Docker host (host.docker.internal) — local Ollama AND the
#              sibling challenge containers' published ports live here
#            • DNS
#            • ONLY the domains in proxy-allowlist.txt (Brave Search, LangSmith,
#              npm), reached through an in-container allowlisting proxy that is the
#              sole process permitted to egress to the public internet
#   BLOCKED  • everything else — the agent's scanners/curl cannot hit real hosts
#
# Fail-closed: if the rules can't be applied we EXIT rather than run unsandboxed.
# Full internet is available only by explicit opt-in: CLANKER_ALLOW_INTERNET=1.
set -euo pipefail

PROXY_PORT=8119
PROXY_USER=clanker-proxy
PROXY_CONF=/etc/clanker/tinyproxy.conf

if [ "${CLANKER_ALLOW_INTERNET:-0}" = "1" ]; then
  echo "[sandbox] CLANKER_ALLOW_INTERNET=1 — egress UNRESTRICTED (agent has full internet)." >&2
  exec "$@"
fi

echo "[sandbox] Applying egress allowlist (agent blocked from the public internet)." >&2

# Host addresses: Ollama and the sibling challenge containers are reached here.
# Docker Desktop may hand back IPv4, IPv6, or both — collect all and route each to
# the matching firewall (feeding an IPv6 addr to IPv4 iptables is a hard error).
HOST_ADDRS="$(getent ahosts host.docker.internal 2>/dev/null | awk '{print $1}' | sort -u)"
if [ -z "$HOST_ADDRS" ]; then
  echo "[sandbox] FATAL: could not resolve host.docker.internal — refusing to run unsandboxed." >&2
  echo "[sandbox]   (need --add-host host.docker.internal:host-gateway). Set CLANKER_ALLOW_INTERNET=1 to bypass." >&2
  exit 3
fi
HOST_V4="$(echo "$HOST_ADDRS" | grep -E '^[0-9]+(\.[0-9]+){3}$' || true)"
HOST_V6="$(echo "$HOST_ADDRS" | grep ':' || true)"
echo "[sandbox] host.docker.internal -> v4:[$(echo $HOST_V4)] v6:[$(echo $HOST_V6)] (allowed: Ollama + challenge ports)" >&2

# ── Start the allowlisting forward proxy ─────────────────────────────────────
mkdir -p /var/log/clanker /run/clanker
chown "$PROXY_USER":"$PROXY_USER" /var/log/clanker /run/clanker
tinyproxy -c "$PROXY_CONF"
# Wait for it to listen (bash /dev/tcp, no extra tools needed).
for _ in $(seq 1 50); do
  if (echo > "/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null; then break; fi
  sleep 0.1
done
if ! (echo > "/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null; then
  echo "[sandbox] FATAL: allowlist proxy did not come up on :$PROXY_PORT" >&2
  exit 4
fi
PROXY_UID="$(id -u "$PROXY_USER")"
echo "[sandbox] allowlist proxy up on 127.0.0.1:$PROXY_PORT (uid=$PROXY_UID)" >&2

# ── Egress firewall ──────────────────────────────────────────────────────────
# Same allowlist applied to both families: loopback, established, DNS, the host
# address(es) for that family, and the proxy uid (the only process allowed to
# open new public connections — it enforces the domain allowlist in userspace and
# so tolerates CDN IP churn). Everything else is dropped.
apply_family() {
  local ipt="$1"; local hosts="$2"
  $ipt -F OUTPUT
  $ipt -A OUTPUT -o lo -j ACCEPT
  $ipt -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
  $ipt -A OUTPUT -p udp --dport 53 -j ACCEPT
  $ipt -A OUTPUT -p tcp --dport 53 -j ACCEPT
  local h
  for h in $hosts; do $ipt -A OUTPUT -d "$h" -j ACCEPT; done
  $ipt -A OUTPUT -m owner --uid-owner "$PROXY_UID" -j ACCEPT
  $ipt -A OUTPUT -j DROP
  $ipt -P OUTPUT ACCEPT   # trailing DROP rule is the deny-all; policy is irrelevant
}

apply_family iptables "$HOST_V4"
if command -v ip6tables >/dev/null 2>&1; then
  apply_family ip6tables "$HOST_V6"
fi

# ── Point proxy-aware clients (LangSmith SDK, npm, curl) at the proxy ─────────
export HTTP_PROXY="http://127.0.0.1:$PROXY_PORT"
export HTTPS_PROXY="http://127.0.0.1:$PROXY_PORT"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="localhost,127.0.0.1,host.docker.internal,$(echo $HOST_V4 $HOST_V6 | tr ' ' ',')"
export no_proxy="$NO_PROXY"
# Best-effort proxy support for the Node-based Brave MCP server.
export GLOBAL_AGENT_HTTP_PROXY="$HTTP_PROXY"
export GLOBAL_AGENT_HTTPS_PROXY="$HTTPS_PROXY"

echo "[sandbox] egress allowlist active. Public internet reachable only via allowlisted domains." >&2
exec "$@"

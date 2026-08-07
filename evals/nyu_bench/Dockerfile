# syntax=docker/dockerfile:1
#
# NYU CTF Bench eval runner. Extends the harness image with the nyuctf dataset
# loader and a Docker client, so run_eval.py can bring each challenge's
# containers up as SIBLINGS on the host daemon via the mounted /var/run/docker.sock
# (never Docker-in-Docker). Built/started under the compose `eval` profile:
#     docker compose --profile eval up --build
FROM filthy-clanker:latest

USER root

# Docker client + compose so run_eval.py can drive each challenge's compose stack
# against the host daemon. run_eval.py auto-detects `docker compose` (v2) or the
# `docker-compose` (v1) fallback.
# tinyproxy + iptables power the egress sandbox (sandbox/entrypoint.sh): the agent
# is blocked from arbitrary public hosts, reaching only the host (Ollama +
# challenge ports) and an allowlist of domains (Brave, LangSmith, npm).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        docker.io docker-compose tinyproxy iptables \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Dedicated unprivileged user: the ONLY process the firewall lets egress to the
# public internet is the allowlisting proxy running as this user.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin clanker-proxy

# Eval-only Python deps (nyuctf dataset loader + Docker SDK), into the same system
# Python the harness uses.
COPY requirements.txt /tmp/eval-requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r /tmp/eval-requirements.txt

# The harness image excludes evals/ (.dockerignore), so add the runner here. The
# rest of the harness (src/, agents.yaml, …) is already present from the base.
COPY . /home/kali/filthy-clanker/evals/nyu_bench/

# Install the egress sandbox to stable paths OUTSIDE the app dir, so a bind-mount
# of the repo over /home/kali/filthy-clanker (dev iteration) does not shadow them.
# Strip any CRLF line endings so the script runs regardless of git autocrlf.
RUN mkdir -p /etc/clanker \
    && sed 's/\r$//' /home/kali/filthy-clanker/evals/nyu_bench/sandbox/entrypoint.sh \
        > /usr/local/bin/clanker-entrypoint.sh \
    && chmod 0755 /usr/local/bin/clanker-entrypoint.sh \
    && sed 's/\r$//' /home/kali/filthy-clanker/evals/nyu_bench/sandbox/tinyproxy.conf \
        > /etc/clanker/tinyproxy.conf \
    && sed 's/\r$//' /home/kali/filthy-clanker/evals/nyu_bench/sandbox/proxy-allowlist.txt \
        > /etc/clanker/proxy-allowlist.txt

# Runs as root (not the harness's `kali` user): the runner needs access to the
# mounted Docker socket to launch challenge containers, and NET_ADMIN to install
# the egress firewall. The entrypoint applies the sandbox, then execs the CMD.
WORKDIR /home/kali/filthy-clanker
ENTRYPOINT ["/usr/local/bin/clanker-entrypoint.sh"]
CMD ["python3", "evals/nyu_bench/run_eval.py"]

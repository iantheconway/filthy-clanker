# syntax=docker/dockerfile:1
#
# filthy-clanker harness — headless Kali tooling + the LangGraph multi-agent
# CTF solver, in one image.
#
# Ollama is deliberately NOT in this image: it runs on the host (with the GPUs)
# and is reached over the network via OLLAMA_HOST (see SPEC-02 / docker-compose).
# Raw-socket capabilities for the security tools are granted at RUN time
# (--cap-add NET_RAW,NET_ADMIN), never baked in as --privileged.
FROM kalilinux/kali-rolling

# --- System packages: Kali offensive toolset (headless) + Python runtime ------
# kali-linux-headless is large; the first build pulls multiple GB and is slow.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        kali-linux-headless \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Unprivileged runtime user ------------------------------------------------
RUN useradd --create-home --shell /bin/bash kali
ENV HEXSTRIKE_DIR=/home/kali/hexstrike-ai \
    HEXSTRIKE_PORT=8888

# --- Hexstrike-AI (MCP security-tool server) ----------------------------------
# Pinned to a commit for reproducible builds. Repo has no release tags; bump
# HEXSTRIKE_REF (a master commit SHA) to update. Repo: 0x4m4/hexstrike-ai.
ARG HEXSTRIKE_REPO=https://github.com/0x4m4/hexstrike-ai
ARG HEXSTRIKE_REF=9b8c780f324ce5145a322bfa23c98886f8424ba3
RUN git clone "$HEXSTRIKE_REPO" "$HEXSTRIKE_DIR" \
    && git -C "$HEXSTRIKE_DIR" checkout --quiet "$HEXSTRIKE_REF"

WORKDIR /home/kali/filthy-clanker

# --- Python dependencies ------------------------------------------------------
# Kali's system Python is externally managed (PEP 668), so --break-system-packages
# is required for a container-global install.
#
# pyhackthebox pins requests==2.27.1, which conflicts with the LLM/MCP stack and
# breaks the whole resolve. Install everything else first, then pyhackthebox with
# --no-deps so the optional HTB commands still work without dragging in the bad
# pin. (Robust whether or not requirements.txt still lists pyhackthebox.)
COPY requirements.txt ./
RUN grep -viE '^[[:space:]]*pyhackthebox([[:space:]<>=!~;#].*)?$' requirements.txt > /tmp/requirements.core.txt \
    && pip install --break-system-packages --no-cache-dir -r /tmp/requirements.core.txt \
    && pip install --break-system-packages --no-cache-dir --no-deps pyhackthebox

# --- Hexstrike environment (its own venv) -------------------------------------
# Hexstrike runs as a SEPARATE MCP server from its own venv — main.py launches
# $HEXSTRIKE_DIR/hexstrike-env/bin/python3 (see MCP_COMMAND in .env.example). Its
# deps (angr, pwntools, mitmproxy, selenium, …) include C extensions, so a build
# toolchain is required. This is the heavy part of the image (adds ~1–2 GB).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv "$HEXSTRIKE_DIR/hexstrike-env" \
    && "$HEXSTRIKE_DIR/hexstrike-env/bin/pip" install --no-cache-dir --upgrade pip \
    && if [ -f "$HEXSTRIKE_DIR/requirements.txt" ]; then \
         "$HEXSTRIKE_DIR/hexstrike-env/bin/pip" install --no-cache-dir -r "$HEXSTRIKE_DIR/requirements.txt"; \
       fi

# --- Harness source -----------------------------------------------------------
COPY . .
RUN chown -R kali:kali /home/kali

USER kali
EXPOSE 8888

# Default: launch the harness. SPEC-01 acceptance overrides this (e.g. `nmap
# --version`, `id`) to prove the tooling and unprivileged user without starting
# the interactive agent loop.
CMD ["python3", "src/main.py"]

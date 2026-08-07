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
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io docker-compose \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Eval-only Python deps (nyuctf dataset loader + Docker SDK), into the same system
# Python the harness uses.
COPY requirements.txt /tmp/eval-requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r /tmp/eval-requirements.txt

# The harness image excludes evals/ (.dockerignore), so add the runner here. The
# rest of the harness (src/, agents.yaml, …) is already present from the base.
COPY . /home/kali/filthy-clanker/evals/nyu_bench/

# Runs as root (not the harness's `kali` user): the runner needs access to the
# mounted Docker socket to launch challenge containers.
WORKDIR /home/kali/filthy-clanker
CMD ["python3", "evals/nyu_bench/run_eval.py"]

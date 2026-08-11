"""
Deterministic pre-solve file sweep.

The universal CTF first move: when a challenge ships files, unpack every archive
and grep the flag-format regex across all of them BEFORE any LLM routing. Many
web / misc / crypto challenges hide the literal flag in a source file, a config,
a Dockerfile, or a challenge.json; relying on the LLM supervisor to route to the
right agent and then `cat` the right file is fragile — it silently regressed real
solves (historypeats, MFW) the moment routing changed. This makes the harness
look at every provided file deterministically, so routing quality stops being
load-bearing for a flag that is literally sitting on disk.

Design guards (all general — no per-challenge logic):
  • Only TEXT / source / config files yield an auto-usable flag. A flag-format
    string inside a BINARY is decoy-prone (`strings` routinely shows a fake
    `flag{...}`) and is returned as a *lead* for the reversing agent, never as an
    auto-end flag.
  • Candidates pass the same placeholder + format filters as the live extractor,
    plus a printable-ASCII check, so template strings (`flag{<value>}`) and binary
    garbage (`Si{\\nB+=i…}` out of an xor blob) never count.
  • Archives extract to a throwaway temp dir with the tar `data` filter (no path
    traversal, no mutation of the challenge dir).

This is a real capability the product should have, not an eval trick — a human
CTF player runs `grep -rE 'flag\\{' .` on the handout before anything else.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

from .summarizer import (
    _FLAG_RE,
    _flag_matches_format,
    _is_placeholder_flag,
    _is_printable_flag,
)

logger = logging.getLogger(__name__)

# Any word{...} bracket, to catch prefixes _FLAG_RE's known-prefix list misses.
_BROAD_BRACKET_RE = re.compile(r'\b\w{2,20}\{[^}]{1,200}\}')

# Archive suffixes we unpack (longest first so `.tar.gz` beats `.gz`).
_TAR_SUFFIXES = ('.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar')


def _decode_text(data: bytes) -> str | None:
    """Return decoded text if ``data`` looks like a text/source/config file, else None.

    Binary detection is deliberately cheap and conservative: a NUL byte in the head,
    a UTF-8 decode failure, or a low printable ratio all mark the file binary so its
    flag-format hits become *leads*, not auto-end flags.
    """
    if b'\x00' in data[:8192]:
        return None
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return None
    sample = text[:8192]
    if not sample:
        return text
    printable = sum(1 for c in sample if c.isprintable() or c in '\r\n\t')
    return text if printable / len(sample) > 0.85 else None


def _safe_extract_tar(path: Path, dest: Path) -> None:
    with tarfile.open(path) as tf:
        # filter='data' (Py 3.12+) strips absolute paths, `..`, and special files.
        tf.extractall(dest, filter='data')


def _safe_extract_zip(path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest)):
                logger.debug("file_sweep: skipping zip member outside dest: %s", member)
                continue
            zf.extract(member, dest)


def _collect_files(paths: list[str], workdir: Path) -> list[Path]:
    """Yield every file to scan: the provided files plus everything unpacked from
    any archives among them (into ``workdir``)."""
    scan: list[Path] = []
    for p in paths:
        pth = Path(p)
        if not pth.exists() or not pth.is_file():
            continue
        name = pth.name.lower()
        try:
            if name.endswith(_TAR_SUFFIXES):
                _safe_extract_tar(pth, workdir)
            elif name.endswith('.zip'):
                _safe_extract_zip(pth, workdir)
            elif name.endswith('.gz'):
                out = workdir / (pth.stem or 'ungz')
                with gzip.open(pth) as gz, open(out, 'wb') as o:
                    shutil.copyfileobj(gz, o)
            else:
                scan.append(pth)
        except Exception as exc:  # a broken archive is still worth grepping raw
            logger.debug("file_sweep: could not unpack %s: %s", p, exc)
            scan.append(pth)
    for root, _dirs, fnames in os.walk(workdir):
        for fn in fnames:
            scan.append(Path(root) / fn)
    return scan


def sweep_files(paths: list[str], flag_format: str = "") -> dict:
    """Grep provided files (and their archives) for flag-format strings.

    Returns ``{"flags": [...], "leads": [...]}``:
      • ``flags`` — high-confidence flags found in TEXT/source/config files. Safe to
        seed into the KB so the run ends immediately (the exact path that solved
        historypeats/MFW before the routing regression).
      • ``leads`` — flag-format candidates from binaries, or text hits whose wrapper
        prefix doesn't match the stated format. Decoy-prone: surface for the agent,
        do not auto-end.
    """
    if not paths:
        return {"flags": [], "leads": []}

    text_flags: list[str] = []
    leads: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="clanker_sweep_"))
    try:
        for f in _collect_files(paths, workdir):
            try:
                data = f.read_bytes()
            except Exception:
                continue
            text = _decode_text(data)
            haystack = text if text is not None else data.decode('latin-1', 'ignore')
            candidates = _FLAG_RE.findall(haystack) + _BROAD_BRACKET_RE.findall(haystack)
            for c in candidates:
                if _is_placeholder_flag(c) or not _is_printable_flag(c):
                    continue
                if text is None:
                    leads.append(c)                       # binary → decoy-prone lead
                elif flag_format and not _flag_matches_format(c, flag_format):
                    leads.append(c)                       # wrong wrapper → lead
                else:
                    text_flags.append(c)                  # text + right shape → flag
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    text_flags = list(dict.fromkeys(text_flags))
    leads = list(dict.fromkeys(x for x in leads if x not in text_flags))
    return {"flags": text_flags, "leads": leads}

"""Tests for config.py"""
import pytest

from config import build_system_prompt


def test_build_system_prompt_no_args():
    prompt = build_system_prompt()
    assert "hackthebox" in prompt.lower()
    assert "/spawn" in prompt


def test_build_system_prompt_with_machine():
    prompt = build_system_prompt(
        machine_name="Lame",
        machine_os="Linux",
        machine_difficulty="Easy",
        machine_ip="10.10.10.3",
    )
    assert "Lame" in prompt
    assert "lame.htb" in prompt
    assert "10.10.10.3" in prompt
    assert "easy" in prompt.lower()
    assert "linux" in prompt.lower()


def test_build_system_prompt_hostname_lowercase():
    prompt = build_system_prompt(machine_name="TwoMillion")
    assert "twomillion.htb" in prompt


def test_build_system_prompt_os_only():
    prompt = build_system_prompt(machine_name="Box", machine_os="Windows")
    assert "windows" in prompt.lower()
    assert "Box" in prompt


def test_build_system_prompt_difficulty_only():
    prompt = build_system_prompt(machine_name="Box", machine_difficulty="Hard")
    assert "hard" in prompt.lower()


def test_build_system_prompt_no_ip_when_not_provided():
    prompt = build_system_prompt(machine_name="Box")
    # IP section should only appear if machine_ip was passed
    assert "The target IP" not in prompt


def test_build_system_prompt_with_ip():
    prompt = build_system_prompt(machine_name="Box", machine_ip="10.10.11.1")
    assert "10.10.11.1" in prompt


def test_default_system_prompt_importable():
    from config import DEFAULT_SYSTEM_PROMPT
    assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
    assert len(DEFAULT_SYSTEM_PROMPT) > 0

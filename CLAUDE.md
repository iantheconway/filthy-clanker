# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Filthy-Clanker is an AI-powered CTF solver that connects LLMs (Claude or Gemini) to 130+ security tools via the Model Context Protocol (MCP). It communicates with the [Hexstrike-AI](https://github.com/your-org/hexstrike-ai) tool server to run reconnaissance, enumeration, and exploitation tools autonomously.

## Running

```bash
source venv/bin/activate
python src/main.py
```

Requires a `.env` file (copy from `.env.example`) with at least one API key (`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`). The Hexstrike-AI repo must be cloned to `/home/kali/hexstrike-ai` (or set `HEXSTRIKE_DIR`).

On startup: starts/detects the Hexstrike Flask server, prompts for LLM provider, connects to MCP, enters interactive chat loop.

## Dependencies

```bash
pip install anthropic google-genai mcp python-dotenv requests
```

Python 3.13+ required. No test suite or linter configured.

## Architecture

```
User → main.py (chat loop) → LLM client → MCP client → Hexstrike MCP server → Hexstrike Flask server → security tools
```

**Key design points:**

- **LLM abstraction**: `src/llms/base.py` defines `BaseLLMClient` ABC. Two implementations: `AnthropicClient` (Claude) and `GeminiClient`. Each handles provider-specific tool formatting and message serialization. The chat loop in `main.py` branches on provider type for message construction since Anthropic and Gemini have different tool result message formats.

- **MCP client** (`src/mcp_client/client.py`): `HexstrikeMCPClient` wraps the `mcp` SDK's stdio transport. Connects to the Hexstrike MCP bridge process, caches tool list, and serializes tool results to text.

- **Tool call loop** (`main.py:chat_loop`): After each LLM response, tool calls are executed and results fed back in a loop until the model stops requesting tools. Anthropic bundles all tool results in a single user message; Gemini sends each as a separate message.

- **Session persistence** (`src/session.py`): Sessions auto-save as JSON on exit. `/save` also generates an LLM-written markdown summary. `/resume` can restore from JSON (same provider only) or fall back to summary-based context injection for cross-provider resume.

- **System prompt** (`src/config.py`): Contains the current challenge-specific prompt (target, constraints, strategy). Update this per challenge.

## In-Chat Commands

`/save [name]`, `/resume [name]`, `/sessions`, `exit`/`quit`

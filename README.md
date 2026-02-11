# Filthy-Clanker

An AI-powered CTF (Capture The Flag) challenge solver that uses LLMs and 130+ security tools to assist with HackTheBox challenges. It connects large language models (Claude or Gemini) to the [Hexstrike-AI](https://github.com/your-org/hexstrike-ai) tool server via the Model Context Protocol (MCP), enabling the AI to autonomously run reconnaissance, enumeration, and exploitation tools based on conversational input.

## Architecture

```
User Input
    |
Filthy-Clanker (this repo)
    |
LLM (Claude / Gemini) -- decides which tools to call
    |
MCP Client -- communicates over stdio
    |
Hexstrike MCP Server -- translates to HTTP
    |
Hexstrike Flask Server -- executes tools
    |
Security Tools (nmap, gobuster, etc.)
```

## Prerequisites

- Python 3.13+
- [Hexstrike-AI](https://github.com/your-org/hexstrike-ai) cloned to `/home/kali/hexstrike-ai` (or set `HEXSTRIKE_DIR`)
- An API key for at least one LLM provider:
  - [Anthropic](https://console.anthropic.com/) (Claude)
  - [Google3 AI Studio](https://aistudio.google.com/) (Gemini)

## Installation

```bash
git clone <repo-url> filthy-clanker
cd filthy-clanker

python3 -m venv venv
source venv/bin/activate

pip install anthropic google-genai mcp python-dotenv requests
```

## Configuration

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

`.env` contents:

```env
# At least one API key is required
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key

# Hexstrike configuration (defaults shown)
HEXSTRIKE_DIR=/home/kali/hexstrike-ai
HEXSTRIKE_PORT=8888

# Optional MCP overrides (derived from the above by default)
# MCP_COMMAND=/home/kali/hexstrike-ai/hexstrike-env/bin/python3
# MCP_ARGS=/home/kali/hexstrike-ai/hexstrike_mcp.py --server http://127.0.0.1:8888
```

## Usage

```bash
source venv/bin/activate
python src/main.py
```

On startup the program will:

1. Start the Hexstrike Flask server (or detect it if already running)
2. Prompt you to select an LLM provider (Anthropic or Gemini)
3. Connect to the Hexstrike MCP server and list available tools
4. Enter an interactive chat loop

Example session:

```
Select LLM provider:
  1) Anthropic (Claude)
  2) Gemini
Enter 1 or 2: 2

[*] Hexstrike server is ready.
[*] Connecting to MCP server: /home/kali/hexstrike-ai/hexstrike-env/bin/python3 ...
[*] MCP session initialized.
[*] 134 MCP tools available:
    - nmap_scan: Run an nmap scan against a target
    ...

Chat started. Type 'exit' or 'quit' to stop.

You: run a quick nmap scan on 10.10.11.50
[tool] Calling nmap_scan({"target": "10.10.11.50", "arguments": "-sC -sV"})
[tool] nmap_scan returned (2340 chars)
Assistant: Based on the scan results, I can see ports 22 (SSH) and 80 (HTTP) are open...
```

Type `exit` or `quit` to stop. On exit, the Hexstrike server is automatically shut down.

## Project Structure

```
src/
├── main.py              # Entry point, server management, chat loop
├── config.py            # System prompt configuration
├── llms/
│   ├── base.py          # Abstract LLM client interface
│   ├── anthropic_client.py  # Claude integration
│   └── gemini_client.py     # Gemini integration
└── mcp_client/
    └── client.py        # MCP protocol client
```

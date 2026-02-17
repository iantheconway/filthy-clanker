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

pip install anthropic google-genai mcp python-dotenv requests pyhackthebox
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

# HackTheBox integration (optional — enables /spawn, /flag, etc.)
HTB_APP_TOKEN=your-htb-app-token

# Hexstrike configuration (defaults shown)
HEXSTRIKE_DIR=/home/kali/hexstrike-ai
HEXSTRIKE_PORT=8888

# Optional MCP overrides (derived from the above by default)
# MCP_COMMAND=/home/kali/hexstrike-ai/hexstrike-env/bin/python3
# MCP_ARGS=/home/kali/hexstrike-ai/hexstrike_mcp.py --server http://127.0.0.1:8888
```

To get an HTB app token, go to your [HackTheBox profile settings](https://app.hackthebox.com/profile/settings) and create one under "App Tokens".

## Usage

```bash
source venv/bin/activate
python src/main.py
```

On startup the program will:

1. Connect to HackTheBox (if `HTB_APP_TOKEN` is set) and detect any active machine
2. Start the Hexstrike Flask server (or detect it if already running)
3. Prompt you to select an LLM provider (Anthropic or Gemini)
4. Connect to the Hexstrike MCP server and list available tools
5. Build a system prompt tailored to the active machine (or a generic one if none is running)
6. Enter an interactive chat loop

Example session:

```
[*] HTB client connected.
[*] Active machine: Headless (Linux, Easy) @ 10.10.11.8
[*] Hexstrike server is ready.

Select LLM provider:
  1) Anthropic (Claude)
  2) Gemini
Enter 1 or 2: 1

[*] MCP session initialized.
[*] 134 MCP tools available:
    - nmap_scan: Run an nmap scan against a target
    ...

Chat started. Type 'exit' or 'quit' to stop.
Commands: /save [name], /resume [name], /sessions
HTB:      /machine [name], /spawn <name>, /stop, /reset, /flag <flag>, /vpn

You: let's start with a quick scan
[tool] Calling nmap_scan({"target": "10.10.11.8", "arguments": "-sC -sV"})
[tool] nmap_scan returned (2340 chars)
Assistant: Based on the scan results, I can see ports 22 (SSH) and 5000 (HTTP) are open...
```

Type `exit` or `quit` to stop. On exit, the Hexstrike server is automatically shut down.

## HackTheBox Integration

When `HTB_APP_TOKEN` is set, you get direct access to HTB machine management from within the chat. The system prompt automatically adapts to the active machine, so the AI knows the target name, OS, difficulty, and IP without you having to type it.

### HTB Commands

| Command | Description |
|---------|-------------|
| `/machine` | Show the currently active machine |
| `/machine <name>` | Look up details for any machine |
| `/spawn <name>` | Spawn a machine and update the AI's context |
| `/stop` | Stop the active machine |
| `/reset` | Reset the active machine (~1 min) |
| `/flag <flag>` | Submit a flag for the active machine |
| `/vpn` | Show current VPN server info |

### Example: Full workflow

```
You: /spawn Headless
[*] Spawning 'Headless'...
[*] Machine spawned at 10.10.11.8

You: let's go
Assistant: Understood — now targeting Headless at 10.10.11.8. I'll start with a quick
nmap scan to see what's open...

    ... hacking happens ...

You: /flag 8a3f5b2c1d4e6f7a8b9c0d1e2f3a4b5c
[*] Congratulations! Machine owned!

You: /stop
[*] Stopped machine 'Headless'.
```

### How it works

- On startup, the client checks for an already-running machine and auto-populates the system prompt.
- `/spawn` updates the system prompt and injects a context message into the conversation so the AI immediately knows the new target.
- `/stop` resets the prompt back to a generic one.
- Without `HTB_APP_TOKEN`, everything works as before — you just won't have the HTB commands, and the system prompt will be generic.

## Project Structure

```
src/
├── main.py              # Entry point, server management, chat loop
├── config.py            # Dynamic system prompt builder
├── htb_client.py        # HackTheBox API wrapper
├── session.py           # Session save/resume logic
├── llms/
│   ├── base.py          # Abstract LLM client interface
│   ├── anthropic_client.py  # Claude integration
│   └── gemini_client.py     # Gemini integration
└── mcp_client/
    └── client.py        # MCP protocol client
```

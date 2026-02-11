import asyncio
import os
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

from config import DEFAULT_SYSTEM_PROMPT
from llms import AnthropicClient, GeminiClient
from mcp_client import HexstrikeMCPClient

HEXSTRIKE_DIR = os.getenv("HEXSTRIKE_DIR", "/home/kali/hexstrike-ai")
HEXSTRIKE_PORT = os.getenv("HEXSTRIKE_PORT", "8888")
HEXSTRIKE_VENV_PYTHON = os.path.join(HEXSTRIKE_DIR, "hexstrike-env", "bin", "python3")
HEXSTRIKE_SERVER_SCRIPT = os.path.join(HEXSTRIKE_DIR, "hexstrike_server.py")
HEXSTRIKE_MCP_SCRIPT = os.path.join(HEXSTRIKE_DIR, "hexstrike_mcp.py")


def start_hexstrike_server() -> subprocess.Popen | None:
    """Start hexstrike_server.py as a background process and wait for health.

    Returns the subprocess handle, or None if the server was already running.
    """
    # Use a lightweight endpoint for readiness checks — /health runs
    # 'which' on 130+ tools and can take 30+ seconds to respond.
    url = f"http://127.0.0.1:{HEXSTRIKE_PORT}/api/cache/stats"

    # Check if already running
    try:
        if requests.get(url, timeout=3).ok:
            print("[*] Hexstrike server already running.")
            return None
    except Exception:
        pass

    print(f"[*] Starting hexstrike server on port {HEXSTRIKE_PORT}...")
    server_log = open(os.path.join(HEXSTRIKE_DIR, "server.log"), "w")
    proc = subprocess.Popen(
        [HEXSTRIKE_VENV_PYTHON, HEXSTRIKE_SERVER_SCRIPT, "--port", HEXSTRIKE_PORT],
        cwd=HEXSTRIKE_DIR,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    # Poll /health up to ~30 seconds
    for i in range(30):
        # Check if process died
        if proc.poll() is not None:
            server_log.close()
            log_path = os.path.join(HEXSTRIKE_DIR, "server.log")
            print(f"[!] Hexstrike server exited with code {proc.returncode}.")
            print(f"[!] Check {log_path} for details.")
            sys.exit(1)
        try:
            if requests.get(url, timeout=3).ok:
                print("[*] Hexstrike server is ready.")
                return proc
        except Exception:
            pass
        time.sleep(1)

    proc.kill()
    server_log.close()
    sys.exit("Error: Hexstrike server failed to start within 30 seconds.")


def select_provider() -> str:
    print("\nSelect LLM provider:")
    print("  1) Anthropic (Claude)")
    print("  2) Gemini")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return "anthropic" if choice == "1" else "gemini"
        print("Invalid choice.")


def build_llm_client(provider: str):
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("Error: ANTHROPIC_API_KEY not set in environment.")
        return AnthropicClient(api_key=api_key)
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.exit("Error: GEMINI_API_KEY not set in environment.")
        return GeminiClient(api_key=api_key)


async def chat_loop(llm_client, mcp_client: HexstrikeMCPClient):
    tools = await mcp_client.list_tools()
    print(f"\n[*] {len(tools)} MCP tools available:")
    for t in tools:
        print(f"    - {t['name']}: {t['description'][:80]}")

    messages: list[dict] = []
    provider = "anthropic" if isinstance(llm_client, AnthropicClient) else "gemini"

    print("\nChat started. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Single turn: get one LLM response, handle at most one round of tool calls,
        # then pause for user input.
        response = await llm_client.generate_response(
            messages, tools, DEFAULT_SYSTEM_PROMPT
        )

        # Print any text the model produced
        if response["text"]:
            print(f"\nAssistant: {response['text']}\n")

        tool_calls = llm_client.parse_tool_calls(response)

        if not tool_calls:
            # Pure text reply — append and loop back to user
            if provider == "anthropic":
                messages.append(AnthropicClient.make_assistant_message(response))
            else:
                messages.append(GeminiClient.make_assistant_message(response))
            continue

        # Append the assistant message that includes the tool_use blocks
        if provider == "anthropic":
            messages.append(AnthropicClient.make_assistant_message(response))
        else:
            messages.append(GeminiClient.make_assistant_message(response))

        # Execute each tool call and collect results
        tool_result_parts = []
        for tc in tool_calls:
            print(f"[tool] Calling {tc['name']}({tc['arguments']})")
            try:
                result = await mcp_client.call_tool(tc["name"], tc["arguments"])
            except Exception as e:
                result = f"Error executing tool: {e}"
            print(f"[tool] {tc['name']} returned ({len(result)} chars)")

            if provider == "anthropic":
                tool_result_parts.append(
                    AnthropicClient.make_tool_result_message(tc["id"], result)
                )
            else:
                messages.append(
                    GeminiClient.make_tool_result_message(tc["name"], result)
                )

        # For Anthropic, tool results go in a single user message
        if provider == "anthropic" and tool_result_parts:
            messages.append({"role": "user", "content": tool_result_parts})

        # Get the model's follow-up after seeing the tool results
        followup = await llm_client.generate_response(
            messages, tools, DEFAULT_SYSTEM_PROMPT
        )

        if followup["text"]:
            print(f"\nAssistant: {followup['text']}\n")

        if provider == "anthropic":
            messages.append(AnthropicClient.make_assistant_message(followup))
        else:
            messages.append(GeminiClient.make_assistant_message(followup))

        # Pause here — do NOT auto-loop further tool calls.
        # The user must provide the next input.


async def main():
    load_dotenv()

    # Start (or detect) the hexstrike Flask server
    server_proc = start_hexstrike_server()

    provider = select_provider()
    llm_client = build_llm_client(provider)

    # Configure MCP server command — defaults use hexstrike venv python
    mcp_command = os.getenv("MCP_COMMAND", HEXSTRIKE_VENV_PYTHON)
    mcp_args_str = os.getenv(
        "MCP_ARGS",
        f"{HEXSTRIKE_MCP_SCRIPT} --server http://127.0.0.1:{HEXSTRIKE_PORT}",
    )
    args = mcp_args_str.split() if mcp_args_str else []

    mcp_client = HexstrikeMCPClient(command=mcp_command, args=args)

    print(f"[*] Connecting to MCP server: {mcp_command} {' '.join(args)}")
    try:
        await mcp_client.connect()
    except Exception as e:
        if server_proc:
            server_proc.terminate()
        sys.exit(f"Failed to connect to MCP server: {e}")

    print("[*] MCP session initialized.")

    try:
        await chat_loop(llm_client, mcp_client)
    finally:
        await mcp_client.disconnect()
        print("[*] MCP session closed.")
        if server_proc:
            server_proc.terminate()
            server_proc.wait()
            print("[*] Hexstrike server stopped.")


if __name__ == "__main__":
    asyncio.run(main())

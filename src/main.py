import asyncio
import os
import select
import signal
import subprocess
import sys
import time

import anthropic
import requests
from dotenv import load_dotenv

from config import build_system_prompt
from htb_client import HTB
from llms import AnthropicClient, GeminiClient
from mcp_client import HexstrikeMCPClient
from session import (
    SESSIONS_DIR,
    default_session_name,
    list_sessions,
    load_session_json,
    load_summary,
    save_session_json,
    save_summary,
)

HEXSTRIKE_DIR = os.getenv("HEXSTRIKE_DIR", "/home/kali/hexstrike-ai")
HEXSTRIKE_PORT = os.getenv("HEXSTRIKE_PORT", "8888")
HEXSTRIKE_VENV_PYTHON = os.path.join(HEXSTRIKE_DIR, "hexstrike-env", "bin", "python3")
HEXSTRIKE_SERVER_SCRIPT = os.path.join(HEXSTRIKE_DIR, "hexstrike_server.py")
HEXSTRIKE_MCP_SCRIPT = os.path.join(HEXSTRIKE_DIR, "hexstrike_mcp.py")


_sigint_received = False
_active_task: asyncio.Task | None = None


def _sigint_handler(signum, frame):
    global _sigint_received
    _sigint_received = True
    # Cancel the currently awaited async task so control returns immediately
    if _active_task and not _active_task.done():
        _active_task.cancel()
    print("\n[*] Ctrl+C received — interrupting...")


def _check_interrupt() -> bool:
    """Return True and reset the flag if SIGINT was received."""
    global _sigint_received
    if _sigint_received:
        _sigint_received = False
        return True
    return False


async def _cancellable(coro):
    """Run a coroutine in a task that can be cancelled by the SIGINT handler."""
    global _active_task
    task = asyncio.ensure_future(coro)
    _active_task = task
    try:
        return await task
    finally:
        _active_task = None


def _flush_stdin():
    """Drain any buffered input from stdin (e.g. leftover newlines from paste)."""
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
    except Exception:
        pass


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


def _auto_save(messages: list[dict], provider: str):
    """Auto-save session JSON on exit (no LLM summary)."""
    if not messages:
        return
    name = default_session_name()
    filepath = os.path.join(SESSIONS_DIR, f"{name}.json")
    save_session_json(messages, provider, filepath)
    print(f"[*] Session auto-saved to {filepath}")


async def _handle_command(
    cmd: str,
    messages: list[dict],
    provider: str,
    llm_client,
    tools: list[dict],
    system_prompt: str,
    htb_client: HTB | None = None,
) -> str | None:
    """Handle slash commands. Returns updated system_prompt if it changed, else None."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # --- HTB commands ---
    if command == "/machine":
        if not htb_client:
            print("[!] HTB integration not configured (set HTB_APP_TOKEN).")
            return None
        if arg:
            info = htb_client.get_machine_info(arg)
            if isinstance(info, str):
                print(f"[!] {info}")
            else:
                print(f"[*] Machine: {info['name']}")
                print(f"    OS: {info['os']}")
                print(f"    Difficulty: {info['difficulty']}")
                print(f"    Points: {info['points']}")
                print(f"    Stars: {info['stars']}")
                print(f"    Active: {info['active']}  Retired: {info['retired']}")
                print(f"    User owned: {info['user_owned']}  Root owned: {info['root_owned']}")
        else:
            info = htb_client.get_active_machine()
            if info:
                print(f"[*] Active machine: {info['name']}")
                print(f"    OS: {info['os']}")
                print(f"    Difficulty: {info['difficulty']}")
                print(f"    IP: {info['ip']}")
            else:
                print("[*] No active machine.")
        return None

    elif command == "/spawn":
        if not htb_client:
            print("[!] HTB integration not configured (set HTB_APP_TOKEN).")
            return None
        if not arg:
            print("[!] Usage: /spawn <machine-name>")
            return None
        print(f"[*] Spawning '{arg}'...")
        result = htb_client.spawn_machine(arg)
        if result.startswith("Error"):
            print(f"[!] {result}")
            return None
        print(f"[*] Machine spawned at {result}")
        # Refresh machine info and rebuild prompt
        info = htb_client.get_active_machine()
        if info:
            new_prompt = build_system_prompt(
                machine_name=info["name"],
                machine_os=info["os"],
                machine_difficulty=info["difficulty"],
                machine_ip=info["ip"],
            )
            messages.append({
                "role": "user",
                "content": (
                    f"[System: Target machine changed. Now working on '{info['name']}' "
                    f"({info['os']}, {info['difficulty']}) at {info['ip']}. "
                    f"Hostname: {info['name'].lower()}.htb]"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": (
                    f"Understood — now targeting {info['name']} at {info['ip']}. "
                    "Ready to begin when you are."
                ),
            })
            return new_prompt
        return None

    elif command == "/stop":
        if not htb_client:
            print("[!] HTB integration not configured (set HTB_APP_TOKEN).")
            return None
        result = htb_client.stop_machine()
        print(f"[*] {result}")
        if not result.startswith("Error") and not result.startswith("No active"):
            messages.append({
                "role": "user",
                "content": "[System: The active machine has been stopped.]",
            })
            messages.append({
                "role": "assistant",
                "content": "Machine stopped. Use /spawn <name> to start a new one.",
            })
            return build_system_prompt()
        return None

    elif command == "/reset":
        if not htb_client:
            print("[!] HTB integration not configured (set HTB_APP_TOKEN).")
            return None
        result = htb_client.reset_machine()
        print(f"[*] {result}")
        return None

    elif command == "/flag":
        if not htb_client:
            print("[!] HTB integration not configured (set HTB_APP_TOKEN).")
            return None
        if not arg:
            print("[!] Usage: /flag <flag-string>")
            return None
        result = htb_client.submit_flag(arg)
        print(f"[*] {result}")
        return None

    elif command == "/vpn":
        if not htb_client:
            print("[!] HTB integration not configured (set HTB_APP_TOKEN).")
            return None
        result = htb_client.get_vpn_status()
        print(f"[*] {result}")
        return None

    elif command == "/save":
        name = arg or default_session_name()
        json_path = os.path.join(SESSIONS_DIR, f"{name}.json")
        md_path = os.path.join(SESSIONS_DIR, f"{name}.md")
        save_session_json(messages, provider, json_path)
        print(f"[*] Session saved to {json_path}")
        print("[*] Generating summary...")
        await save_summary(messages, llm_client, tools, system_prompt, md_path)
        print(f"[*] Summary saved to {md_path}")

    elif command == "/resume":
        if not arg:
            # List available sessions
            sessions = list_sessions()
            if not sessions:
                print("[*] No saved sessions found.")
                return
            print("[*] Available sessions:")
            for s in sessions:
                flags = []
                if s["has_json"]:
                    flags.append("json")
                if s["has_md"]:
                    flags.append("summary")
                print(f"    - {s['name']} [{', '.join(flags)}]")
            print("\nUsage: /resume <name>")
            return

        # Strip file extensions if user included them
        for ext in (".json", ".md"):
            if arg.endswith(ext):
                arg = arg[: -len(ext)]
                break

        json_path = os.path.join(SESSIONS_DIR, f"{arg}.json")
        md_path = os.path.join(SESSIONS_DIR, f"{arg}.md")

        if os.path.exists(json_path):
            loaded_messages, loaded_provider = load_session_json(json_path)
            if loaded_provider != provider:
                print(
                    f"[!] Session was saved with provider '{loaded_provider}', "
                    f"but current provider is '{provider}'."
                )
                print("[!] Message formats are incompatible for JSON resume.")
                if os.path.exists(md_path):
                    summary_text = load_summary(md_path)
                    context_msg = (
                        f"[Previous session summary for context]\n\n{summary_text}\n\n"
                        "[End of previous session summary. Continue from where we left off.]"
                    )
                    messages.clear()
                    messages.append({"role": "user", "content": context_msg})
                    messages.append({
                        "role": "assistant",
                        "content": "I've reviewed the previous session summary and I'm ready to continue. What would you like to focus on next?",
                    })
                    print(f"[*] Falling back to summary resume for '{arg}'.")
                    print(f"\n--- Session Summary ---\n{summary_text}\n--- End Summary ---\n")
                else:
                    print("[!] No summary file available either. Cannot resume this session.")
                return
            messages.clear()
            messages.extend(loaded_messages)
            # Drop trailing user messages (e.g. interrupt markers) so the
            # next user input doesn't create consecutive user messages.
            while messages and messages[-1].get("role") == "user":
                messages.pop()
            print(f"[*] Resumed session '{arg}' with {len(messages)} messages (JSON).")
        elif os.path.exists(md_path):
            summary_text = load_summary(md_path)
            context_msg = (
                f"[Previous session summary for context]\n\n{summary_text}\n\n"
                "[End of previous session summary. Continue from where we left off.]"
            )
            messages.clear()
            messages.append({"role": "user", "content": context_msg})
            messages.append({
                "role": "assistant",
                "content": "I've reviewed the previous session summary and I'm ready to continue. What would you like to focus on next?",
            })
            print(f"[*] Resumed session '{arg}' from summary (no JSON found).")
            print(f"\n--- Session Summary ---\n{summary_text}\n--- End Summary ---\n")
        else:
            print(f"[!] No session found with name '{arg}'.")

    elif command == "/sessions":
        sessions = list_sessions()
        if not sessions:
            print("[*] No saved sessions found.")
            return
        print("[*] Saved sessions:")
        for s in sessions:
            flags = []
            if s["has_json"]:
                flags.append("json")
            if s["has_md"]:
                flags.append("summary")
            print(f"    - {s['name']} [{', '.join(flags)}]")

    else:
        print(f"[!] Unknown command: {command}")
        return None


async def chat_loop(
    llm_client,
    mcp_client: HexstrikeMCPClient,
    initial_messages: list[dict] | None = None,
    system_prompt: str | None = None,
    htb_client: HTB | None = None,
):
    if system_prompt is None:
        system_prompt = build_system_prompt()

    tools = await mcp_client.list_tools()
    print(f"\n[*] {len(tools)} MCP tools available:")
    for t in tools:
        print(f"    - {t['name']}: {t['description'][:80]}")

    messages: list[dict] = initial_messages if initial_messages else []
    provider = "anthropic" if isinstance(llm_client, AnthropicClient) else "gemini"

    if initial_messages:
        print(f"\n[*] Resumed session with {len(messages)} messages.")

    # Install SIGINT handler so Ctrl+C works reliably in async context
    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    print("\nChat started. Type 'exit' or 'quit' to stop.")
    print("Commands: /save [name], /resume [name], /sessions")
    if htb_client:
        print("HTB:      /machine [name], /spawn <name>, /stop, /reset, /flag <flag>, /vpn")
    print()

    try:
        while True:
            _check_interrupt()  # clear any stale flag
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                _auto_save(messages, provider)
                break

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                _auto_save(messages, provider)
                break
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                new_prompt = await _handle_command(
                    user_input, messages, provider, llm_client, tools,
                    system_prompt, htb_client,
                )
                if new_prompt is not None:
                    system_prompt = new_prompt
                _flush_stdin()
                continue

            messages.append({"role": "user", "content": user_input})

            # Single turn: get one LLM response, handle tool calls, then
            # pause for user input.
            try:
                response = await _cancellable(
                    llm_client.generate_response(
                        messages, tools, system_prompt
                    )
                )
            except asyncio.CancelledError:
                messages.pop()
                print("[*] Interrupted. You can now type a message.")
                _flush_stdin()
                continue
            except anthropic.BadRequestError as e:
                print(f"\n[!] API error: {e.message}")
                messages.pop()
                continue
            except Exception as e:
                print(f"\n[!] LLM request failed: {e}")
                messages.pop()
                continue

            if _check_interrupt():
                messages.pop()
                print("[*] Interrupted. You can now type a message.")
                _flush_stdin()
                continue

            # Print any text the model produced
            if response["text"]:
                print(f"\nAssistant: {response['text']}\n")

            tool_calls = llm_client.parse_tool_calls(response)

            if not tool_calls:
                if provider == "anthropic":
                    messages.append(AnthropicClient.make_assistant_message(response))
                else:
                    messages.append(GeminiClient.make_assistant_message(response))
                continue

            # Process tool calls in a loop.  Track which calls have been
            # completed so we can stub the rest if the user interrupts.
            follow_up_ok = True
            pending_tool_calls: list[dict] = []

            while tool_calls:
                if _check_interrupt():
                    break

                if provider == "anthropic":
                    messages.append(AnthropicClient.make_assistant_message(response))
                else:
                    messages.append(GeminiClient.make_assistant_message(response))

                pending_tool_calls = list(tool_calls)
                tool_result_parts = []
                interrupted_during_tools = False
                for tc in tool_calls:
                    if _check_interrupt():
                        interrupted_during_tools = True
                        break
                    print(f"[tool] Calling {tc['name']}({tc['arguments']})")
                    try:
                        result = await _cancellable(
                            mcp_client.call_tool(tc["name"], tc["arguments"])
                        )
                    except asyncio.CancelledError:
                        result = "[Tool call cancelled by user]"
                        interrupted_during_tools = True
                    except Exception as e:
                        result = f"Error executing tool: {e}"
                    print(f"[tool] {tc['name']} returned ({len(result)} chars)")
                    pending_tool_calls.remove(tc)

                    if provider == "anthropic":
                        tool_result_parts.append(
                            AnthropicClient.make_tool_result_message(tc["id"], result)
                        )
                    else:
                        messages.append(
                            GeminiClient.make_tool_result_message(tc["name"], result)
                        )

                    if interrupted_during_tools:
                        break

                if interrupted_during_tools:
                    if provider == "anthropic" and tool_result_parts:
                        messages.append({"role": "user", "content": tool_result_parts})
                    break

                if provider == "anthropic" and tool_result_parts:
                    messages.append({"role": "user", "content": tool_result_parts})

                pending_tool_calls = []

                # Retry up to 3 times on transient failures
                follow_up_ok = False
                for attempt in range(3):
                    if _check_interrupt():
                        break
                    try:
                        response = await _cancellable(
                            llm_client.generate_response(
                                messages, tools, system_prompt
                            )
                        )
                        follow_up_ok = True
                        break
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        if attempt < 2:
                            wait = 2 ** attempt
                            print(f"\n[!] LLM request failed ({e}), retrying in {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            print(f"\n[!] LLM follow-up failed after 3 attempts: {e}")

                if not follow_up_ok:
                    break

                if _check_interrupt():
                    break

                if response["text"]:
                    print(f"\nAssistant: {response['text']}\n")

                tool_calls = llm_client.parse_tool_calls(response)

            # After the tool loop — completed normally or interrupted
            if pending_tool_calls:
                print("[*] Interrupted by user. You can now type a message.")
                stub_msg = "[Tool call interrupted by user before execution]"
                if provider == "anthropic":
                    stub_results = [
                        AnthropicClient.make_tool_result_message(tc["id"], stub_msg)
                        for tc in pending_tool_calls
                    ]
                    messages.append({"role": "user", "content": stub_results})
                else:
                    for tc in pending_tool_calls:
                        messages.append(
                            GeminiClient.make_tool_result_message(tc["name"], stub_msg)
                        )
                messages.append({
                    "role": "user",
                    "content": (
                        "[The user interrupted the tool call sequence. "
                        "All tool calls and results above are preserved. "
                        "Stop calling tools and wait for the user's next instruction.]"
                    ),
                })
                _flush_stdin()
            elif _check_interrupt():
                print("[*] Interrupted by user. You can now type a message.")
                messages.append({
                    "role": "user",
                    "content": (
                        "[The user interrupted the assistant. "
                        "Stop calling tools and wait for the user's next instruction.]"
                    ),
                })
                _flush_stdin()
            elif follow_up_ok:
                if provider == "anthropic":
                    messages.append(AnthropicClient.make_assistant_message(response))
                else:
                    messages.append(GeminiClient.make_assistant_message(response))

    finally:
        signal.signal(signal.SIGINT, prev_handler)


async def main():
    load_dotenv()

    # --- HTB integration (optional) ---
    htb_client = None
    system_prompt = build_system_prompt()
    htb_token = os.getenv("HTB_APP_TOKEN")
    if htb_token:
        try:
            htb_client = HTB(htb_token)
            print("[*] HTB client connected.")
            active = htb_client.get_active_machine()
            if active:
                print(f"[*] Active machine: {active['name']} ({active['os']}, "
                      f"{active['difficulty']}) @ {active['ip']}")
                system_prompt = build_system_prompt(
                    machine_name=active["name"],
                    machine_os=active["os"],
                    machine_difficulty=active["difficulty"],
                    machine_ip=active["ip"],
                )
            else:
                print("[*] No active HTB machine. Use /spawn <name> to start one.")
        except Exception as e:
            print(f"[!] Failed to initialize HTB client: {e}")
    else:
        print("[*] HTB_APP_TOKEN not set — HTB commands disabled.")

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
        await chat_loop(llm_client, mcp_client, system_prompt=system_prompt, htb_client=htb_client)
    finally:
        await mcp_client.disconnect()
        print("[*] MCP session closed.")
        if server_proc:
            server_proc.terminate()
            server_proc.wait()
            print("[*] Hexstrike server stopped.")


if __name__ == "__main__":
    asyncio.run(main())

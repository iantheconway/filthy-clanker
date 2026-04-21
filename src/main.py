"""
Filthy-Clanker — LangGraph Multi-Agent CTF Solver

Entry point. Orchestrates:
  1. Hexstrike Flask server startup
  2. MCP client connection
  3. LangGraph StateGraph with SqliteSaver checkpointing
  4. Interactive HITL loop with autonomous agent execution
  5. Trajectory logging to data/training/
"""
import asyncio
import logging
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime

import requests
import yaml
from dotenv import load_dotenv
from langgraph.types import Command

from config import build_system_prompt
from htb_client import HTB
from mcp_client import HexstrikeMCPClient, MCPClientPool
from graph import build_graph, create_checkpointer, TeamState
from graph.graph import initial_state
from data_capture import TrajectoryLogger

# ---------------------------------------------------------------------------
# Constants / environment
# ---------------------------------------------------------------------------
HEXSTRIKE_DIR = os.getenv("HEXSTRIKE_DIR", "/home/kali/hexstrike-ai")
HEXSTRIKE_PORT = os.getenv("HEXSTRIKE_PORT", "8888")
HEXSTRIKE_VENV_PYTHON = os.path.join(HEXSTRIKE_DIR, "hexstrike-env", "bin", "python3")
HEXSTRIKE_SERVER_SCRIPT = os.path.join(HEXSTRIKE_DIR, "hexstrike_server.py")
HEXSTRIKE_MCP_SCRIPT = os.path.join(HEXSTRIKE_DIR, "hexstrike_mcp.py")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.0.2.2:11434")

_sigint_received = False
_active_task: asyncio.Task | None = None


def setup_logging(session_id: str, project_root: str) -> str:
    """
    Configure the 'filthy_clanker' logger to write to both console and a
    per-session log file at <project_root>/logs/<session_id>.log.
    Returns the log file path.
    """
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{session_id}.log")

    log = logging.getLogger("filthy_clanker")
    log.setLevel(logging.INFO)
    log.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # File handler — full record of all agent activity
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    # Console handler — mirrors to stdout so the user sees it live
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    return log_path


def _sigint_handler(signum, frame):
    global _sigint_received
    _sigint_received = True
    if _active_task and not _active_task.done():
        _active_task.cancel()
    print("\n[*] Ctrl+C received — interrupting...")


def _check_interrupt() -> bool:
    global _sigint_received
    if _sigint_received:
        _sigint_received = False
        return True
    return False


async def _cancellable(coro):
    global _active_task
    task = asyncio.ensure_future(coro)
    _active_task = task
    try:
        return await task
    finally:
        _active_task = None


def _flush_stdin():
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hexstrike server management (unchanged from original)
# ---------------------------------------------------------------------------

def start_hexstrike_server() -> subprocess.Popen | None:
    url = f"http://127.0.0.1:{HEXSTRIKE_PORT}/api/cache/stats"
    try:
        if requests.get(url, timeout=3).ok:
            print("[*] Hexstrike server already running.")
            return None
    except Exception:
        pass

    print(f"[*] Starting hexstrike server on port {HEXSTRIKE_PORT}...")
    log_path = os.path.join(HEXSTRIKE_DIR, "server.log")
    server_log = open(log_path, "w")
    proc = subprocess.Popen(
        [HEXSTRIKE_VENV_PYTHON, HEXSTRIKE_SERVER_SCRIPT, "--port", HEXSTRIKE_PORT],
        cwd=HEXSTRIKE_DIR,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        if proc.poll() is not None:
            server_log.close()
            print(f"[!] Hexstrike server exited (code {proc.returncode}). Check {log_path}")
            sys.exit(1)
        try:
            if requests.get(url, timeout=3).ok:
                server_log.close()
                print("[*] Hexstrike server is ready.")
                return proc
        except Exception:
            pass
        time.sleep(1)

    proc.kill()
    server_log.close()
    sys.exit("Error: Hexstrike server failed to start within 30 seconds.")


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: str = "agents.yaml") -> dict:
    """Load and return the agents.yaml config dict."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    if not os.path.exists(config_path):
        sys.exit(f"Error: {config_path} not found. Copy agents.yaml.example or check the path.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_provider(config: dict) -> str | None:
    """
    Ask whether to use per-agent config or a single global provider override.
    Returns the override provider string, or None to use each agent's own config.
    """
    agents_cfg = config.get("agents", {})
    print("\nLLM provider configuration:")
    print("  1) Use per-agent config from agents.yaml")
    for name, cfg in agents_cfg.items():
        if name == "summarizer":
            continue
        p = cfg.get("provider", "?")
        m = cfg.get("model", "?")
        print(f"       {name}: {p} / {m}")
    print("  2) Override all agents — Anthropic (Claude)")
    print("  3) Override all agents — Gemini")
    print("  4) Override all agents — Ollama (local)")
    while True:
        choice = input("Enter 1–4 [1]: ").strip() or "1"
        if choice == "1":
            return None
        if choice == "2":
            return "anthropic"
        if choice == "3":
            return "gemini"
        if choice == "4":
            _select_and_set_ollama_model(config)
            return "ollama"
        print("Invalid choice.")


def _select_and_set_ollama_model(config: dict) -> None:
    """Prompt user to pick an Ollama model and store it in OLLAMA_MODEL env var.
    If the configured host is unreachable, prompts the user to enter the correct one."""
    default_host = (
        config.get("agents", {}).get("summarizer", {}).get("host")
        or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )

    # Try up to 3 hosts (let user correct if needed)
    host = default_host
    for attempt in range(3):
        try:
            resp = requests.get(f"{host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            break
        except Exception as e:
            print(f"[!] Could not reach Ollama at {host}: {e}")
            new_host = input(f"    Enter Ollama host URL (or Enter to skip Ollama): ").strip()
            if not new_host:
                print("[!] Skipping Ollama model selection. Set OLLAMA_HOST and OLLAMA_MODEL in .env to use Ollama.")
                return
            host = new_host.rstrip("/")
    else:
        print("[!] Could not connect to Ollama after 3 attempts.")
        return

    if not models:
        print(f"[!] No models found on Ollama at {host}.")
        return

    print(f"\nAvailable Ollama models (from {host}):")
    for i, name in enumerate(models, 1):
        print(f"  {i}) {name}")
    while True:
        choice = input(f"Select model [1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected = models[int(choice) - 1]
            os.environ["OLLAMA_MODEL"] = selected
            os.environ["OLLAMA_HOST"] = host
            print(f"[*] Ollama model: {selected}")
            # Also update the summarizer host in the loaded config so compaction works
            config.setdefault("agents", {}).setdefault("summarizer", {})["host"] = host
            return
        print("Invalid choice.")


# ---------------------------------------------------------------------------
# Session management (using SqliteSaver thread IDs)
# ---------------------------------------------------------------------------

def list_graph_sessions(db_path: str) -> list[dict]:
    """List all saved graph sessions from the SQLite checkpoint DB."""
    import sqlite3
    sessions = []
    if not os.path.exists(db_path):
        return sessions
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # LangGraph checkpoint table structure
        cursor.execute(
            "SELECT DISTINCT thread_id, MAX(checkpoint_id) as latest "
            "FROM checkpoints GROUP BY thread_id ORDER BY latest DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        for thread_id, checkpoint_id in rows:
            sessions.append({"thread_id": thread_id, "checkpoint_id": checkpoint_id})
    except Exception:
        pass
    return sessions


# ---------------------------------------------------------------------------
# HTB command helpers (preserved from original)
# ---------------------------------------------------------------------------

def _handle_htb_command(cmd: str, htb_client: HTB | None) -> str | None:
    """Handle /machine, /spawn, /stop, /reset, /flag, /vpn commands.
    Returns new IP if machine changed, else None."""
    if not htb_client:
        print("[!] HTB integration not configured (set HTB_APP_TOKEN).")
        return None

    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command == "/machine":
        info = htb_client.get_machine_info(arg) if arg else htb_client.get_active_machine()
        if not info:
            print("[*] No active machine.")
        elif isinstance(info, str):
            print(f"[!] {info}")
        else:
            print(f"[*] Machine: {info.get('name')} | OS: {info.get('os')} | "
                  f"IP: {info.get('ip', 'N/A')} | {info.get('difficulty')}")
        return None

    elif command == "/spawn":
        if not arg:
            print("[!] Usage: /spawn <machine-name>")
            return None
        print(f"[*] Spawning '{arg}'...")
        result = htb_client.spawn_machine(arg)
        if isinstance(result, str) and result.startswith("Error"):
            print(f"[!] {result}")
            return None
        print(f"[*] Machine spawned at {result}")
        return result

    elif command == "/stop":
        print(f"[*] {htb_client.stop_machine()}")
    elif command == "/reset":
        print(f"[*] {htb_client.reset_machine()}")
    elif command == "/flag":
        if not arg:
            print("[!] Usage: /flag <flag-string>")
        else:
            print(f"[*] {htb_client.submit_flag(arg)}")
    elif command == "/vpn":
        print(f"[*] {htb_client.get_vpn_status()}")
    else:
        print(f"[!] Unknown HTB command: {command}")
    return None


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def _print_agent_output(node_name: str, output: dict) -> None:
    """Pretty-print state updates emitted by an agent node."""
    if node_name == "supervisor":
        next_target = output.get("next", "")
        if next_target and next_target != "__end__":
            print(f"\n{'='*60}")
            print(f"  [Supervisor] → Routing to: {next_target.upper()}")
            print(f"{'='*60}")
        return

    messages = output.get("messages", [])
    # Unwrap the compaction replacement sentinel — it's a dict, not a list.
    if isinstance(messages, dict) and "__replace__" in messages:
        messages = messages["__replace__"]
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            print(f"\n[{node_name.upper()}] {text}\n")
            elif isinstance(content, str) and content.strip():
                print(f"\n[{node_name.upper()}] {content.strip()}\n")

    kb = output.get("knowledge_base")
    if kb:
        flags = kb.get("flags", [])
        if flags:
            print(f"\n{'!'*60}")
            print(f"  FLAG CAPTURED: {flags}")
            print(f"{'!'*60}\n")


# ---------------------------------------------------------------------------
# Main graph session loop
# ---------------------------------------------------------------------------

async def run_session(
    graph,
    session_id: str,
    task: str,
    provider: str,
    config: dict,
    trajectory_logger: TrajectoryLogger,
    target_ip: str = "",
    machine_name: str = "",
    resume: bool = False,
) -> None:
    """
    Drive the LangGraph graph for one CTF session.
    Handles streaming output, HITL interrupts, and trajectory logging.
    """
    thread_config: dict = {"configurable": {"thread_id": session_id}}

    # Attach LangSmith tracer so all LLM calls and tool calls appear as
    # child spans under the LangGraph node traces in Smith.
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        try:
            from langchain_core.tracers.langchain import LangChainTracer
            _project = os.getenv("LANGSMITH_PROJECT", "filthy-clanker")
            thread_config["callbacks"] = [LangChainTracer(project_name=_project)]
            print(f"[*] LangSmith tracing enabled → project '{_project}'")
        except Exception as _e:
            print(f"[!] LangSmith tracer init failed: {_e}")

    if resume:
        # Resume from checkpoint
        current_graph_state = await graph.aget_state(thread_config)
        if current_graph_state.values:
            print(f"[*] Resuming session {session_id} from checkpoint.")
            trajectory_logger.set_session(session_id)

            # Check whether there is a pending interrupt() call waiting for a value
            pending_interrupts = current_graph_state.tasks and any(
                hasattr(t, "interrupts") and t.interrupts
                for t in current_graph_state.tasks
            )

            if task and not pending_interrupts:
                # User provided a task override but there is no pending interrupt.
                # Command(resume=...) would be silently ignored in this case.
                # Instead, inject the override message and re-schedule via the
                # compaction → supervisor unconditional edge so the supervisor
                # sees the full KB + override and makes a fresh routing decision.
                override_msg = {"role": "user", "content": f"[Human Override]: {task}"}
                await graph.aupdate_state(
                    thread_config,
                    {"messages": [override_msg]},
                    as_node="compaction",
                )
                print(f"[*] Task override injected — routing through supervisor.")
                input_payload = None  # astream(None, config) continues from updated checkpoint
            else:
                # Either no override, or there is a pending interrupt that needs a value.
                resume_text = f"[Resuming session. Continue from where we left off. Task: {task}]" if task else "Continue from where we left off."
                input_payload = Command(resume=resume_text)
        else:
            print(f"[!] No checkpoint found for {session_id}. Starting fresh.")
            resume = False

    if not resume:
        state = initial_state(task, provider, session_id, config, target_ip, machine_name)
        input_payload = state
        trajectory_logger.set_session(session_id)

    print(f"\n[*] Session ID: {session_id}")
    print("[*] Running autonomous agents. Press Ctrl+C to interrupt.\n")

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            _check_interrupt()

            # Stream graph execution node-by-node
            interrupted = False
            try:
                async for event in graph.astream(input_payload, thread_config, stream_mode="updates"):
                    if _check_interrupt():
                        interrupted = True
                        break
                    for node_name, output in event.items():
                        if node_name == "__interrupt__":
                            # HITL interrupt from supervisor
                            interrupt_data = output[0].value if output else {}
                            reason = interrupt_data.get("reason", "manual_intervention")
                            message = interrupt_data.get("message", "Human input required.")
                            print(f"\n{'*'*60}")
                            print(f"  [HITL INTERRUPT — {reason}]")
                            print(f"{'*'*60}")
                            print(f"{message}\n")
                            try:
                                human_response = input("Your response (or Enter to skip): ").strip()
                            except (EOFError, KeyboardInterrupt):
                                human_response = "Continue with best effort."
                            input_payload = Command(resume=human_response or "Continue with best effort.")
                            interrupted = True
                            break
                        else:
                            _print_agent_output(node_name, output)

                        # Log trajectories for agent nodes
                        if node_name in ("recon", "exploit", "privesc", "webexplorer", "refusal_specialist"):
                            _log_node_trajectories(node_name, output, trajectory_logger)

                    if interrupted:
                        break

            except asyncio.CancelledError:
                interrupted = True
                print("\n[*] Agent loop interrupted by user.")

            except Exception as e:
                print(f"\n[!] Graph execution error: {e}")
                import traceback
                traceback.print_exc()
                break

            if interrupted and not isinstance(input_payload, Command):
                # User pressed Ctrl+C (not a HITL interrupt) — ask what to do
                print("\nOptions:")
                print("  1) Continue running")
                print("  2) Inject a message to agents")
                print("  3) Save and exit")
                print("  4) Exit without saving")
                choice = input("Choice [1]: ").strip() or "1"

                if choice == "1":
                    # Re-invoke from last checkpoint
                    input_payload = Command(resume="Continue autonomously.")
                    continue
                elif choice == "2":
                    user_msg = input("Message to agents: ").strip()
                    if user_msg:
                        input_payload = Command(resume=user_msg)
                        continue
                    else:
                        input_payload = Command(resume="Continue autonomously.")
                        continue
                else:
                    break

            elif isinstance(input_payload, Command):
                # We sent a Command(resume=...) — check if graph still has work
                graph_state = await graph.aget_state(thread_config)
                if not graph_state.next:
                    print("[*] Graph completed.")
                    break
                # Graph paused again at interrupt or completed — loop back
                continue

            else:
                # Normal completion
                break

    finally:
        signal.signal(signal.SIGINT, prev_handler)
        # Record session end
        try:
            final_state = await graph.aget_state(thread_config)
            if final_state.values:
                trajectory_logger.record_session_end(final_state.values)
                stats = trajectory_logger.get_stats()
                print(f"\n[DataCapture] Trajectories: {stats.get('records', 0)} records, "
                      f"avg score: {stats.get('avg_score', 0)}, "
                      f"high-value: {stats.get('high_value', 0)}")
        except Exception:
            pass


def _log_node_trajectories(node_name: str, output: dict, logger: TrajectoryLogger) -> None:
    """Extract tool call/result pairs from node output and log as trajectories."""
    messages = output.get("messages", [])
    kb_after = output.get("knowledge_base", {})

    # Find tool result messages and pair with preceding tool calls
    # This is a best-effort extraction from the message list
    for i, msg in enumerate(messages):
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    result_text = ""
                    result_content = block.get("content", [])
                    if isinstance(result_content, list):
                        result_text = " ".join(
                            b.get("text", "") for b in result_content
                            if isinstance(b, dict)
                        )
                    elif isinstance(result_content, str):
                        result_text = result_content

                    # Find the corresponding tool_use block
                    tool_name = ""
                    tool_args = {}
                    for prev_msg in reversed(messages[:i]):
                        prev_content = prev_msg.get("content", [])
                        if isinstance(prev_content, list):
                            for b in prev_content:
                                if (isinstance(b, dict) and
                                        b.get("type") == "tool_use" and
                                        b.get("id") == tool_use_id):
                                    tool_name = b.get("name", "")
                                    tool_args = b.get("input", {})

                    if tool_name and result_text:
                        # Use minimal state snapshots for logging
                        state_before = {"knowledge_base": {}, "task": "", "session_id": ""}
                        state_after = {"knowledge_base": kb_after, "exploit_attempts": output.get("exploit_attempts", 0)}
                        logger.record(
                            state_before=state_before,
                            action={"tool_name": tool_name, "arguments": tool_args},
                            result=result_text,
                            state_after=state_after,
                            agent_name=node_name,
                        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    load_dotenv()

    # --- Load config ---
    config = load_config("agents.yaml")
    settings = config.get("settings", {})
    checkpoint_db = settings.get("checkpoint_db", "sessions/checkpoint.db")
    training_dir = settings.get("training_data_dir", "data/training")

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_db = os.path.join(project_root, checkpoint_db)
    training_dir = os.path.join(project_root, training_dir)

    # --- HTB integration ---
    htb_client = None
    active_machine = None
    htb_token = os.getenv("HTB_APP_TOKEN")
    if htb_token:
        try:
            htb_client = HTB(htb_token)
            print("[*] HTB client connected.")
            active_machine = htb_client.get_active_machine()
            if active_machine:
                print(f"[*] Active machine: {active_machine['name']} "
                      f"({active_machine['os']}, {active_machine['difficulty']}) "
                      f"@ {active_machine['ip']}")
            else:
                print("[*] No active HTB machine.")
        except Exception as e:
            print(f"[!] Failed to initialize HTB client: {e}")
    else:
        print("[*] HTB_APP_TOKEN not set — HTB commands disabled.")

    # --- Hexstrike server ---
    server_proc = start_hexstrike_server()

    # --- Provider selection ---
    provider = select_provider(config)

    # --- MCP client pool ---
    mcp_pool = MCPClientPool()

    # Hexstrike server (security tools)
    mcp_command = os.getenv("MCP_COMMAND", HEXSTRIKE_VENV_PYTHON)
    mcp_args_str = os.getenv(
        "MCP_ARGS",
        f"{HEXSTRIKE_MCP_SCRIPT} --server http://127.0.0.1:{HEXSTRIKE_PORT}",
    )
    mcp_args = mcp_args_str.split() if mcp_args_str else []
    hexstrike_client = HexstrikeMCPClient(command=mcp_command, args=mcp_args)
    mcp_pool.add_server("hexstrike", hexstrike_client)

    # Brave Search server (web search — optional, requires BRAVE_API_KEY)
    brave_api_key = os.getenv("BRAVE_API_KEY", "")
    if brave_api_key:
        brave_client = HexstrikeMCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": brave_api_key},
        )
        mcp_pool.add_server("brave-search", brave_client)
        print("[*] Brave Search MCP server configured.")
    else:
        print("[*] BRAVE_API_KEY not set — Brave Search tools unavailable.")

    print(f"[*] Connecting to MCP servers...")
    try:
        await mcp_pool.connect()
    except Exception as e:
        if server_proc:
            server_proc.terminate()
        sys.exit(f"Failed to connect to MCP servers: {e}")
    print("[*] MCP sessions initialized.")

    # List available tools
    tools = await mcp_pool.list_tools()
    print(f"[*] {len(tools)} MCP tools available.")

    # Alias so the rest of the function uses a single name
    mcp_client = mcp_pool

    trajectory_logger = TrajectoryLogger(training_dir)

    # --- Session selection (before opening checkpointer) ---
    print("\nSession options:")
    print("  1) New session")
    sessions = list_graph_sessions(checkpoint_db)
    for i, s in enumerate(sessions[:5], 2):
        print(f"  {i}) Resume: {s['thread_id']}")
    print()

    session_id = None
    resume = False
    task = ""
    target_ip = ""
    machine_name = ""

    while True:
        choice = input("Choice [1]: ").strip() or "1"
        if choice == "1":
            session_id = f"session-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
            if active_machine:
                machine_name = active_machine["name"]
                target_ip = active_machine["ip"]
                task = (
                    f"Hack the HackTheBox machine '{machine_name}' "
                    f"({active_machine['os']}, {active_machine['difficulty']}) "
                    f"at IP {target_ip}. Hostname: {machine_name.lower()}.htb. "
                    f"Capture user.txt and root.txt flags."
                )
                print(f"[*] Task auto-set from active HTB machine: {machine_name}")
            else:
                task = input("Enter target/task description: ").strip()
                if not task:
                    task = "Enumerate and exploit the target machine. Capture all flags."
                ip_input = input("Target IP (optional): ").strip()
                if ip_input:
                    target_ip = ip_input
            print(f"[*] New session: {session_id}")
            break
        elif choice.isdigit() and 2 <= int(choice) < 2 + len(sessions):
            idx = int(choice) - 2
            session_id = sessions[idx]["thread_id"]
            resume = True
            task = input("Override task (Enter to keep original): ").strip()
            print(f"[*] Resuming session: {session_id}")
            break
        else:
            print("Invalid choice.")

    # --- HTB commands before starting ---
    print("\nCommands during session:")
    print("  Ctrl+C  — interrupt and show options")
    if htb_client:
        print("  Use /spawn, /stop, /reset, /flag, /vpn before starting the graph")
        while True:
            cmd = input("\nHTB command (or Enter to start): ").strip()
            if not cmd:
                break
            if cmd.startswith("/"):
                new_ip = _handle_htb_command(cmd, htb_client)
                if new_ip and cmd.startswith("/spawn"):
                    active_machine = htb_client.get_active_machine()
                    if active_machine:
                        target_ip = active_machine["ip"]
                        machine_name = active_machine["name"]
                        task = (
                            f"Hack the HackTheBox machine '{machine_name}' "
                            f"({active_machine['os']}, {active_machine['difficulty']}) "
                            f"at IP {target_ip}. Hostname: {machine_name.lower()}.htb. "
                            f"Capture user.txt and root.txt flags."
                        )
                        print(f"[*] Task updated for {machine_name} @ {target_ip}")

    # --- Set up per-session logging ---
    log_path = setup_logging(session_id, project_root)
    print(f"[*] Agent log: {log_path}")

    # --- Build LangGraph and run (checkpointer must stay open for the session) ---
    try:
        async with create_checkpointer(checkpoint_db) as checkpointer:
            graph = build_graph(mcp_client, config, checkpointer)
            await run_session(
                graph=graph,
                session_id=session_id,
                task=task,
                provider=provider,
                config=config,
                trajectory_logger=trajectory_logger,
                target_ip=target_ip,
                machine_name=machine_name,
                resume=resume,
            )
            print(f"\n[*] Session {session_id} checkpointed to {checkpoint_db}")
            print(f"[*] Training data saved to {training_dir}/")
    finally:
        await mcp_client.disconnect()
        print("[*] MCP session closed.")
        if server_proc:
            server_proc.terminate()
            server_proc.wait()
            print("[*] Hexstrike server stopped.")


if __name__ == "__main__":
    asyncio.run(main())

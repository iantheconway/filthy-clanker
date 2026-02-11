import asyncio
import os
import sys

from dotenv import load_dotenv

from config import DEFAULT_SYSTEM_PROMPT
from llms import AnthropicClient, GeminiClient
from mcp_client import HexstrikeMCPClient


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

    provider = select_provider()
    llm_client = build_llm_client(provider)

    # Configure MCP server command.  Adjust path/args to your hexstrike binary.
    mcp_command = os.getenv("MCP_COMMAND", "hexstrike-mcp")
    mcp_args = os.getenv("MCP_ARGS", "")
    args = mcp_args.split() if mcp_args else []

    mcp_client = HexstrikeMCPClient(command=mcp_command, args=args)

    print(f"[*] Connecting to MCP server: {mcp_command} {' '.join(args)}")
    try:
        await mcp_client.connect()
    except Exception as e:
        sys.exit(f"Failed to connect to MCP server: {e}")

    print("[*] MCP session initialized.")

    try:
        await chat_loop(llm_client, mcp_client)
    finally:
        await mcp_client.disconnect()
        print("[*] MCP session closed.")


if __name__ == "__main__":
    asyncio.run(main())

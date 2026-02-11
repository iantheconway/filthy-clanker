from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_response(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Send messages to the LLM and return a structured response.

        Returns a dict with keys:
            - "text": optional textual reply from the model
            - "tool_calls": list of dicts with "name" and "arguments"
            - "raw": the raw SDK response object
        """
        ...

    @abstractmethod
    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract tool call dicts from generate_response output.

        Each dict has:
            - "id": call id (provider-specific)
            - "name": tool name
            - "arguments": dict of arguments
        """
        ...

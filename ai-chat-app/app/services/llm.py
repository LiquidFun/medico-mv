import os
from openai import AsyncOpenAI
from typing import AsyncGenerator, Optional
import json


class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://llm-dev-01.planet-ai.de/v1/"),
            api_key=os.getenv("LLM_API_KEY", "pai"),
        )
        self.model = os.getenv("LLM_MODEL_NAME", "pai-chat")

    async def stream_chat_completion(
        self, messages: list[dict[str, str]], tools: Optional[list[dict]] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Stream chat completion responses from the LLM with optional tool support.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool definitions for function calling

        Yields:
            Dictionary with 'type' and content:
            - {'type': 'content', 'content': str} for text chunks
            - {'type': 'tool_call', 'tool_name': str, 'arguments': dict} for tool calls
        """
        print(f"\n{'='*80}")
        print(f"DEBUG LLM: Starting stream_chat_completion")
        print(f"DEBUG LLM: Model: {self.model}")
        print(f"DEBUG LLM: Tools provided: {tools is not None}")
        if tools:
            print(f"DEBUG LLM: Number of tools: {len(tools)}")
            print(f"DEBUG LLM: Tool names: {[t['function']['name'] for t in tools]}")
        print(f"DEBUG LLM: Messages count: {len(messages)}")
        print(f"{'='*80}\n")

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            print(f"DEBUG LLM: Added tools to request with tool_choice='auto'")

        print(f"DEBUG LLM: Creating completion stream...")
        stream = await self.client.chat.completions.create(**kwargs)
        print(f"DEBUG LLM: Stream created successfully")

        tool_call_id = None
        tool_name = None
        tool_args = ""
        chunk_count = 0

        async for chunk in stream:
            chunk_count += 1
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            print(f"DEBUG LLM: Chunk #{chunk_count}, finish_reason={finish_reason}, has_tool_calls={delta.tool_calls is not None}, has_content={delta.content is not None}")

            # Handle tool calls
            if delta.tool_calls:
                print(f"DEBUG LLM: Got tool_calls in delta: {delta.tool_calls}")
                for tool_call in delta.tool_calls:
                    if tool_call.id:
                        tool_call_id = tool_call.id
                        print(f"DEBUG LLM: Tool call ID: {tool_call_id}")
                    if tool_call.function.name:
                        tool_name = tool_call.function.name
                        print(f"DEBUG LLM: Tool name: {tool_name}")
                    if tool_call.function.arguments:
                        tool_args += tool_call.function.arguments
                        print(f"DEBUG LLM: Tool args chunk: {tool_call.function.arguments}")

            # Handle regular content
            if delta.content:
                print(f"DEBUG LLM: Got content chunk: '{delta.content[:50]}'...")
                yield {"type": "content", "content": delta.content}

            # Check if streaming finished and we have a complete tool call
            # This can happen in the same chunk or a later chunk
            if finish_reason == "tool_calls" and tool_name:
                print(f"DEBUG LLM: Finish reason is 'tool_calls', processing complete tool call")
                print(f"DEBUG LLM: Complete tool args: {tool_args}")
                try:
                    arguments = json.loads(tool_args)
                    print(f"DEBUG LLM: Parsed arguments successfully: {arguments}")
                    result = {
                        "type": "tool_call",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": arguments
                    }
                    print(f"DEBUG LLM: Yielding tool call: {result}")
                    yield result
                except json.JSONDecodeError as e:
                    print(f"DEBUG LLM: ERROR parsing tool args: {e}")
                    yield {"type": "content", "content": f"\n\n[Error: Invalid tool call arguments]"}

        print(f"DEBUG LLM: Stream completed after {chunk_count} chunks")
        print(f"DEBUG LLM: Final state - tool_name={tool_name}, tool_args length={len(tool_args)}")

        # Fallback: If we have tool data but never got finish_reason="tool_calls", yield it now
        if tool_name and tool_args and not finish_reason:
            print(f"DEBUG LLM: Stream ended without finish_reason='tool_calls', but we have tool data. Yielding anyway.")
            try:
                arguments = json.loads(tool_args)
                print(f"DEBUG LLM: Parsed arguments successfully: {arguments}")
                result = {
                    "type": "tool_call",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments": arguments
                }
                print(f"DEBUG LLM: Yielding tool call: {result}")
                yield result
            except json.JSONDecodeError as e:
                print(f"DEBUG LLM: ERROR parsing tool args: {e}")
                yield {"type": "content", "content": f"\n\n[Error: Invalid tool call arguments]"}

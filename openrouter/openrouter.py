from __future__ import annotations
import json
import time
from copy import deepcopy
from typing import Iterator, AsyncIterator

from dotenv import load_dotenv
from pydantic import BaseModel

from openrouter.llms import *
from openrouter.message import *
from openrouter.openrouter_provider import *
from openrouter.tool import *


_base_system_prompt = """
It's [TIME] today.
You are an intelligent AI. You must follow the system_instruction below, which is provided by the user.

<system_instruction>
[SYSTEM_INSTRUCTION]
</system_instruction>
"""

class OpenRouterClient:
    def __init__(self, system_prompt: str = "", tools: list[tool_model] = None) -> None:
        load_dotenv()
        
        self._memory: list[Message] = []
        self.tools: list[tool_model] = tools or []
        self.set_system_prompt(system_prompt)
        
    def set_system_prompt(self, prompt: str) -> None:
        month, day, year = time.localtime()[:3]

        system_prompt = _base_system_prompt
        system_prompt = system_prompt.replace("[TIME]", f"{month}/{day}/{year}")
        system_prompt = system_prompt.replace("[SYSTEM_INSTRUCTION]", prompt)

        self._system_prompt = Message(text=system_prompt, role=Role.system)

    def _execute_tools(self, reply: Message, tools: list[tool_model]) -> Message:
        if not reply.tool_calls:
            return reply

        reply_copy = deepcopy(reply)

        for requested_tool in reply_copy.tool_calls:
            args = requested_tool.arguments
            if isinstance(args, str):
                args = json.loads(args)

            for tool in tools:
                if tool.name == requested_tool.name:
                    result = tool(**args)
                    requested_tool.result = result
                    break

        return reply_copy

    def execute_tool(self, reply: Message, tool_index: int, tools: List[tool_model] = []) -> Message:
        if not reply.tool_calls:
            return reply

        if tool_index < 0 or tool_index >= len(reply.tool_calls):
            raise IndexError(f"Tool index {tool_index} is out of range. Available tools: {len(reply.tool_calls)}")

        requested_tool = reply.tool_calls[tool_index]

        args = requested_tool.arguments
        if isinstance(args, str):
            args = json.loads(args)

        all_tools = self.tools + tools
        for tool in all_tools:
            if tool.name == requested_tool.name:
                result = tool(**args)
                requested_tool.result = result
                break
        else:
            raise ValueError(f"Tool '{requested_tool.name}' not found in registered tools")

        for i, msg in enumerate(self._memory):
            if msg.id == reply.id:
                self._memory[i] = reply
                break

        return reply

    def clear_memory(self) -> None:
        self._memory = []
        
    def print_memory(self) -> None:
        from tqdm import tqdm
        
        reset_code = "\033[0m"
        
        for message in self._memory:
            role = message.role.value
            text = message.text.strip()
            
            role_str = f"{role.ljust(9)}:"
            indent = " " * len(role_str)
            lines = text.splitlines()
            
            color_codes = {
                "user": "\033[94m",
                "assistant": "\033[92m", 
                "tool": "\033[93m",
                "default": "\033[0m"
            }
            
            color_code = color_codes.get(role, color_codes["default"])
            
            if role in ["user", "assistant"]:
                if lines:
                    print(f"{color_code}{role_str}{reset_code} {lines[0]}")
                    for line in lines[1:]:
                        print(f"{color_code}{indent}{reset_code} {line}")
                else:
                    print(f"{color_code}{role_str}{reset_code}")
            
            elif role == "tool":
                print(f"{color_code}{role_str}{reset_code} ", end="")
                for tool in message.tool_calls:
                    print(f"{tool.name}({json.loads(tool.arguments)}), ", end="")
                print()

    def invoke(
        self,
        model: LLMModel,
        query: Message = None,
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3,
        auto_tool_exec: bool = True
    ) -> Message:
        tools = tools or []
        if query is not None:
            self._memory.append(query)
        client = OpenRouterProvider()

        reply = client.invoke(
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
            querys=self._memory,
            tools=self.tools + tools,
            provider=provider,
        )
        reply.answered_by = model
        self._memory.append(reply)

        if auto_tool_exec and reply.tool_calls:
            reply = self._execute_tools(reply, self.tools + tools)
            self._memory[-1] = reply

            reply = client.invoke(
                model=model,
                temperature=temperature,
                system_prompt=self._system_prompt,
                querys=self._memory,
                provider=provider
            )
            reply.answered_by = model
            self._memory.append(reply)

        return reply
    
    def invoke_stream(
        self,
        model: LLMModel,
        query: Message,
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> Iterator[str]:
        tools = tools or []
        self._memory.append(query)
        client = OpenRouterProvider()
        generator = client.invoke_stream(
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
            querys=self._memory,
            tools=self.tools + tools,
            provider=provider
        )
        
        text = ""
        for token in generator:
            text += token.choices[0].delta.content
            yield token.choices[0].delta.content

        self._memory.append(Message(text=text, role=Role.ai, answered_by=model))
        
    async def async_invoke(
        self,
        model: LLMModel,
        query: Message = None,
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3,
        auto_tool_exec: bool = True
    ) -> Message:
        tools = tools or []
        if query is not None:
            self._memory.append(query)
        client = OpenRouterProvider()
        reply = await client.async_invoke(
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
            querys=self._memory,
            tools=self.tools + tools,
            provider=provider
        )
        reply.answered_by = model
        self._memory.append(reply)

        if auto_tool_exec and reply.tool_calls:
            reply = self._execute_tools(reply, self.tools + tools)

            reply = await client.async_invoke(
                model=model,
                system_prompt=self._system_prompt,
                querys=self._memory,
                tools=self.tools + tools,
                provider=provider
            )
            reply.answered_by = model
            self._memory.append(reply)

        return reply

    async def async_invoke_stream(
        self,
        model: LLMModel,
        query: Message,
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> AsyncIterator[str]:
        tools = tools or []
        self._memory.append(query)
        client = OpenRouterProvider()

        stream = client.async_invoke_stream(
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
            querys=self._memory,
            tools=self.tools + tools,
            provider=provider
        )

        text = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            text += delta
            yield delta

        self._memory.append(Message(text=text, role=Role.ai, answered_by=model))
        
    def structured_output(
        self,
        model: LLMModel,
        query: Message,
        provider: ProviderConfig = None,
        json_schema: BaseModel = None,
        temperature: float = 0.3
    ) -> BaseModel:
        self._memory.append(query)
        client = OpenRouterProvider()
        reply = client.structured_output(
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
            querys=self._memory,
            provider=provider,
            json_schema=json_schema
        )
        
        self._memory.append(Message(text=reply.model_dump_json(), role=Role.ai, answered_by=model))
        
        return reply


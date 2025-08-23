from __future__ import annotations
import json
import time
from typing import Iterator, AsyncIterator

from dotenv import load_dotenv
from pydantic import BaseModel

from .llms import *
from .message import *
from .openrouter_provider import *
from .tool import *


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
        query: Message,
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> Message:
        tools = tools or []
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

        if reply.tool_calls:
            for requested_tool in reply.tool_calls:
                args = requested_tool.arguments
                if isinstance(args, str):
                    args = json.loads(args)

                for tool in (self.tools + tools):
                    if tool.name == requested_tool.name:
                        result = tool(**args)
                        requested_tool.result = result
                        break
                else:
                    return reply
                    
            reply = client.invoke(
                model=model,
                system_prompt=self._system_prompt,
                querys=self._memory,
                tools=self.tools + tools,
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
        query: Message,
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> Message:
        tools = tools or []
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

        if reply.tool_calls:
            for requested_tool in reply.tool_calls:
                args = requested_tool.arguments
                if isinstance(args, str):
                    args = json.loads(args)

                for tool in (self.tools + tools):
                    if tool.name == requested_tool.name:
                        result = tool(**args)
                        requested_tool.result = result
                        break
                else:
                    return reply

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
        return reply

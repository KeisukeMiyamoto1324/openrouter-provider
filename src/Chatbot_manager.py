from .Chat_message import *
from .OpenRouterProvider import *
from .LLMs import LLMModel

from dotenv import load_dotenv
import time
import json
from typing import Iterator, AsyncIterator
from pydantic import BaseModel


_base_system_prompt = """
It's [TIME] today.
You are an intelligent AI. You must follow the system_instruction below, which is provided by the user.

<system_instruction>
[SYSTEM_INSTRUCTION]
</system_instruction>
"""

class Chatbot_manager:
    def __init__(self, system_prompt:str="", tools:list[tool_model]=[]) -> None:
        load_dotenv()
        
        self._memory: list[Chat_message] = []
        self.tools: list[tool_model] = tools
        self.set_system_prompt(prompt=system_prompt)
        
    def set_system_prompt(self, prompt: str):
        m, d, y = time.localtime()[:3]

        system_prompt = _base_system_prompt
        system_prompt = system_prompt.replace("[TIME]", f"{m}/{d}/{y}")
        system_prompt = system_prompt.replace("[SYSTEM_INSTRUCTION]", prompt)
        
        self._system_prompt = Chat_message(text=system_prompt, role=Role.system)
        
    def clear_memory(self):
        self._memory = []
        
    def print_memory(self):
        print("\n--------------------- Chatbot memory ---------------------")
        print(f"system   : {self._system_prompt.text}")

        for message in self._memory:
            role = message.role.value
            text = message.text.strip()
            
            reset_code = "\033[0m"
            role_str = f"{role.ljust(9)}:"
            indent = " " * len(role_str)
            lines = text.splitlines()
            
            if role == "user":
                color_code = "\033[94m"  # blue
                if lines:
                    print(f"{color_code}{role_str}{reset_code} {lines[0]}")
                    for line in lines[1:]:
                        print(f"{color_code}{indent}{reset_code} {line}")
                else:
                    print(f"{color_code}{role_str}{reset_code}")
            
            elif role == "assistant":
                color_code = "\033[92m"  # green
                if lines:
                    print(f"{color_code}{role_str}{reset_code} {lines[0]}")
                    for line in lines[1:]:
                        print(f"{color_code}{indent}{reset_code} {line}")
                else:
                    print(f"{color_code}{role_str}{reset_code}")

            elif role == "tool":
                color_code = "\033[93m"  # orange
                print(f"{color_code}{role_str}{reset_code} ", end="")
                
                for tool in message.tool_calls:
                    print(f"{tool.name}({json.loads(tool.arguments)}), ", end="")
                print()
                
            else:
                color_code = "\033[0m"   # default color
                print("Print error: The role is invalid.")
            
        print("----------------------------------------------------------\n")

    def invoke(
        self, 
        model: LLMModel, 
        query: Chat_message, 
        tools: list[tool_model]=[], 
        provider:ProviderConfig=None,
        temperature: float=0.3
    ) -> Chat_message:
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
        reply.answeredBy = model
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
                    print("Tool Not found", requested_tool.name)
                    return reply
                    
            reply = client.invoke(
                model=model,
                system_prompt=self._system_prompt,
                querys=self._memory,
                tools=self.tools + tools,
                provider=provider
            )
            
            reply.answeredBy = model
            self._memory.append(reply)

        return reply
    
    def invoke_stream(
        self, 
        model: LLMModel, 
        query: Chat_message, 
        tools: list[tool_model]=[], 
        provider:ProviderConfig=None,
        temperature: float=0.3
    ) -> Iterator[str]:
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

        self._memory.append(Chat_message(text=text, role=Role.ai, answerdBy=LLMModel))
        
    async def async_invoke(
        self, 
        model: LLMModel, 
        query: Chat_message, 
        tools: list[tool_model] = [], 
        provider: ProviderConfig = None,
        temperature: float=0.3
    ) -> Chat_message:
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
        reply.answeredBy = model
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
                    print("Tool Not found", requested_tool.name)
                    return reply

            reply = await client.async_invoke(
                model=model,
                system_prompt=self._system_prompt,
                querys=self._memory,
                tools=self.tools + tools,
                provider=provider
            )
            reply.answeredBy = model
            self._memory.append(reply)

        return reply

    async def async_invoke_stream(
        self, 
        model: LLMModel, 
        query: Chat_message, 
        tools: list[tool_model] = [], 
        provider: ProviderConfig = None,
        temperature: float=0.3
    ) -> AsyncIterator[str]:
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

        self._memory.append(Chat_message(text=text, role=Role.ai, answerdBy=model))
        
    def structured_output(
        self, 
        model: LLMModel, 
        query: Chat_message, 
        provider:ProviderConfig=None, 
        json_schema: BaseModel=None,
        temperature: float=0.3
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

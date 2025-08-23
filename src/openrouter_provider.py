from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal, Iterator, AsyncIterator

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from .message import Message, Role, ToolCall
from .tool import tool_model
from .llms import LLMModel




@dataclass
class ProviderConfig:
    order: Optional[List[str]] = None
    allow_fallbacks: bool = None
    require_parameters: bool = None
    data_collection: Literal["allow", "deny"] = None
    only: Optional[List[str]] = None
    ignore: Optional[List[str]] = None
    quantizations: Optional[List[str]] = None
    sort: Optional[Literal["price", "throughput"]] = None
    max_price: Optional[dict] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class OpenRouterProvider:
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def make_prompt(
        self,
        system_prompt: Message,
        querys: list[Message]
    ) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt.text}]

        for query in querys:
            if query.role == Role.user:
                if query.images is None:
                    messages.append({"role": "user", "content": query.text})
                else:
                    content = [{"type": "text", "text": query.text}]
                    for img in query.images[:50]:
                        content.append(
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                    messages.append({"role": "user", "content": content})

            elif query.role == Role.ai or query.role == Role.tool:
                assistant_msg = {"role": "assistant"}
                assistant_msg["content"] = query.text or None

                if query.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": str(t.id),
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "arguments": t.arguments
                            }
                        }
                        for t in query.tool_calls
                    ]
                messages.append(assistant_msg)

            for t in query.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(t.id),
                    "content": str(t.result)
                })
            
        return messages

    def invoke(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> Message:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        provider_dict = provider.to_dict() if provider else None
        
        response = self.client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            extra_body={"provider": provider_dict},
        )

        reply = Message(text=response.choices[0].message.content, role=Role.ai, raw_response=response)

        if response.choices[0].message.tool_calls:
            reply.role = Role.tool
            for tool in response.choices[0].message.tool_calls:
                reply.tool_calls.append(ToolCall(id=tool.id, name=tool.function.name, arguments=tool.function.arguments))
        return reply
        
    def invoke_stream(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> Iterator[ChatCompletionChunk]:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        provider_dict = provider.to_dict() if provider else None

        response = self.client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            extra_body={"provider": provider_dict},
            stream=True
        )
        
        return response

    async def async_invoke(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> Message:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        provider_dict = provider.to_dict() if provider else None

        response = await self.async_client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            extra_body={"provider": provider_dict}
        )

        reply = Message(text=response.choices[0].message.content, role=Role.ai, raw_response=response)

        if response.choices[0].message.tool_calls:
            reply.role = Role.tool
            for tool in response.choices[0].message.tool_calls:
                reply.tool_calls.append(ToolCall(id=tool.id, name=tool.function.name, arguments=tool.function.arguments))
        return reply
        
    async def async_invoke_stream(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        temperature: float = 0.3
    ) -> AsyncIterator[ChatCompletionChunk]:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        provider_dict = provider.to_dict() if provider else None

        response = await self.async_client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            extra_body={"provider": provider_dict},
            stream=True
        )

        async for chunk in response:
            yield chunk
        
    def structured_output(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        provider: ProviderConfig = None,
        json_schema: BaseModel = None,
        temperature: float = 0.3
    ) -> BaseModel:
        messages = self.make_prompt(system_prompt, querys)
        provider_dict = provider.to_dict() if provider else None
        
        response = self.client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": json_schema.__name__, "schema": json_schema.model_json_schema()}},
            extra_body={"provider": provider_dict},
        )

        return json_schema.model_validate_json(response.choices[0].message.content)
        
    
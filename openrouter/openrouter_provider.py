from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal, Iterator, AsyncIterator, TYPE_CHECKING

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ValidationError

from openrouter.message import Message, Role, _ToolCall
from openrouter.tool import tool_model

if TYPE_CHECKING:
    from openrouter.llms import LLMModel


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


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


@dataclass
class ReasoningConfig:
    effort: Optional[Literal["xhigh", "high", "medium", "low", "minimal", "none"]] = None
    max_tokens: Optional[int] = None
    exclude: Optional[bool] = None
    enabled: Optional[bool] = None

    def to_dict(self) -> dict:
        if self.enabled is False:
            reasoning = {"effort": "none"}
            if self.exclude is not None:
                reasoning["exclude"] = self.exclude
            return reasoning

        if self.effort is not None and self.max_tokens is not None:
            raise ValueError("ReasoningConfig cannot set both effort and max_tokens.")

        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_think(self) -> Optional[bool | Literal["high", "medium", "low"]]:
        if self.enabled is False or self.effort == "none":
            return False

        if self.effort in ["high", "medium", "low"]:
            return self.effort

        if self.effort == "minimal":
            return "low"

        if self.effort == "xhigh":
            return "high"

        if self.enabled is True or self.max_tokens is not None:
            return True

        return None


class _OpenRouterProvider:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> None:
        self.base_url = base_url
        api_key = self._resolve_api_key(api_key)
        timeout_options = self._make_timeout_options(timeout=timeout)
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            **timeout_options,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            **timeout_options,
        )

    def _resolve_api_key(self, api_key: Optional[str] = None) -> str:
        if api_key:
            return api_key

        if self.base_url != DEFAULT_BASE_URL:
            return "dummy"

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")
        
        return api_key

    def _make_extra_body(
        self,
        provider: Optional[ProviderConfig] = None,
        reasoning: Optional[ReasoningConfig] = None
    ) -> dict:
        extra_body = {}

        if provider:
            extra_body["provider"] = provider.to_dict()

        if reasoning:
            extra_body["reasoning"] = reasoning.to_dict()
            think = reasoning.to_think()
            if think is not None:
                extra_body["think"] = think

        if not extra_body:
            return {}

        return {"extra_body": extra_body}

    def _make_timeout_options(self, timeout: Optional[float] = None) -> dict:
        if timeout is None:
            return {}

        return {"timeout": timeout}

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

                if query.reasoning:
                    assistant_msg["reasoning"] = query.reasoning

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
        reasoning: ReasoningConfig = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None
    ) -> Message:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        extra_body = self._make_extra_body(provider=provider, reasoning=reasoning)
        timeout_options = self._make_timeout_options(timeout=timeout)
        
        response = self.client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            **extra_body,
            **timeout_options,
        )

        response_message = response.choices[0].message
        reply = Message(
            text=response_message.content,
            role=Role.ai,
            raw_response=response,
            reasoning=getattr(response_message, "reasoning", None),
        )

        if response_message.tool_calls:
            reply.role = Role.tool
            for tool in response_message.tool_calls:
                reply.tool_calls.append(_ToolCall(id=tool.id, name=tool.function.name, arguments=tool.function.arguments))
        return reply
    
    def invoke_stream(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        reasoning: ReasoningConfig = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None
    ) -> Iterator[ChatCompletionChunk]:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        extra_body = self._make_extra_body(provider=provider, reasoning=reasoning)
        timeout_options = self._make_timeout_options(timeout=timeout)

        response = self.client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            stream=True,
            **extra_body,
            **timeout_options,
        )
        
        return response

    async def async_invoke(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        reasoning: ReasoningConfig = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None
    ) -> Message:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        extra_body = self._make_extra_body(provider=provider, reasoning=reasoning)
        timeout_options = self._make_timeout_options(timeout=timeout)

        response = await self.async_client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            **extra_body,
            **timeout_options,
        )

        response_message = response.choices[0].message
        reply = Message(
            text=response_message.content,
            role=Role.ai,
            raw_response=response,
            reasoning=getattr(response_message, "reasoning", None),
        )

        if response_message.tool_calls:
            reply.role = Role.tool
            for tool in response_message.tool_calls:
                reply.tool_calls.append(_ToolCall(id=tool.id, name=tool.function.name, arguments=tool.function.arguments))
        return reply
        
    async def async_invoke_stream(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        tools: list[tool_model] = None,
        provider: ProviderConfig = None,
        reasoning: ReasoningConfig = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None
    ) -> AsyncIterator[ChatCompletionChunk]:
        tools = tools or []
        messages = self.make_prompt(system_prompt, querys)

        tool_defs = [tool.tool_definition for tool in tools] if tools else None
        extra_body = self._make_extra_body(provider=provider, reasoning=reasoning)
        timeout_options = self._make_timeout_options(timeout=timeout)

        response = await self.async_client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            tools=tool_defs,
            stream=True,
            **extra_body,
            **timeout_options,
        )

        async for chunk in response:
            yield chunk
        
    def structured_output(
        self,
        model: LLMModel,
        system_prompt: Message,
        querys: list[Message],
        provider: ProviderConfig = None,
        reasoning: ReasoningConfig = None,
        json_schema: BaseModel = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None
    ) -> BaseModel:
        messages = self.make_prompt(system_prompt, querys)
        extra_body = self._make_extra_body(provider=provider, reasoning=reasoning)
        timeout_options = self._make_timeout_options(timeout=timeout)
        
        schema = json_schema.model_json_schema()
        
        def add_additional_properties_false(obj):
            if isinstance(obj, dict):
                if "properties" in obj:
                    obj["additionalProperties"] = False
                for value in obj.values():
                    add_additional_properties_false(value)
            elif isinstance(obj, list):
                for item in obj:
                    add_additional_properties_false(item)
        
        add_additional_properties_false(schema)

        response = self.client.chat.completions.create(
            model=model.name,
            temperature=temperature,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": json_schema.__name__, "schema": schema}},
            **extra_body,
            **timeout_options,
        )

        content = response.choices[0].message.content

        try:
            return json_schema.model_validate_json(content)
        except ValidationError:
            formatted_content = content
            try:
                parsed = json.loads(content)
                formatted_content = json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
            print("structured_output validation failed, response content:")
            print(formatted_content)
            raise

    

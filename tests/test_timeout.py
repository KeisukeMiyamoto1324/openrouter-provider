import asyncio
import unittest
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import Mock, patch

from pydantic import BaseModel

from openrouter.llms import LLMModel
from openrouter.message import Message, Role
from openrouter.openrouter import OpenRouterClient
from openrouter.openrouter_provider import _OpenRouterProvider


class TimeoutSchema(BaseModel):
    answer: str


class TimeoutTest(unittest.TestCase):
    def _response(self, content: str = "answer") -> SimpleNamespace:
        response_message = SimpleNamespace(
            content=content,
            reasoning=None,
            tool_calls=None,
        )

        return SimpleNamespace(choices=[SimpleNamespace(message=response_message)])

    def test_provider_invoke_passes_timeout_to_request(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        create = Mock(return_value=self._response())
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        client.invoke(
            model=LLMModel(name="dummy"),
            system_prompt=Message(text="system", role=Role.system),
            querys=[Message(text="question")],
            timeout=10,
        )

        self.assertEqual(create.call_args.kwargs["timeout"], 10)

    def test_provider_invoke_omits_none_timeout(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        create = Mock(return_value=self._response())
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        client.invoke(
            model=LLMModel(name="dummy"),
            system_prompt=Message(text="system", role=Role.system),
            querys=[Message(text="question")],
        )

        self.assertNotIn("timeout", create.call_args.kwargs)

    def test_client_default_timeout_is_forwarded(self) -> None:
        with patch("openrouter.openrouter._OpenRouterProvider") as provider:
            provider.return_value.invoke.return_value = Message(text="answer", role=Role.ai)

            client = OpenRouterClient(base_url="http://localhost:11434/v1", timeout=30)
            client.invoke(model=LLMModel(name="dummy"), query=Message(text="question"))

            provider.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key=None,
                timeout=30,
            )
            self.assertEqual(provider.return_value.invoke.call_args.kwargs["timeout"], 30)

    def test_client_call_timeout_overrides_default_timeout(self) -> None:
        with patch("openrouter.openrouter._OpenRouterProvider") as provider:
            provider.return_value.invoke.return_value = Message(text="answer", role=Role.ai)

            client = OpenRouterClient(base_url="http://localhost:11434/v1", timeout=30)
            client.invoke(model=LLMModel(name="dummy"), query=Message(text="question"), timeout=5)

            provider.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key=None,
                timeout=5,
            )
            self.assertEqual(provider.return_value.invoke.call_args.kwargs["timeout"], 5)

    def test_provider_invoke_stream_passes_timeout_to_request(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        create = Mock(return_value=iter([]))
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        client.invoke_stream(
            model=LLMModel(name="dummy"),
            system_prompt=Message(text="system", role=Role.system),
            querys=[Message(text="question")],
            timeout=10,
        )

        self.assertEqual(create.call_args.kwargs["timeout"], 10)

    def test_provider_structured_output_passes_timeout_to_request(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        create = Mock(return_value=self._response('{"answer": "ok"}'))
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        client.structured_output(
            model=LLMModel(name="dummy"),
            system_prompt=Message(text="system", role=Role.system),
            querys=[Message(text="question")],
            json_schema=TimeoutSchema,
            timeout=10,
        )

        self.assertEqual(create.call_args.kwargs["timeout"], 10)

    def test_provider_async_invoke_passes_timeout_to_request(self) -> None:
        async def run_test() -> None:
            client = _OpenRouterProvider(base_url="http://localhost:11434/v1")

            async def create(**kwargs: object) -> SimpleNamespace:
                create.kwargs = kwargs
                return self._response()

            client.async_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )

            await client.async_invoke(
                model=LLMModel(name="dummy"),
                system_prompt=Message(text="system", role=Role.system),
                querys=[Message(text="question")],
                timeout=10,
            )

            self.assertEqual(create.kwargs["timeout"], 10)

        asyncio.run(run_test())

    def test_provider_async_invoke_stream_passes_timeout_to_request(self) -> None:
        async def run_test() -> None:
            client = _OpenRouterProvider(base_url="http://localhost:11434/v1")

            async def empty_stream() -> AsyncIterator[None]:
                if False:
                    yield None

            async def create(**kwargs: object) -> AsyncIterator[None]:
                create.kwargs = kwargs
                return empty_stream()

            client.async_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )

            stream = client.async_invoke_stream(
                model=LLMModel(name="dummy"),
                system_prompt=Message(text="system", role=Role.system),
                querys=[Message(text="question")],
                timeout=10,
            )

            async for _ in stream:
                pass

            self.assertEqual(create.kwargs["timeout"], 10)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

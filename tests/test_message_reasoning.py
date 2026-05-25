import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from openrouter.llms import LLMModel
from openrouter.message import Message, Role
from openrouter.openrouter_provider import _OpenRouterProvider


class MessageReasoningTest(unittest.TestCase):
    def test_message_stores_reasoning_separately_from_text(self) -> None:
        message = Message(text="answer", reasoning="thinking")

        self.assertEqual(message.text, "answer")
        self.assertEqual(message.reasoning, "thinking")

    def test_make_prompt_preserves_assistant_reasoning(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        system_prompt = Message(text="system", role=Role.system)
        assistant_message = Message(text="answer", role=Role.ai, reasoning="thinking")

        messages = client.make_prompt(system_prompt, [assistant_message])

        self.assertEqual(messages[1]["content"], "answer")
        self.assertEqual(messages[1]["reasoning"], "thinking")

    def test_invoke_maps_response_reasoning_to_message(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        response_message = SimpleNamespace(
            content="answer",
            reasoning="thinking",
            tool_calls=None,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=response_message)]
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response)))
        )

        reply = client.invoke(
            model=LLMModel(name="dummy"),
            system_prompt=Message(text="system", role=Role.system),
            querys=[Message(text="question")],
        )

        self.assertEqual(reply.text, "answer")
        self.assertEqual(reply.reasoning, "thinking")


if __name__ == "__main__":
    unittest.main()

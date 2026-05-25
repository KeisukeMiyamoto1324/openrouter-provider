import unittest

from openrouter.openrouter_provider import ProviderConfig, ReasoningConfig, _OpenRouterProvider


class ReasoningConfigTest(unittest.TestCase):
    def test_to_dict_removes_none_values(self) -> None:
        reasoning = ReasoningConfig(effort="high", exclude=True)

        self.assertEqual(reasoning.to_dict(), {"effort": "high", "exclude": True})

    def test_to_dict_rejects_effort_and_max_tokens_together(self) -> None:
        reasoning = ReasoningConfig(effort="high", max_tokens=2000)

        with self.assertRaises(ValueError):
            reasoning.to_dict()

    def test_make_extra_body_merges_provider_and_reasoning(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        provider = ProviderConfig(order=["cerebras"], allow_fallbacks=False)
        reasoning = ReasoningConfig(max_tokens=2000, exclude=True)

        extra_body = client._make_extra_body(provider=provider, reasoning=reasoning)

        self.assertEqual(
            extra_body,
            {
                "extra_body": {
                    "provider": {"order": ["cerebras"], "allow_fallbacks": False},
                    "reasoning": {"max_tokens": 2000, "exclude": True},
                }
            },
        )


if __name__ == "__main__":
    unittest.main()

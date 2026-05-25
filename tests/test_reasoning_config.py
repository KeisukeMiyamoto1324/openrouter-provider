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

    def test_to_dict_normalizes_disabled_reasoning_to_effort_none(self) -> None:
        reasoning = ReasoningConfig(max_tokens=2000, exclude=False, enabled=False)

        self.assertEqual(reasoning.to_dict(), {"effort": "none", "exclude": False})

    def test_to_think_maps_reasoning_values(self) -> None:
        self.assertFalse(ReasoningConfig(enabled=False).to_think())
        self.assertEqual(ReasoningConfig(effort="xhigh").to_think(), "high")
        self.assertEqual(ReasoningConfig(effort="minimal").to_think(), "low")
        self.assertTrue(ReasoningConfig(max_tokens=2000).to_think())

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
                    "think": True,
                }
            },
        )

    def test_make_extra_body_sends_reasoning_and_think_for_disabled_reasoning(self) -> None:
        client = _OpenRouterProvider(base_url="http://localhost:11434/v1")
        reasoning = ReasoningConfig(enabled=False)

        extra_body = client._make_extra_body(reasoning=reasoning)

        self.assertEqual(
            extra_body,
            {"extra_body": {"reasoning": {"effort": "none"}, "think": False}},
        )


if __name__ == "__main__":
    unittest.main()

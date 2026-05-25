# python3 -m tests.ollama

from openrouter.openrouter import *


ai = OpenRouterClient(
    system_prompt="Please answer in English.",
    base_url="http://localhost:11434/v1",
)
query = Message(text="Introduce yourself briefly. Who are you?")
model = LLMModel(name="qwen3.5:4b")
reasoning = ReasoningConfig(enabled=True)

response = ai.invoke(model=model, query=query, reasoning=reasoning)

print("CONTENT:", response.text)
print("REASONING:", response.reasoning)

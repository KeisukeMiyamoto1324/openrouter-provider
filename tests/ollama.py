# python3 -m tests.ollama

from openrouter.openrouter import *


ai = OpenRouterClient(
    system_prompt="Please answer in English.",
    base_url="http://localhost:11434/v1",
)
query = Message(text="Introduce yourself briefly. Who are you?")
model = LLMModel(name="gemma4:e4b-mlx")

response = ai.invoke(model=model, query=query)
print(response.text)


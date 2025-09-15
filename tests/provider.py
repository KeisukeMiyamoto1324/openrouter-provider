# python3 -m tests.provider

from openrouter.openrouter import *

ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="Introduce yourself, please.")
provider = ProviderConfig(order=["cerebras"], allow_fallbacks=False)
response = ai.invoke(model=llama_4_scout, query=query, provider=provider)
ai.print_memory()

# invalid_model = LLMModel(name="dummy")
# response = ai.invoke(model=invalid_model, query=query)
# ai.print_memory()
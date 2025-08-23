# python3 -m tests.provider

from src.OpenRouterProvider.Chatbot_manager import *

ai = Chatbot_manager(system_prompt="Please answer in English.")
query = Chat_message(text="Introduce yourself, please.")
provider = ProviderConfig(order=["cerebras"], allow_fallbacks=False)
response = ai.invoke(model=llama_4_scout, query=query, provider=provider)
ai.print_memory()

# invalid_model = LLMModel(name="dummy")
# response = ai.invoke(model=invalid_model, query=query)
# ai.print_memory()
# python3 -m tests.basic

from src.openrouter import *

ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="Introduce yourself, please.")
response = ai.invoke(model=claude_3_7_sonnet, query=query)
ai.print_memory()


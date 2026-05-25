# python3 -m tests.reasoning

from openrouter.openrouter import *


ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="Which is larger, 9.11 or 9.9? Explain briefly.")
reasoning = ReasoningConfig(max_tokens=512, exclude=False, enabled=False)

response = ai.invoke(
    model=gemini_3_flash_preview,
    query=query,
    reasoning=reasoning,
    temperature=0.1,
)

print("MODEL:", gemini_3_flash_preview.name)
print("CONTENT:", response.text)
print("REASONING:", response.reasoning)

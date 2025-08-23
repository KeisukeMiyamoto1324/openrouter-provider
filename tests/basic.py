# python3 -m tests.basic

from src.Chatbot_manager import *


ai = Chatbot_manager(system_prompt="Please answer in English.")
query = Chat_message(text="Introduce yourself, please.")
response = ai.invoke(model=claude_3_7_sonnet, query=query)
ai.print_memory()


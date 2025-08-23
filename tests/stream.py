# python3 -m tests.stream

from src.Chatbot_manager import *


ai = Chatbot_manager(system_prompt="Please answer in English.")
query = Chat_message(text="Tell me the history of aviation briefly.")
response = ai.invoke_stream(model=gpt_4o_mini, query=query)

for token in response:
    print(token, end="")

ai.print_memory()
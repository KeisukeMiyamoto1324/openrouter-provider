# python3 -m tests.tool

from src.Chatbot_manager import *
from src.Tool import tool_model


@tool_model
def user_info():
    """
    Return user's personal info, such as name, age and address.
    """
    
    name = "Satoshi Tanaka"
    age = "43"
    address = "Italy"
    
    return f"name: {name}\nage: {age}\naddress: {address}"

ai = Chatbot_manager(system_prompt="Please answer in English.", tools=[user_info])
query = Chat_message(text="What is the name, age, address of the user?")
response = ai.invoke(model=gpt_4o_mini, query=query)
ai.print_memory()



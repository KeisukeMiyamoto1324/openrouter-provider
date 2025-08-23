# python3 -m tests.async

from src.openrouter import *
from src.tool import tool_model

import asyncio


@tool_model
def user_info():
    """
    Return user's personal info, such as name, age and address.
    """
    
    name = "Satoshi Tanaka"
    age = "43"
    address = "Italy"
    
    return f"name: {name}\nage: {age}\naddress: {address}"

async def test_async_invoke_stream():
    ai = OpenRouterClient(system_prompt="Please answer in English.")
    query = Message(text="Tell me three interesting facts about space.")
    
    print("\n--- Streaming Response ---")
    async for token in ai.async_invoke_stream(model=gpt_4o_mini, query=query):
        print(token, end="", flush=True)
    print()  # 改行

async def main():
    ai1 = OpenRouterClient(system_prompt="Please answer in English.", tools=[user_info])
    ai2 = OpenRouterClient(system_prompt="Please answer in English.", tools=[user_info])
    ai3 = OpenRouterClient(system_prompt="Please answer in English.", tools=[user_info])
    
    query1 = Message(text="What is the user's name?")
    query2 = Message(text="What is the user's age?")
    query3 = Message(text="What is the user's address?")

    responses = await asyncio.gather(
        # ai1.async_invoke(model=gpt_4o_mini, query=query1),
        # ai2.async_invoke(model=gpt_4o_mini, query=query2),
        # ai3.async_invoke(model=gpt_4o_mini, query=query3),
        test_async_invoke_stream()  
    )

    # for i, r in enumerate(responses[:-1], start=1):
    #     print(f"\n--- Response {i} ---")
    #     print(r.text)



if __name__ == "__main__":
    asyncio.run(main())
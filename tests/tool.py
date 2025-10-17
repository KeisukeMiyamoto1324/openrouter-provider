# python3 -m tests.tool

from openrouter.openrouter import *


@tool_model
def get_weather(location: str) -> str:
    return f"{location}: Sunny, 25°C"


print("=== Auto Tool Execution (auto_tool_exec=True) ===")
ai = OpenRouterClient(tools=[get_weather])
response = ai.invoke(model=gpt_4o_mini, query=Message("What's the weather in Tokyo?"), auto_tool_exec=True)
print(f"Response: {response.text}\n")


print("=== Manual Tool Execution (auto_tool_exec=False) ===")
ai2 = OpenRouterClient(tools=[get_weather])
response = ai2.invoke(model=gpt_4o_mini, query=Message("What's the weather in Paris?"), auto_tool_exec=False)

if response.tool_calls:
    print(f"Tool requested: {response.tool_calls[0].name}")
    print(f"Arguments: {response.tool_calls[0].arguments}")

    response_with_result = ai2.execute_tool(response, tool_index=0)
    print(f"Tool result: {response_with_result.tool_calls[0].result}")

    final_response = ai2.invoke(model=gpt_4o_mini, auto_tool_exec=False)
    print(f"Final response: {final_response.text}")
else:
    print("No tools called")



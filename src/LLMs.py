from dataclasses import dataclass

@dataclass
class LLMModel:
    name: str
    input_cost: float = 0
    output_cost: float = 0


# OpenAI
gpt_4o = LLMModel(name='openai/gpt-4o', input_cost=2.5, output_cost=10.0)
gpt_4o_mini = LLMModel(name='openai/gpt-4o-mini', input_cost=0.15, output_cost=0.60)
gpt_4_1 = LLMModel(name='openai/gpt-4.1', input_cost=2, output_cost=8.0)
gpt_4_1_mini = LLMModel(name='openai/gpt-4.1-mini', input_cost=0.4, output_cost=1.6)
gpt_4_1_nano = LLMModel(name='openai/gpt-4.1-nano', input_cost=0.1, output_cost=0.4)
o4_mini = LLMModel(name='openai/o4-mini', input_cost=1.1, output_cost=4.4)
o4_mini_high = LLMModel(name='openai/o4-mini-high', input_cost=1.1, output_cost=4.4)
o3 = LLMModel(name='openai/o3', input_cost=10, output_cost=40)

# Anthropic
claude_3_7_sonnet = LLMModel(name='anthropic/claude-3.7-sonnet', input_cost=3.0, output_cost=15.0)
claude_3_7_sonnet_thinking = LLMModel(name='anthropic/claude-3.7-sonnet:thinking', input_cost=3.0, output_cost=15.0)
claude_3_5_haiku = LLMModel(name='anthropic/claude-3.5-haiku', input_cost=0.8, output_cost=4.0)

# Google
gemini_2_0_flash = LLMModel(name='google/gemini-2.0-flash-001', input_cost=0.1, output_cost=0.4)
gemini_2_0_flash_free = LLMModel(name='google/gemini-2.0-flash-exp:free', input_cost=0.1, output_cost=0.4)
gemini_2_5_flash = LLMModel(name='google/gemini-2.5-flash-preview', input_cost=0.15, output_cost=0.60)
gemini_2_5_flash_thinking = LLMModel(name='google/gemini-2.5-flash-preview:thinking', input_cost=0.15, output_cost=3.5)
gemini_2_5_pro = LLMModel(name='google/gemini-2.5-pro-preview-03-25', input_cost=1.25, output_cost=10)

# Deepseek
deepseek_v3_free = LLMModel(name='deepseek/deepseek-chat-v3-0324:free', input_cost=0, output_cost=0)
deepseek_v3 = LLMModel(name='deepseek/deepseek-chat-v3-0324', input_cost=0.3, output_cost=1.2)
deepseek_r1_free = LLMModel(name='deepseek/deepseek-r1:free', input_cost=0, output_cost=0)
deepseek_r1 = LLMModel(name='deepseek/deepseek-r1', input_cost=0.5, output_cost=2.2)

# xAI
grok_3_mini = LLMModel(name='x-ai/grok-3-mini-beta', input_cost=0.3, output_cost=0.5)
grok_3 = LLMModel(name='x-ai/grok-3-beta', input_cost=3, output_cost=15)

# Microsoft
mai_ds_r1_free = LLMModel(name="microsoft/mai-ds-r1:free", input_cost=0, output_cost=0)

# Others
llama_4_maverick_free = LLMModel(name="meta-llama/llama-4-maverick:free", input_cost=0, output_cost=0)
llama_4_scout = LLMModel(name="meta-llama/llama-4-scout", input_cost=0.11, output_cost=0.34)
mistral_small_3_1_24B_free = LLMModel(name="mistralai/mistral-small-3.1-24b-instruct:free", input_cost=0, output_cost=0)
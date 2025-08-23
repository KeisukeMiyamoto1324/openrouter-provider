## Introduction

Welcome to **openrouter-provider**, an unofficial Python wrapper for the OpenRouter API. This library lets you easily integrate with OpenRouter models, manage chat sessions, process images, and call tools within your Python application.


## Features

* Simple chat interface with system, user, assistant, and tool roles
* Automatic image resizing and Base64 encoding
* Built-in tool decorator for defining custom functions


## Installation

### From PyPI

```bash
pip3 install openrouter-provider
```

### From Source

```bash
git clone https://github.com/yourusername/openrouter-provider.git
cd openrouter-provider
pip3 install .
```



## Configuration

1. Create a `.env` file in your project root.
2. Add your OpenRouter API key:

   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```



## Usage

**Note**: The examples below show imports for development/source usage. When using the installed package from PyPI, replace `src.` with `openrouter_provider.` in all import statements.

### Basic chat bot
Chat history is automatically sent, by OpenRouterClient. If you want to delete chat history, use `clear_memory` method.

```python
from src.OpenRouterClient import Message, OpenRouterClient
from src.LLMs import gpt_4o_mini

# Declare chat bot
ai = OpenRouterClient(system_prompt="Please answer in English.")

# Send query
query = Message(text="Introduce yourself, please.")
response = ai.invoke(model=gpt_4o_mini, query=query)
print(response.text)

# Send next query. OpenRouterClient automatically handle chat history.
query = Message(text="Tell me a short story.")
response = ai.invoke(model=gpt_4o_mini, query=query)
print(response.text)

# Print all chat history
ai.print_memory()  

# Delete all chat history
ai.clear_memory()
```

### Chat bot with images
You can use images in the chat.

```python
from src.OpenRouterClient import Message, OpenRouterClient
from src.LLMs import gpt_4o_mini
from PIL import Image

dog = Image.open("dog.jpg")
cat = Image.open("cat.jpg")

# Send query with images
ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="What can you see in the images?", images=[dog, cat])
response = ai.invoke(model=gpt_4o_mini, query=query)
print(response.text) 
```

### With tools

Use the `@tool_model` decorator to expose Python functions as callable tools in the chat. Tools are automatically processed by Chat_manager, so you don't need to care it.

```python
from src.OpenRouterClient import Message, OpenRouterClient
from src.LLMs import gpt_4o_mini
from src.Tool import tool_model

@tool_model
def get_user_info():
    """
    Return user's personal info: name, age, and address.
    """
    return "name: Alice\nage: 30\naddress: Wonderland"

ai = OpenRouterClient(system_prompt="Please answer in English.", tools=[get_user_info])
query = Message(text="What is the name, age, address of the user?")
response = ai.invoke(model=gpt_4o_mini, query=query)
ai.print_memory()
```

## Advanced Usage
### Prebuilt and Custom Model Usage

You can use prebuilt models defined or declare your own custom models easily.
This library provides many ready-to-use models from OpenAI, Anthropic, Google, and others.

```python
from src.OpenRouterClient import Message, OpenRouterClient
from src.LLMs import gpt_4o, claude_3_7_sonnet

# Use OpenAI GPT-4o
ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="Tell me a joke.")
response = ai.invoke(model=gpt_4o, query=query)
print(response.text)

# Use Anthropic Claude 3.7 Sonnet
query = Message(text="Summarize the story of Hamlet.")
response = ai.invoke(model=claude_3_7_sonnet, query=query)
print(response.text)
```

Available prebuilt models include:

#### **OpenAI**

* `gpt_4o`
* `gpt_4o_mini`
* `gpt_4_1`
* `gpt_4_1_mini`
* `gpt_4_1_nano`
* `o4_mini`
* `o4_mini_high`
* `o3`

#### **Anthropic**

* `claude_3_7_sonnet`
* `claude_3_7_sonnet_thinking`
* `claude_3_5_haiku`

#### **Google**

* `gemini_2_0_flash`
* `gemini_2_0_flash_free`
* `gemini_2_5_flash`
* `gemini_2_5_flash_thinking`
* `gemini_2_5_pro`

#### **Deepseek**

* `deepseek_v3_free`
* `deepseek_v3`
* `deepseek_r1_free`
* `deepseek_r1`

#### **xAI**

* `grok_3_mini`
* `grok_3`

#### **Microsoft**

* `mai_ds_r1_free`

#### **Others**

* `llama_4_maverick_free`
* `llama_4_scout`
* `mistral_small_3_1_24B_free`

All of them are instances of `LLMModel`, which includes cost and model name settings.

### Using Custom Models

You can define and use your own custom model if it's available on OpenRouter.

```python
from src.OpenRouterClient import Message, OpenRouterClient
from src.LLMs import LLMModel

# Declare a custom model
my_model = LLMModel(
    name="my-org/my-custom-model",  # Model name for OpenRouter
    input_cost=0.5,                 # Optional: cost per 1M input tokens
    output_cost=2.0                 # Optional: cost per 1M output tokens
)

# Use the custom model
ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="Explain black holes simply.")
response = ai.invoke(model=my_model, query=query)
print(response.text)
```

You only need to know the model name as used on OpenRouter. `input_cost` and `output_cost` are optional and currently, they are not used in this library. Please wait the future update.

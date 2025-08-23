# OpenRouter Provider

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/openrouter-provider.svg)](https://badge.fury.io/py/openrouter-provider)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An unofficial Python wrapper for the OpenRouter API that provides a simple, intuitive interface for interacting with multiple LLM models. OpenRouter Provider supports chat conversations, image processing, tool integration, streaming responses, and structured output generation.

## Features

- **Multi-Model Support**: Access 40+ models from OpenAI, Anthropic, Google, DeepSeek, xAI, Microsoft, and Meta
- **Conversation Memory**: Automatic chat history management with easy memory control
- **Image Processing**: Built-in image resizing and base64 encoding for multimodal interactions
- **Tool Integration**: Decorator-based function calling with automatic tool execution
- **Streaming Support**: Real-time response streaming for both sync and async operations
- **Structured Output**: JSON schema-based response formatting using Pydantic models
- **Async Support**: Full async/await support for non-blocking operations
- **Provider Configuration**: OpenRouter-specific routing and fallback options

## Installation

### From PyPI (Recommended)

```bash
pip install openrouter-provider
```

### From Source

```bash
git clone https://github.com/yourusername/openrouter-provider.git
cd openrouter-provider
pip install .
```

## =' Configuration

1. Get your API key from [OpenRouter](https://openrouter.ai/)
2. Set up your environment:

**Option 1: Environment Variable**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

**Option 2: .env File**
```bash
# Create .env file in your project root
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

## <� Quick Start

```python
from openrouter import *

# Create client
ai = OpenRouterClient(system_prompt="You are a helpful assistant.")

# Send a message
query = Message(text="What's the capital of France?")
response = ai.invoke(model=gpt_4o_mini, query=query)
print(response.text)
```

## Usage Examples

### Basic Chat Conversation

```python
from openrouter import *

# Initialize client with system prompt
ai = OpenRouterClient(system_prompt="You are a friendly coding assistant.")

# First message
query = Message(text="Explain what Python is in simple terms.")
response = ai.invoke(model=claude_3_7_sonnet, query=query)
print(response.text)

# Follow-up message (conversation history is automatically maintained)
query = Message(text="Give me a simple Python example.")
response = ai.invoke(model=claude_3_7_sonnet, query=query)
print(response.text)

# View conversation history
ai.print_memory()

# Clear conversation history
ai.clear_memory()
```

### Image Processing

```python
from openrouter import *
from PIL import Image

# Load images
dog_image = Image.open("dog.jpg")
cat_image = Image.open("cat.jpg")

# Create client
ai = OpenRouterClient(system_prompt="You are an image analysis expert.")

# Send message with images
query = Message(
    text="Compare these two animals. What are the key differences?",
    images=[dog_image, cat_image]
)
response = ai.invoke(model=gpt_4o, query=query)
print(response.text)
```

### Tool Integration

```python
from openrouter import *

@tool_model
def get_weather(city: str, country: str = "US") -> str:
    """
    Get current weather information for a specific city.
    
    Args:
        city: Name of the city
        country: Country code (default: US)
    """
    # In real implementation, you'd call a weather API
    return f"The weather in {city}, {country} is sunny with 22�C"

@tool_model
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
    """
    Calculate tip amount and total bill.
    
    Args:
        bill_amount: The original bill amount
        tip_percentage: Tip percentage (default: 15%)
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return f"Tip: ${tip:.2f}, Total: ${total:.2f}"

# Create client with tools
ai = OpenRouterClient(
    system_prompt="You are a helpful assistant with access to weather and calculator tools.",
    tools=[get_weather, calculate_tip]
)

# The AI will automatically use tools when needed
query = Message(text="What's the weather in Tokyo and calculate a 20% tip on a $50 bill?")
response = ai.invoke(model=gpt_4o_mini, query=query)
print(response.text)
```

### Streaming Responses

```python
from openrouter import *

ai = OpenRouterClient(system_prompt="You are a storyteller.")
query = Message(text="Tell me a short story about a magical forest.")

# Stream the response
for token in ai.invoke_stream(model=claude_3_7_sonnet, query=query):
    print(token, end="", flush=True)
```

### Async Operations

```python
import asyncio
from openrouter import *

async def main():
    ai = OpenRouterClient(system_prompt="You are a helpful assistant.")
    
    # Async invoke
    query = Message(text="Explain quantum computing in simple terms.")
    response = await ai.async_invoke(model=gpt_4o_mini, query=query)
    print(response.text)
    
    # Async streaming
    query = Message(text="Write a poem about the ocean.")
    async for token in ai.async_invoke_stream(model=gpt_4o_mini, query=query):
        print(token, end="", flush=True)

# Run async function
asyncio.run(main())
```

### Structured Output

```python
from openrouter import *
from pydantic import BaseModel, Field
from typing import List

class BookRecommendation(BaseModel):
    title: str = Field(description="Title of the book")
    author: str = Field(description="Author of the book")
    genre: str = Field(description="Genre of the book")
    rating: float = Field(description="Rating out of 5.0")
    summary: str = Field(description="Brief summary of the book")

class BookList(BaseModel):
    recommendations: List[BookRecommendation] = Field(description="List of book recommendations")
    total_count: int = Field(description="Total number of recommendations")

ai = OpenRouterClient(system_prompt="You are a book recommendation expert.")
query = Message(text="Recommend 3 science fiction books for beginners.")

# Get structured output
result: BookList = ai.structured_output(
    model=gpt_4o_mini,
    query=query,
    json_schema=BookList
)

print(f"Found {result.total_count} recommendations:")
for book in result.recommendations:
    print(f"- {book.title} by {book.author} ({book.rating}/5)")
```

## Available Models

The library provides pre-configured models from major providers:

### OpenAI
- `gpt_4o` - GPT-4 Omni
- `gpt_4o_mini` - GPT-4 Omni Mini
- `o3` - OpenAI o3 Reasoning Model

### Anthropic
- `claude_3_7_sonnet` - Claude 3.7 Sonnet
- `claude_3_5_haiku` - Claude 3.5 Haiku

### Google
- `gemini_2_0_flash` - Gemini 2.0 Flash
- `gemini_2_5_pro` - Gemini 2.5 Pro

### DeepSeek
- `deepseek_v3` - DeepSeek V3
- `deepseek_r1` - DeepSeek R1

### Others
- `grok_3` - xAI Grok 3
- `llama_4_scout` - Meta Llama 4 Scout

### Custom Models

You can also use any model available on OpenRouter:

```python
from openrouter import *

# Define custom model
custom_model = LLMModel(
    name="anthropic/claude-3-haiku",
    input_cost=0.25,  # Optional: cost per 1M input tokens
    output_cost=1.25  # Optional: cost per 1M output tokens
)

# Use custom model
response = ai.invoke(model=custom_model, query=query)
```

## Advanced Configuration

### Provider Configuration

```python
from openrouter import *

# Configure provider preferences
provider_config = ProviderConfig(
    order=["OpenAI", "Anthropic"],  # Preferred provider order
    allow_fallbacks=True,           # Allow fallback providers
    data_collection="deny"          # Opt out of data collection
)

response = ai.invoke(
    model=gpt_4o_mini,
    query=query,
    provider=provider_config
)
```

### Temperature Control

```python
# More creative responses
response = ai.invoke(
    model=claude_3_7_sonnet,
    query=query,
    temperature=0.9
)

# More deterministic responses
response = ai.invoke(
    model=claude_3_7_sonnet,
    query=query,
    temperature=0.1
)
```

## API Reference

### OpenRouterClient

```python
class OpenRouterClient:
    def __init__(self, system_prompt: str = "", tools: List[tool_model] = None)
    
    def invoke(self, model: LLMModel, query: Message, tools: List[tool_model] = None, 
              provider: ProviderConfig = None, temperature: float = 0.3) -> Message
    
    def invoke_stream(self, model: LLMModel, query: Message, tools: List[tool_model] = None,
                     provider: ProviderConfig = None, temperature: float = 0.3) -> Iterator[str]
    
    async def async_invoke(self, model: LLMModel, query: Message, tools: List[tool_model] = None,
                          provider: ProviderConfig = None, temperature: float = 0.3) -> Message
    
    async def async_invoke_stream(self, model: LLMModel, query: Message, tools: List[tool_model] = None,
                                 provider: ProviderConfig = None, temperature: float = 0.3) -> AsyncIterator[str]
    
    def structured_output(self, model: LLMModel, query: Message, provider: ProviderConfig = None,
                         json_schema: BaseModel = None, temperature: float = 0.3) -> BaseModel
    
    def clear_memory(self) -> None
    def print_memory(self) -> None
    def set_system_prompt(self, prompt: str) -> None
```

### Message

```python
class Message:
    def __init__(self, text: str, images: Optional[List[Image.Image]] = None, 
                role: Role = Role.user, answered_by: Optional[LLMModel] = None)
```

### Tool Decorator

```python
@tool_model
def your_function(param1: str, param2: int = 10) -> str:
    """Function description for the AI."""
    return "result"
```

## Development

### Running Tests

```bash
# Basic functionality
python -m tests.basic

# Image processing
python -m tests.image

# Tool integration
python -m tests.tool

# Streaming
python -m tests.stream

# Async operations
python -m tests.async

# Structured output
python -m tests.structured_output
```

### Building from Source

```bash
pip install build
python -m build
```

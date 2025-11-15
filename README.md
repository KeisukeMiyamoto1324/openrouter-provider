# OpenRouter Provider

An unofficial Python wrapper for the OpenRouter API that provides a simple, intuitive interface for interacting with multiple LLM models. OpenRouter Provider supports chat conversations, image processing, tool integration, streaming responses, and structured output generation.

## Features

- **Conversation Memory**: Automatic chat history management with easy memory control
- **Image Processing**: Built-in image resizing and base64 encoding for multimodal interactions
- **Tool Integration**: Decorator-based function calling with automatic tool execution
- **Streaming Support**: Real-time response streaming for both sync and async operations
- **Structured Output**: JSON schema-based response formatting using Pydantic models
- **Async Support**: Full async/await support for non-blocking operations
- **Provider Configuration**: You can select your favorite provoder easily

## Quick start

### Installation

```bash
pip install openrouter-provider
```

### Set API key

Please get your API key from [OpenRouter](https://openrouter.ai/) and set it as environment valuable.

```bash
OPENROUTER_API_KEY="your-api-key-here"
```

## Basic Usage
```python
from openrouter import *
from dotenv import load_dotenv

load_dotenv()

client = OpenRouterClient(system_prompt="You are a friendly AI assistant.")
model = LLMModel(name="openai/gpt-5")
query = Message(text="Hello, how are you?")

reply = client.invoke(model=model, query=query)
print(reply.text)
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an unofficial Python wrapper for the OpenRouter API that provides a simple interface for interacting with various LLM models through OpenRouter. The library supports chat management, image processing, tool calling, streaming responses, async operations, and structured output.

## Development Commands

### Setup and Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install from source
pip install .
```

### Running Tests
Tests are example scripts rather than formal test suites. Run them individually:
```bash
# Basic chat functionality
python3 -m tests.basic

# Async operations
python3 -m tests.async

# Image processing
python3 -m tests.image

# Tool calling
python3 -m tests.tool

# Streaming responses
python3 -m tests.stream

# Provider configuration
python3 -m tests.provider

# Structured output
python3 -m tests.structured_output
```

### Building and Distribution
```bash
# Build package
python3 -m build

# Upload to PyPI (requires credentials)
python3 -m twine upload dist/*
```

## Architecture

### Core Components

1. **OpenRouterProvider** (`src/OpenRouterProvider.py`): Main client class that handles API communication
   - Synchronous and asynchronous methods
   - Streaming support
   - Structured output generation
   - Provider configuration management

2. **OpenRouterClient** (`src/OpenRouterClient.py`): High-level chat management interface
   - Automatic conversation history management
   - System prompt handling with timestamps
   - Tool execution integration
   - Memory management (clear_memory, print_memory)

3. **Message** (`src/Message.py`): Message representation with role-based typing
   - Supports text and image content
   - Automatic image resizing and Base64 encoding
   - Tool call handling
   - RGBA to RGB conversion for images

4. **LLMs** (`src/LLMs.py`): Predefined model definitions with cost information
   - OpenAI models (GPT-4o, GPT-4o-mini, O4, etc.)
   - Anthropic models (Claude 3.7 Sonnet, Claude 3.5 Haiku)
   - Google models (Gemini 2.0/2.5 variants)
   - Deepseek, xAI Grok, Microsoft, and other providers

5. **Tool** (`src/Tool.py`): Decorator for creating callable tools
   - Automatic JSON schema generation from function signatures
   - Type mapping for parameters
   - Support for lists and primitive types

### Key Patterns

- **Environment Configuration**: Uses `.env` file for `OPENROUTER_API_KEY`
- **Role-based Messaging**: Uses enum for message roles (system, user, assistant, tool)
- **Image Processing**: Automatic resizing to 1024px max dimension with aspect ratio preservation
- **Tool Integration**: Automatic tool execution and result injection into conversation flow
- **Cost Tracking**: Model definitions include input/output cost per 1M tokens

### API Methods

**Synchronous**:
- `invoke()` - Single response
- `invoke_stream()` - Streaming response  
- `structured_output()` - JSON schema-based responses

**Asynchronous**:
- `async_invoke()` - Single async response
- `async_invoke_stream()` - Streaming async response

### Configuration Options

- **ProviderConfig**: Controls model selection behavior
  - `order`: Preferred provider sequence
  - `allow_fallbacks`: Enable provider fallbacks
  - `quantizations`: Specific quantization requirements
  - `sort`: Sort by price or throughput
  - `max_price`: Price constraints

## Important Implementation Details

- Images are limited to first 50 per message
- Image processing converts RGBA to RGB automatically
- System prompts include automatic timestamp injection
- Tool calls are automatically executed and results inserted into conversation
- Streaming methods return generators/async generators
- Error handling returns fallback messages rather than exceptions
- Chat history is managed automatically by OpenRouterClient
- Memory can be cleared with `clear_memory()` method
- Conversation history can be displayed with colored output via `print_memory()`

## Environment Variables

- `OPENROUTER_API_KEY`: Required API key for OpenRouter service
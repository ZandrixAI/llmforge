# Server Guide

Run LLMForge as an API server.

## Basic Server

```python
from llmforge.server import serve

serve(
    model="meta-llama/Llama-3.2-1B-Instruct",
    host="0.0.0.0",
    port=8000,
)
```

## With Authentication

```python
from llmforge.server import serve

serve(
    model="meta-llama/Llama-3.2-1B-Instruct",
    api_key="your-api-key",
    allowed_origins=["https://yourapp.com"],
)
```

## Using the Server

### Generate Endpoint

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR-API-KEY" \
  -d '{
    "prompt": "Write a poem",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128
  }'
```

## OpenAI-Compatible API

```python
from llmforge.server import OpenAICompatibleServer

server = OpenAICompatibleServer(
    model="meta-llama/Llama-3.2-1B-Instruct",
)
```

### OpenAI SDK Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-dummy",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="llama-3.2-1b",
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=128,
)
```

## Server Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | required | Model name or path |
| `host` | "0.0.0.0" | Server host |
| `port` | 8000 | Server port |
| `api_key` | None | API key for auth |
| `allowed_origins` | ["*"] | CORS origins |
| `max_batch_size` | 8 | Max batch size |
| `timeout` | 120 | Request timeout |
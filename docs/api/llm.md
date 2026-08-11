# LLM Adapters

## `OpenAICompatibleLLM`

Generic adapter for OpenAI-compatible APIs (Groq, OpenRouter, OpenAI, etc.).

```python
from ai_lang import OpenAICompatibleLLM

# Groq
llm = OpenAICompatibleLLM.groq(api_key="gsk-...", model="llama-3.3-70b-versatile")

# OpenRouter
llm = OpenAICompatibleLLM.openrouter(api_key="sk-or-...", model="google/gemini-2.0-flash-001")

# OpenAI
llm = OpenAICompatibleLLM.openai(api_key="sk-...", model="gpt-4")

# Custom
llm = OpenAICompatibleLLM(base_url="https://...", api_key="...", model="...")
```

## `from_env()`

Auto-detect provider from environment variables:

```python
from ai_lang import from_env

llm = from_env()  # Uses GROQ_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY
```

## `MockLLM`

For testing without API calls:

```python
from ai_lang import MockLLM

llm = MockLLM()  # Returns fixed responses
```

## `LLMResponse`

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | str | Raw response text |
| `tokens_used` | int | Total tokens consumed |
| `cost` | float | Estimated cost in USD |

Methods:
- `as_json()` — Parse response as JSON

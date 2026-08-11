import json
import time
from abc import ABC, abstractmethod
from typing import Any


class LLMResponse:
    def __init__(self, content: str, tokens_used: int = 0, cost: float = 0.0):
        self.content = content
        self.tokens_used = tokens_used
        self.cost = cost

    def as_json(self) -> dict:
        try:
            return json.loads(self.content)
        except json.JSONDecodeError:
            return {"raw": self.content}


class BaseLLM(ABC):
    @abstractmethod
    def call(self, prompt: str, context: dict = None) -> LLMResponse:
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        pass


class MockLLM(BaseLLM):
    def __init__(self, name: str = "mock-llm-v1"):
        self.name = name
        self.call_count = 0
        self.total_tokens = 0

    def call(self, prompt: str, context: dict = None) -> LLMResponse:
        self.call_count += 1
        tokens = self.estimate_tokens(prompt)
        self.total_tokens += tokens
        response = json.dumps({
            "status": "ok",
            "action": "proceed",
            "confidence": 0.92,
            "reasoning": f"Mock response for: {prompt[:60]}..."
        })
        return LLMResponse(content=response, tokens_used=tokens, cost=tokens * 0.00001)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


class OpenAICompatibleLLM(BaseLLM):
    """Generic OpenAI-compatible LLM adapter.

    Works with any provider that implements the OpenAI /v1/chat/completions endpoint:
    - OpenAI (default)
    - OpenRouter (https://openrouter.ai/api/v1)
    - Groq (https://api.groq.com/openai/v1)
    - Together, Fireworks, Ollama, vLLM, etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        headers: dict = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.call_count = 0
        self.total_tokens = 0
        self._client = None

    @classmethod
    def openrouter(cls, api_key: str, model: str = "google/gemini-2.0-flash-001") -> "OpenAICompatibleLLM":
        return cls(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            timeout=60.0,
        )

    @classmethod
    def groq(cls, api_key: str, model: str = "llama-3.3-70b-versatile") -> "OpenAICompatibleLLM":
        return cls(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model,
            timeout=10.0,
            max_retries=1,
        )

    @classmethod
    def openai(cls, api_key: str, model: str = "gpt-4") -> "OpenAICompatibleLLM":
        return cls(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model,
            timeout=30.0,
        )

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers=self.headers if self.headers else None,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def call(self, prompt: str, context: dict = None, response_format: dict = None) -> LLMResponse:
        messages = []
        if context:
            context_str = json.dumps(context, default=str) if isinstance(context, dict) else str(context)
            messages.append({"role": "system", "content": f"Context: {context_str}"})
        messages.append({"role": "user", "content": prompt})

        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        content = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else self.estimate_tokens(prompt)
        completion_tokens = usage.completion_tokens if usage else self.estimate_tokens(content)
        total = prompt_tokens + completion_tokens

        self.call_count += 1
        self.total_tokens += total

        return LLMResponse(
            content=content,
            tokens_used=total,
            cost=self._estimate_cost(prompt_tokens, completion_tokens),
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * 0.0000025) + (completion_tokens * 0.00001)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


def from_env() -> BaseLLM:
    """Factory: detect API key from environment and return appropriate LLM.

    Priority:
    1. OPENROUTER_API_KEY → OpenRouter
    2. GROQ_API_KEY → Groq
    3. OPENAI_API_KEY → OpenAI
    4. Fallback → MockLLM
    """
    import os

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
        return OpenAICompatibleLLM.openrouter(api_key=openrouter_key, model=model)

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return OpenAICompatibleLLM.groq(api_key=groq_key, model=model)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-4")
        return OpenAICompatibleLLM.openai(api_key=openai_key, model=model)

    return MockLLM()

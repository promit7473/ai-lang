# Installation

## pip

```bash
pip install ai-lang
```

## Optional Dependencies

| Feature | Install Command |
|---------|----------------|
| Groq support | `pip install ai-lang[groq]` |
| OpenRouter support | `pip install ai-lang[openrouter]` |
| OpenAI support | `pip install ai-lang[openai]` |
| All providers | `pip install ai-lang[all]` |
| Development | `pip install ai-lang[dev]` |

## Verify Installation

```python
import ai_lang
print(ai_lang.__version__)
```

## API Keys

Set environment variables for your chosen provider:

```bash
# Groq (recommended for speed)
export GROQ_API_KEY="gsk-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# OpenAI
export OPENAI_API_KEY="sk-..."
```

## Requirements

- Python >= 3.10
- PyTorch (CPU or CUDA)
- openai >= 1.0 (for LLM features)
- numpy, matplotlib (for robotics features)

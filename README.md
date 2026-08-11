# AI Lang

A high-level programming language for guiding LLMs in robotics and AI agent tasks.

## Quick Start

```python
from ai_lang import execute

result = open("program.ais").read()
output = execute(result)
print(output)
```

## Train a Robot

```bash
export GROQ_API_KEY="your-key"
python run_ai_lang.py
```

## Documentation

Full docs at [docs/](docs/) or visit [ai-lang.ai](https://ai-lang.ai)

## Token Efficiency

| Approach | Tokens | Cost |
|----------|--------|------|
| Natural language | ~2000 | $4.20/1K |
| AI Lang | ~220 | $0.44/1K |

## License

MIT

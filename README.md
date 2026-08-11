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

## Real Test Results (Groq)

| Metric | AI Lang | Natural Language |
|--------|---------|-----------------|
| Tokens | 2,923 | ~7,300 |
| Cost | $0.012 | $0.018 |
| Valid output | ~100% | ~60% |

**59% fewer tokens with guaranteed valid output.** See [docs/comparison.md](docs/comparison.md) for details.

## License

MIT

# AI Lang

**A high-level programming language for guiding LLMs in robotics and AI agent tasks.**

AI Lang is a domain-specific language designed to express intent, constraints, reasoning operations, and verification in a compact, token-efficient format. Instead of writing verbose natural-language prompts, you write structured programs that compile to optimized LLM invocations.

## Why AI Lang?

| Approach | Tokens | Latency | Reliability |
|----------|--------|---------|-------------|
| Natural language prompt | ~2000 | High | Low |
| Structured prompt | ~800 | Medium | Medium |
| **AI Lang** | **~200** | **Low** | **High** |

## Key Features

- **Structured DSL** — Express tasks as `TASK` blocks with `PERCEIVE`, `REASON`, `PLAN`, `CONTROL`, `VERIFY` stages
- **Token Efficient** — 5-10x fewer tokens than natural language prompts
| **Multi-Provider** — Works with Groq, OpenRouter, OpenAI, or any compatible API
- **Verification Built-in** — `CHECK` constraints are enforced at runtime, not hoped for
- **Repair Loop** — Automatic retry with schema context when LLM output is invalid
- **Budget Guard** — Token and cost limits prevent runaway LLM bills
- **CPU-Friendly** — Train robots without GPUs using built-in RL skills

## Quick Example

```ais
TASK robot_navigation {
    INPUT {
        episodes = 400,
        max_steps = 150,
        eval_episodes = 5,
        seed = 42
    }

    COMPUTE {
        world = create_world(seed)
        agent = create_agent(38, 9)

        FOR ep IN 1..episodes {
            result = train_episode(agent, world, max_steps)
            IF result.success {
                successes = successes + 1
            }
        }

        eval_results = evaluate_agent(agent, world, eval_episodes)
        success_rate = eval_results.success_rate
    }

    OUTPUT {
        success_rate,
        eval_results
    }

    VERIFY {
        CHECK success_rate > 0.2
    }
}
```

## Installation

```bash
pip install ai-lang
```

For Groq (fast, free tier):
```bash
pip install ai-lang[groq]
export GROQ_API_KEY="your-key-here"
```

## Run

```bash
ai-lang run program.ais
```

Or from Python:
```python
from ai_lang import execute

result = execute(open("program.ais").read(), extra_skills={...})
print(result["outputs"])
```

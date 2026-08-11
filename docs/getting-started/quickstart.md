# Quick Start

## 1. Write Your First AI Lang Program

Create a file `hello.ais`:

```ais
TASK hello {
    INPUT { name = "World" }

    COMPUTE {
        greeting = "Hello, " + name
        LOG "{greeting}"
    }

    OUTPUT { greeting }
}
```

## 2. Run It

```bash
ai-lang run hello.ais
```

Or from Python:

```python
from ai_lang import execute

result = execute(open("hello.ais").read())
print(result["outputs"]["greeting"])  # "Hello, World"
```

## 3. Add LLM Intelligence

```ais
TASK analyze_image {
    INPUT {
        image = "camera_feed.jpg",
        model = "llama-3.3-70b-versatile"
    }

    PERCEIVE {
        objects = detect_objects(image)
        scene = classify_scene(image)
    }

    REASON {
        risk = assess_risk(objects, scene)
        action = plan_action(risk)
    }

    OUTPUT {
        objects,
        risk,
        action
    }

    VERIFY {
        CHECK risk >= 0 AND risk <= 1
    }
}
```

## 4. Run with Groq

```bash
export GROQ_API_KEY="gsk-..."
ai-lang run analyze_image.ais --provider groq --use-llm
```

## Next Steps

- [Write your first robot](first-program.md)
- [Robot Navigation Tutorial](../tutorials/robot-navigation.md)
- [Language Syntax Reference](../language/syntax.md)

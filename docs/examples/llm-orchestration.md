# LLM Orchestration Example

Chain multiple LLM calls with verification.

```ais
TASK analyze {
    INPUT {
        image = "camera.jpg",
        model = "llama-3.3-70b-versatile"
    }

    PERCEIVE {
        objects = detect_objects(image)
    }

    REASON {
        description = describe_scene(objects)
        risk = assess_risk(objects)
    }

    OUTPUT {
        objects,
        description,
        risk
    }

    VERIFY {
        CHECK risk >= 0 AND risk <= 1
    }
}
```

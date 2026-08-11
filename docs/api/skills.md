# Skills Registry

## Global Registry

```python
from ai_lang.skills import registry, Skill

# Check if skill exists
if registry.has("detect_objects"):
    skill = registry.get("detect_objects")

# List all skills
all_skills = registry.all()

# Filter by category
perception_skills = registry.list_by_category("perception")
```

## Register Custom Skills

```python
from ai_lang.skills import register_skill

register_skill(
    name="my_detector",
    func="detect",
    category="perception",
    description="Custom object detector",
    implementation=my_detector_function,
)
```

## Skill Definition

```python
from ai_lang.skills import Skill

skill = Skill(
    name="my_skill",
    func="my_func",
    category="custom",
    description="Does something",
    implementation=lambda x: x,
)
```

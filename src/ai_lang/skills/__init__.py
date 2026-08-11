import json
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class SkillParam:
    name: str
    type_hint: str = "any"
    description: str = ""


@dataclass
class Skill:
    name: str
    func: str
    category: str
    description: str
    params: list = field(default_factory=list)
    implementation: Callable = None
    llm_prompt_template: str = ""
    uses_llm: bool = False
    output_schema: dict = None

    def execute(self, *args, **kwargs) -> Any:
        if self.implementation:
            return self.implementation(*args, **kwargs)
        return None

    def build_prompt(self, **kwargs) -> str:
        if not self.llm_prompt_template:
            base = f"Execute {self.name} with args={kwargs}"
        else:
            try:
                base = self.llm_prompt_template.format(**kwargs)
            except KeyError:
                base = self.llm_prompt_template

        if self.output_schema:
            schema_str = json.dumps(self.output_schema, indent=2)
            base += f"\n\nReturn valid JSON matching this schema:\n{schema_str}"

        return base

    def build_repair_prompt(self, original_prompt: str, invalid_output: str, error: str) -> str:
        schema_str = json.dumps(self.output_schema, indent=2) if self.output_schema else "N/A"
        return (
            f"Your previous output failed validation.\n\n"
            f"Error: {error}\n\n"
            f"Required schema:\n{schema_str}\n\n"
            f"Original request:\n{original_prompt}\n\n"
            f"Your invalid output:\n{invalid_output}\n\n"
            f"Return valid JSON matching the schema. No explanation, just JSON."
        )

    def validate_output(self, output: Any) -> tuple[bool, str]:
        if self.output_schema is None:
            return True, ""
        return _validate_json_schema(output, self.output_schema)


def _validate_json_schema(data: Any, schema: dict) -> tuple[bool, str]:
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            return False, f"Expected object, got {type(data).__name__}"
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                return False, f"Missing required field: '{key}'"
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in data:
                valid, err = _validate_json_schema(data[key], prop_schema)
                if not valid:
                    return False, f"Field '{key}': {err}"

    elif schema_type == "array":
        if not isinstance(data, list):
            return False, f"Expected array, got {type(data).__name__}"
        min_items = schema.get("minItems")
        if min_items is not None and len(data) < min_items:
            return False, f"Array too short: {len(data)} < {min_items}"
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                valid, err = _validate_json_schema(item, items_schema)
                if not valid:
                    return False, f"Item [{i}]: {err}"

    elif schema_type == "number" or schema_type == "integer":
        if not isinstance(data, (int, float)):
            return False, f"Expected {schema_type}, got {type(data).__name__}"
        minimum = schema.get("minimum")
        if minimum is not None and data < minimum:
            return False, f"Value {data} below minimum {minimum}"
        maximum = schema.get("maximum")
        if maximum is not None and data > maximum:
            return False, f"Value {data} above maximum {maximum}"

    elif schema_type == "string":
        if not isinstance(data, str):
            return False, f"Expected string, got {type(data).__name__}"

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            return False, f"Expected boolean, got {type(data).__name__}"

    return True, ""


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill
        self._skills[skill.func] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def has(self, name: str) -> bool:
        return name in self._skills

    def list_by_category(self, category: str) -> list[Skill]:
        return [s for s in self._skills.values() if s.category == category]

    def all(self) -> list[Skill]:
        seen = set()
        result = []
        for skill in self._skills.values():
            if skill.name not in seen:
                seen.add(skill.name)
                result.append(skill)
        return result


registry = SkillRegistry()


def register_skill(
    name: str,
    func: str,
    category: str,
    description: str = "",
    params: list = None,
    implementation: Callable = None,
    llm_prompt_template: str = "",
    uses_llm: bool = False,
    output_schema: dict = None,
):
    skill = Skill(
        name=name,
        func=func,
        category=category,
        description=description,
        params=params or [],
        implementation=implementation,
        llm_prompt_template=llm_prompt_template,
        uses_llm=uses_llm,
        output_schema=output_schema,
    )
    registry.register(skill)
    return skill

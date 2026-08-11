"""Unit tests for new v0.2 functionality: OpenAICompatibleLLM, schema validation, repair loop, budget guard."""
import sys
import os
sys.path.insert(0, "src")

from ai_lang.llm import OpenAICompatibleLLM, MockLLM, from_env, LLMResponse
from ai_lang.skills import registry, Skill, register_skill, _validate_json_schema


class TestOpenAICompatibleLLM:
    def test_groq_factory(self):
        llm = OpenAICompatibleLLM.groq(api_key="test-key")
        assert llm.base_url == "https://api.groq.com/openai/v1"
        assert llm.model == "llama-3.3-70b-versatile"
        assert llm.api_key == "test-key"
        assert llm.timeout == 10.0
        assert llm.max_retries == 1

    def test_openrouter_factory(self):
        llm = OpenAICompatibleLLM.openrouter(api_key="test-key")
        assert llm.base_url == "https://openrouter.ai/api/v1"
        assert llm.model == "google/gemini-2.0-flash-001"
        assert llm.timeout == 60.0

    def test_openai_factory(self):
        llm = OpenAICompatibleLLM.openai(api_key="test-key")
        assert llm.base_url == "https://api.openai.com/v1"
        assert llm.model == "gpt-4"

    def test_custom_model(self):
        llm = OpenAICompatibleLLM.groq(api_key="key", model="llama-3.1-8b-instant")
        assert llm.model == "llama-3.1-8b-instant"

    def test_base_url_trailing_slash(self):
        llm = OpenAICompatibleLLM(base_url="https://example.com/", api_key="key", model="m")
        assert llm.base_url == "https://example.com"

    def test_estimate_tokens(self):
        llm = OpenAICompatibleLLM.groq(api_key="key")
        assert llm.estimate_tokens("hello world") == 2  # 11 // 4 = 2


class TestFromEnv:
    def test_detects_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = from_env()
        assert isinstance(llm, OpenAICompatibleLLM)
        assert "groq" in llm.base_url

    def test_detects_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = from_env()
        assert isinstance(llm, OpenAICompatibleLLM)
        assert "openrouter" in llm.base_url

    def test_fallback_to_mock(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = from_env()
        assert isinstance(llm, MockLLM)

    def test_groq_model_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = from_env()
        assert llm.model == "llama-3.1-8b-instant"


class TestSchemaValidation:
    def test_valid_object(self):
        schema = {
            "type": "object",
            "required": ["risk", "safe"],
            "properties": {
                "risk": {"type": "number", "minimum": 0, "maximum": 1},
                "safe": {"type": "boolean"},
            },
        }
        valid, err = _validate_json_schema({"risk": 0.15, "safe": True}, schema)
        assert valid, f"Expected valid, got error: {err}"

    def test_missing_required_field(self):
        schema = {"type": "object", "required": ["risk", "safe"]}
        valid, err = _validate_json_schema({"risk": 0.15}, schema)
        assert not valid
        assert "safe" in err

    def test_number_out_of_range(self):
        schema = {"type": "number", "minimum": 0, "maximum": 1}
        valid, err = _validate_json_schema(1.5, schema)
        assert not valid
        assert "maximum" in err

    def test_wrong_type(self):
        schema = {"type": "number"}
        valid, err = _validate_json_schema("not a number", schema)
        assert not valid

    def test_array_validation(self):
        schema = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "object", "required": ["id"]},
        }
        valid, err = _validate_json_schema([{"id": 1}, {"id": 2}], schema)
        assert valid

    def test_array_too_short(self):
        schema = {"type": "array", "minItems": 2}
        valid, err = _validate_json_schema([1], schema)
        assert not valid
        assert "short" in err

    def test_nested_object_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "object",
                    "required": ["thrust"],
                    "properties": {"thrust": {"type": "number"}},
                }
            },
        }
        valid, err = _validate_json_schema({"command": {"thrust": 0.7}}, schema)
        assert valid

    def test_nested_error_path(self):
        schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "object",
                    "properties": {"thrust": {"type": "number"}}
                }
            },
        }
        valid, err = _validate_json_schema({"command": {"thrust": "bad"}}, schema)
        assert not valid
        assert "command" in err and "thrust" in err


class TestSkillSchemaIntegration:
    def test_skill_has_schema(self):
        skill = registry.get("assess_collision")
        assert skill is not None
        assert skill.output_schema is not None
        assert skill.output_schema["type"] == "object"
        assert "risk" in skill.output_schema["required"]

    def test_skill_validate_output(self):
        skill = registry.get("assess_collision")
        valid, err = skill.validate_output({"risk": 0.15, "safe": True})
        assert valid

    def test_skill_validate_output_invalid(self):
        skill = registry.get("assess_collision")
        valid, err = skill.validate_output({"risk": 2.0, "safe": True})
        assert not valid

    def test_build_prompt_includes_schema(self):
        skill = registry.get("assess_collision")
        prompt = skill.build_prompt()
        assert "schema" in prompt.lower() or "JSON" in prompt

    def test_build_repair_prompt(self):
        skill = registry.get("assess_collision")
        repair = skill.build_repair_prompt(
            original_prompt="test prompt",
            invalid_output='{"risk": 2.0}',
            error="Value 2.0 above maximum 1"
        )
        assert "failed" in repair.lower() or "error" in repair.lower()
        assert "2.0" in repair
        assert "schema" in repair.lower() or "JSON" in repair

    def test_skill_without_schema(self):
        register_skill(
            name="no_schema_skill",
            func="no_schema_func",
            category="test",
            implementation=lambda: "test",
        )
        skill = registry.get("no_schema_skill")
        valid, err = skill.validate_output("anything")
        assert valid
        assert err == ""


class TestBudgetGuard:
    def test_runtime_accepts_budget_params(self):
        from ai_lang.runtime import AIRuntime
        from ai_lang.llm import MockLLM
        runtime = AIRuntime(llm=MockLLM(), use_llm=True, max_tokens=100, max_cost=0.01)
        assert runtime.max_tokens == 100
        assert runtime.max_cost == 0.01

    def test_budget_exceeded_tokens(self):
        from ai_lang.runtime import AIRuntime, RuntimeResult
        from ai_lang.llm import MockLLM
        runtime = AIRuntime(llm=MockLLM(), use_llm=True, max_tokens=1)
        runtime._tokens_used = 10
        result = RuntimeResult(task_name="test", success=True)
        assert runtime._check_budget(result) is False
        assert result.budget_exceeded is True

    def test_budget_not_exceeded(self):
        from ai_lang.runtime import AIRuntime, RuntimeResult
        from ai_lang.llm import MockLLM
        runtime = AIRuntime(llm=MockLLM(), use_llm=True, max_tokens=1000)
        runtime._tokens_used = 5
        result = RuntimeResult(task_name="test", success=True)
        assert runtime._check_budget(result) is True
        assert result.budget_exceeded is False


class TestRepairLoop:
    def test_repair_prompt_includes_error_context(self):
        skill = registry.get("assess_collision")
        repair = skill.build_repair_prompt(
            original_prompt="Assess collision risk for trajectory X",
            invalid_output='{"risk": "high"}',
            error="Field 'risk': Expected number, got str"
        )
        assert "Assess collision risk" in repair
        assert '"high"' in repair or "high" in repair
        assert "Expected number" in repair

    def test_max_retries_parameter(self):
        from ai_lang.runtime import AIRuntime
        from ai_lang.llm import MockLLM
        runtime = AIRuntime(llm=MockLLM(), use_llm=True, max_retries=5)
        assert runtime.max_retries == 5


def run_tests():
    """Run all tests without pytest dependency."""
    import traceback

    test_classes = [
        TestOpenAICompatibleLLM,
        TestFromEnv,
        TestSchemaValidation,
        TestSkillSchemaIntegration,
        TestBudgetGuard,
        TestRepairLoop,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            method = getattr(instance, method_name)
            try:
                # Handle monkeypatch for TestFromEnv
                import inspect
                sig = inspect.signature(method)
                if 'monkeypatch' in sig.parameters:
                    # Simple monkeypatch mock
                    class MonkeyPatch:
                        def __init__(self):
                            self._originals = []
                        def setenv(self, key, value):
                            old = os.environ.get(key)
                            self._originals.append((key, old))
                            os.environ[key] = value
                        def delenv(self, key, raising=True):
                            old = os.environ.get(key)
                            self._originals.append((key, old))
                            if key in os.environ:
                                del os.environ[key]
                        def undo(self):
                            for key, old in reversed(self._originals):
                                if old is None:
                                    os.environ.pop(key, None)
                                else:
                                    os.environ[key] = old
                    mp = MonkeyPatch()
                    try:
                        method(mp)
                    finally:
                        mp.undo()
                else:
                    method()
                print(f"  PASS: {cls.__name__}.{method_name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {cls.__name__}.{method_name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

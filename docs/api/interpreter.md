# Interpreter API

## `execute(source, inputs=None, extra_skills=None)`

Execute an AI Lang program.

**Parameters:**
- `source` (str): AI Lang program source code
- `inputs` (dict, optional): Input variable values
- `extra_skills` (dict, additional skill functions

**Returns:**
```python
{
    "success": bool,
    "task_name": str,
    "outputs": dict,
    "log": list[str],
    "variables": dict,
}
```

**Example:**
```python
from ai_lang import execute

result = execute("""
TASK hello {
    COMPUTE { msg = "Hello!" }
    OUTPUT { msg }
}
""")

print(result["outputs"]["msg"])  # "Hello!"
```

## `AIInterpreter`

Lower-level interface for executing parsed programs.

```python
from ai_lang import AIInterpreter
from ai_lang.parser import parse

interpreter = AIInterpreter(extra_skills={...})
task = parse(source)
result = interpreter.execute(task, inputs={...})
```

# Types

AI Lang is dynamically typed. The following types are supported:

## Primitive Types

| Type | Example | Description |
|------|---------|-------------|
| `int` | `42` | Integer numbers |
| `float` | `3.14` | Floating-point numbers |
| `string` | `"hello"` | Text strings |
| `bool` | `true`, `false` | Boolean values |

## Composite Types

| Type | Example | Description |
|------|---------|-------------|
| `list` | `[1, 2, 3]` | Ordered collection |
| `dict` | `{"key": "value"}` | Key-value mapping |

## Skill Return Types

Skills can return any type. Common return types:

| Skill | Return Type |
|-------|-------------|
| `create_world()` | `World2D` (object) |
| `create_agent()` | `DQNAgent` (object) |
| `train_episode()` | `dict` with keys: `reward`, `steps`, `success`, `collision` |
| `evaluate_agent()` | `dict` with keys: `success_rate`, `avg_reward`, `episodes` |

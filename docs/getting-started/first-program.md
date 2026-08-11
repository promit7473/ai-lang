# First Program

Write your first AI Lang program step by step.

## Step 1: Create a file

Create `hello.ais`:

```ais
TASK hello {
    COMPUTE {
        message = "Hello, AI Lang!"
        LOG "{message}"
    }

    OUTPUT { message }
}
```

## Step 2: Run it

```python
from ai_lang import execute

result = execute(open("hello.ais").read())
print(result)
```

Output:
```
{
    'success': True,
    'task_name': 'hello',
    'outputs': {'message': 'Hello, AI Lang!'},
    'log': ['  Hello, AI Lang!'],
    'variables': {'message': 'Hello, AI Lang!'}
}
```

## Step 3: Add inputs

```ais
TASK greet {
    INPUT { name = "World" }

    COMPUTE {
        greeting = "Hello, " + name + "!"
    }

    OUTPUT { greeting }
}
```

## Step 4: Add control flow

```ais
TASK countdown {
    INPUT { start = 10 }

    COMPUTE {
        FOR i IN 1..start {
            LOG "Count: {i}"
        }
    }
}
```

## Next

- [Language Syntax](../language/syntax.md)
- [Robot Navigation Tutorial](../tutorials/robot-navigation.md)

# Expressions

## Arithmetic

```ais
a + b        # Addition
a - b        # Subtraction
a * b        # Multiplication
a / b        # Division
a % b        # Modulo
```

## Comparison

```ais
a < b        # Less than
a > b        # Greater than
a <= b       # Less than or equal
a >= b       # Greater than or equal
a == b       # Equal
a != b       # Not equal
```

## Logical

```ais
a AND b      # Logical AND
a OR b       # Logical OR
NOT a        # Logical NOT
```

## Field Access

Access fields of objects or dictionaries:

```ais
result.success
agent.epsilon
eval_results.success_rate
```

## Array/List Access

```ais
rewards[0]       # First element
rewards[-1]      # Last element
my_list[2]       # Third element
```

## Function Calls

```ais
create_world(42)
create_agent(38, 9)
train_episode(agent, world, max_steps)
sum([1, 2, 3])
```

## Operator Precedence

1. `NOT`, unary `-`
2. `*`, `/`, `%`
3. `+`, `-`
4. `<`, `>`, `<=`, `>=`
5. `==`, `!=`
6. `AND`
7. `OR`

Use parentheses for explicit grouping:

```ais
result = (a + b) * c
```

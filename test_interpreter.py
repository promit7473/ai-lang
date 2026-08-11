"""Quick test of the AI Lang interpreter."""
import sys
sys.path.insert(0, "src")

from ai_lang import execute

# Test 1: Basic arithmetic and variables
source1 = """
TASK test_math {
    INPUT { x = 10, y = 20 }

    COMPUTE {
        result = x + y
        doubled = result * 2
        FOR i IN 1..5 {
            doubled = doubled + 1
        }
    }

    OUTPUT { result, doubled }
}
"""

print("Test 1: Basic arithmetic")
result = execute(source1)
print(f"  Success: {result['success']}")
print(f"  Variables: {result['variables']}")
assert result['variables'].get('result') == 30
assert result['variables'].get('doubled') == 65
print("  PASS\n")

# Test 2: IF/ELSE
source2 = """
TASK test_if {
    COMPUTE {
        x = 15
        IF x > 10 {
            category = "big"
        } ELSE {
            category = "small"
        }

        IF x > 20 {
            large = "yes"
        } ELSE {
            large = "no"
        }
    }

    OUTPUT { category, large }
}
"""

print("Test 2: IF/ELSE")
result = execute(source2)
print(f"  Success: {result['success']}")
print(f"  Variables: {result['variables']}")
assert result['variables'].get('category') == "big"
assert result['variables'].get('large') == "no"
print("  PASS\n")

# Test 3: FOR loop with accumulation
source3 = """
TASK test_loop {
    COMPUTE {
        total = 0
        FOR i IN 1..10 {
            total = total + i
        }
    }

    OUTPUT { total }
}
"""

print("Test 3: FOR loop accumulation")
result = execute(source3)
print(f"  Success: {result['success']}")
print(f"  Variables: {result['variables']}")
assert result['variables'].get('total') == 55
print("  PASS\n")

# Test 4: LOG with interpolation
source4 = """
TASK test_log {
    INPUT { name = "Robot", count = 42 }

    COMPUTE {
        LOG "Starting {name} with count={count}"
        result = count * 2
        LOG "Result is {result}"
    }

    OUTPUT { result }
}
"""

print("Test 4: LOG with interpolation")
result = execute(source4)
print(f"  Success: {result['success']}")
print(f"  Log: {result['log']}")
assert len(result['log']) == 2
print("  PASS\n")

print("All interpreter tests passed!")

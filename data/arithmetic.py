import random
import re


def generate_arithmetic_problems(n, difficulty):
    """
    Generate arithmetic problems at the given difficulty level.

    Difficulty levels:
         1: 4-digit addition (1234 + 5678)
         2: 5-digit addition (12345 + 67890)
         3: 4-digit subtraction (5678 - 1234)
         4: 5-digit subtraction (67890 - 12345)
         5: 2x2 multiplication (23 * 45)
         6: 3x2 multiplication (123 * 45)
         7: 3x3 multiplication (123 * 456)
         8: 4-digit division (7392 / 12)
         9: mixed large operations
        10: 6-digit addition (123456 + 654321)
    """
    problems = []
    for _ in range(n):
        if difficulty == 1:
            a, b = random.randint(1000, 9999), random.randint(1000, 9999)
            op, answer = "+", a + b
        elif difficulty == 2:
            a, b = random.randint(10000, 99999), random.randint(10000, 99999)
            op, answer = "+", a + b
        elif difficulty == 3:
            a = random.randint(1000, 9999)
            b = random.randint(1000, a)
            op, answer = "-", a - b
        elif difficulty == 4:
            a = random.randint(10000, 99999)
            b = random.randint(10000, a)
            op, answer = "-", a - b
        elif difficulty == 5:
            a, b = random.randint(10, 99), random.randint(10, 99)
            op, answer = "\u00d7", a * b
        elif difficulty == 6:
            a, b = random.randint(100, 999), random.randint(10, 99)
            op, answer = "\u00d7", a * b
        elif difficulty == 7:
            a, b = random.randint(100, 999), random.randint(100, 999)
            op, answer = "\u00d7", a * b
        elif difficulty == 8:
            b = random.randint(2, 99)
            answer = random.randint(10, 9999)
            a = b * answer
            op = "\u00f7"
        elif difficulty == 9:
            # Mixed: randomly pick a hard operation
            sub_diff = random.choice([2, 4, 6, 7])
            sub = generate_arithmetic_problems(1, sub_diff)
            problems.append(sub[0])
            continue
        elif difficulty == 10:
            a, b = random.randint(100000, 999999), random.randint(100000, 999999)
            op, answer = "+", a + b
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")

        prompt = f"{a} {op} {b} = "
        problems.append({"prompt": prompt, "answer": str(answer), "a": a, "b": b, "op": op})
    return problems


def extract_number(text):
    """Extract the first integer (possibly negative) from a string."""
    match = re.search(r"-?\d+", text)
    if match:
        return match.group()
    return None

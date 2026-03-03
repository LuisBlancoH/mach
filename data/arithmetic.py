import random
import re


def generate_arithmetic_problems(n, difficulty):
    """
    Generate arithmetic problems at the given difficulty level.

    Difficulty levels:
        1: single digit addition (3 + 5)
        2: two digit addition (23 + 45)
        3: three digit addition (347 + 589)
        4: two digit subtraction (67 - 23)
        5: three digit subtraction (523 - 178)
        6: single x single multiplication (7 * 8)
        7: two digit x single digit multiplication (23 * 7)
    """
    problems = []
    for _ in range(n):
        if difficulty == 1:
            a, b = random.randint(1, 9), random.randint(1, 9)
            op, answer = "+", a + b
        elif difficulty == 2:
            a, b = random.randint(10, 99), random.randint(10, 99)
            op, answer = "+", a + b
        elif difficulty == 3:
            a, b = random.randint(100, 999), random.randint(100, 999)
            op, answer = "+", a + b
        elif difficulty == 4:
            a = random.randint(10, 99)
            b = random.randint(1, a)
            op, answer = "-", a - b
        elif difficulty == 5:
            a = random.randint(100, 999)
            b = random.randint(1, a)
            op, answer = "-", a - b
        elif difficulty == 6:
            a, b = random.randint(2, 9), random.randint(2, 9)
            op, answer = "\u00d7", a * b
        elif difficulty == 7:
            a, b = random.randint(10, 99), random.randint(2, 9)
            op, answer = "\u00d7", a * b
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

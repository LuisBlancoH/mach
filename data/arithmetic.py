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


def _make_operands(op_type):
    """
    Generate operands with UNIFORM ranges across all operations.

    All ops use a, b in [10, 99]. Order is random EXCEPT for sub/div
    where a >= b is enforced (to keep answers non-negative / meaningful).

    The test prompt "a ? b = " is indistinguishable across operations —
    the model MUST read demos to determine which operation to apply.
    """
    a = random.randint(10, 99)
    b = random.randint(10, 99)

    if op_type in ("sub", "div") and b > a:
        a, b = b, a

    if op_type == "add":
        return a, b, a + b
    elif op_type == "sub":
        return a, b, a - b
    elif op_type == "mul":
        return a, b, a * b
    elif op_type == "div":
        return a, b, a // b
    elif op_type == "mod":
        return a, b, a % b
    elif op_type == "max":
        return a, b, max(a, b)
    elif op_type == "min":
        return a, b, min(a, b)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")


def generate_few_shot_episode(n_problems, n_demos=None, op_type=None):
    """
    Generate a few-shot episode with a hidden operation.

    Picks a random operation (add/sub/mul/div) but uses '?' as the
    symbol. Demo problems show the complete answer; test problems
    show only the question. The model must infer the operation from
    demos and apply it to test problems.
    """
    if n_demos is None:
        n_demos = max(2, n_problems // 4)
    if op_type is None:
        op_type = random.choice(["add", "sub", "mul", "div", "mod", "max", "min"])

    problems = []
    for i in range(n_problems):
        a, b, answer = _make_operands(op_type)
        is_demo = (i < n_demos)

        if is_demo:
            prompt = f"{a} ? {b} = {answer}"
        else:
            prompt = f"{a} ? {b} = "

        problems.append({
            "prompt": prompt,
            "answer": str(answer),
            "a": a,
            "b": b,
            "op": op_type,
            "is_demo": is_demo,
            "difficulty": op_type,
        })

    return problems

import re

def repetition_score(text: str) -> float:
    """
    Very simple repetition measure:
    counts repeated 3+ word sequences.
    """
    tokens = re.findall(r"\w+|\S", text.lower())
    if len(tokens) < 50:
        return 0.0

    seen = {}
    repeats = 0
    total = 0

    for i in range(len(tokens) - 3):
        gram = tuple(tokens[i:i+3])
        total += 1
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] == 2:
            repeats += 1

    return repeats / max(total, 1)
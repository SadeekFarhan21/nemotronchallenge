"""Local re-implementation of the official competition grader.

Copied VERBATIM from the official `metric/nvidia-nemotron-metric` Kaggle
notebook (functions `extract_final_answer` and `verify`) so that our offline
validation scores match the hidden leaderboard exactly. Do not "improve" these
functions — they must mirror the grader bit-for-bit.
"""

import math
import re


def extract_final_answer(text: str | None) -> str:
    r"""Extract the final answer from a model response.

    Prioritises content inside ``\boxed{}``; otherwise falls back to
    "final answer" phrasings, then the last number, then the last line.
    """
    if text is None:
        return 'NOT_FOUND'

    # For each \boxed{ occurrence, take everything up to the last } before the
    # next \boxed{ (or end of text). Handles answers containing '}' and nested
    # LaTeX like \boxed{\frac{1}{2}}.
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
    """Return True if ``predicted`` matches ``stored_answer``.

    Numeric answers compare within rel_tol=1e-2 (abs_tol=1e-5); binary strings
    and everything else compare as case-insensitive strings.
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # Binary strings compare strictly (never as numbers).
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def score_predictions(answers: list[str], raw_outputs: list[str | None]) -> float:
    """Proportion of raw model outputs whose extracted answer verifies."""
    assert len(answers) == len(raw_outputs)
    if not answers:
        return 0.0
    correct = sum(
        verify(a, extract_final_answer(o)) for a, o in zip(answers, raw_outputs)
    )
    return correct / len(answers)

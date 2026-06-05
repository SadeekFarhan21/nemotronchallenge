"""Task-family classification and closed-form solvers.

The benchmark has 6 algorithmic task families, all phrased "In Alice's
Wonderland". Each problem embeds in-context (input -> output) examples that
fully determine a hidden rule; the task is to infer the rule and apply it to a
query. Several families are solvable in closed form, which lets us (a) measure
an achievable ceiling and (b) generate unlimited *verified* CoT training data.
"""

import re

FAMILIES = ["bitops", "gravity", "units", "cipher", "roman", "transform"]


def classify(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p:
        return "bitops"
    if "gravitational constant" in p:
        return "gravity"
    if "unit conversion" in p:
        return "units"
    if "encryption rules" in p:
        return "cipher"
    if "converted into" in p and "numeral" in p:
        return "roman"
    if "set of transformation" in p:
        return "transform"
    return "other"


# ----------------------------------------------------------------------------
# Example/query extraction helpers
# ----------------------------------------------------------------------------

def _pairs_arrow(prompt: str):
    """Return list of (lhs, rhs) for 'lhs -> rhs' example lines."""
    out = []
    for line in prompt.splitlines():
        if "->" in line:
            lhs, rhs = line.split("->", 1)
            out.append((lhs.strip(), rhs.strip()))
    return out


# ----------------------------------------------------------------------------
# Roman numerals
# ----------------------------------------------------------------------------

def to_roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
            (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
            (5, "V"), (4, "IV"), (1, "I")]
    res = []
    for v, s in vals:
        while n >= v:
            res.append(s)
            n -= v
    return "".join(res)


def solve_roman(prompt: str):
    m = re.search(r"write the number (\d+)", prompt)
    if not m:
        return None
    return to_roman(int(m.group(1)))


# ----------------------------------------------------------------------------
# Gravity: d = 0.5 * g * t^2  (fit g from examples)
# ----------------------------------------------------------------------------

def solve_gravity(prompt: str):
    obs = re.findall(r"t\s*=\s*([\d.]+)\s*s,\s*distance\s*=\s*([\d.]+)", prompt)
    if not obs:
        return None
    gs = [2 * float(d) / (float(t) ** 2) for t, d in obs]
    g = sum(gs) / len(gs)
    qm = re.search(r"distance for t\s*=\s*([\d.]+)", prompt)
    if not qm:
        return None
    t = float(qm.group(1))
    return round(0.5 * g * t * t, 2)


# ----------------------------------------------------------------------------
# Units: linear out = a * in  (fit a from examples)
# ----------------------------------------------------------------------------

def solve_units(prompt: str):
    ex = re.findall(r"([\d.]+)\s*m\s*becomes\s*([\d.]+)", prompt)
    if not ex:
        return None
    ratios = [float(o) / float(i) for i, o in ex]
    a = sum(ratios) / len(ratios)
    qm = re.search(r"convert the following measurement:\s*([\d.]+)", prompt)
    if not qm:
        return None
    return round(a * float(qm.group(1)), 2)


# ----------------------------------------------------------------------------
# Cipher: monoalphabetic substitution, derived from plaintext/ciphertext pairs
# (the prompt gives cipher -> plain examples; we decrypt the query).
# ----------------------------------------------------------------------------

def solve_cipher(prompt: str):
    pairs = _pairs_arrow(prompt)
    if not pairs:
        return None
    dec = {}  # cipher char -> plain char
    for ct, pt in pairs:
        for c, p in zip(ct, pt):
            if c.isalpha() and p.isalpha():
                if c in dec and dec[c] != p:
                    return None  # inconsistent
                dec[c] = p
    qm = re.search(r"decrypt the following text:\s*(.+)$", prompt, re.MULTILINE)
    if not qm:
        return None
    q = qm.group(1).strip()
    out = []
    for ch in q:
        out.append(dec.get(ch, ch) if ch.isalpha() else ch)
    res = "".join(out)
    if any(ch.isalpha() and ch not in dec for ch in q):
        return None  # unknown char -> can't fully solve
    return res


SOLVERS = {
    "roman": solve_roman,
    "gravity": solve_gravity,
    "units": solve_units,
    "cipher": solve_cipher,
}


def solve(prompt: str):
    fam = classify(prompt)
    fn = SOLVERS.get(fam)
    return fn(prompt) if fn else None

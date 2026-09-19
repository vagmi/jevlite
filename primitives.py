#!/usr/bin/env python3
"""
primitives.py — the one place a row becomes a prompt.

TypeSafe serves three question types against a state:

  choice   pick one of several named options   -> probabilities + confidence
  score    rate against ordered levels          -> expected level + legend
  noul     is this true?                        -> a single probability

All three arrive with `criteria`: a description per option, or per level. The
student is asked to read those descriptions at serve time, so it has to be
trained on them — and the teacher has to judge against the same text, or the
soft label describes a prompt nobody will ever send. That is why build_data.py,
teacher.py and jev_lite.py all render through this module and none of them
format options themselves.

Row shape (a superset of what jev_lite trains on):

  {"type": "choice", "state": ..., "question": ...,
   "options": ["billing", "technical"],
   "criteria": {"billing": "Payments, invoicing, refunds", ...},
   "label": [0.7, 0.3]}

  {"type": "score", "ordered": true, "options": ["Calm", "Frustrated", "Angry"],
   "label": [...]}                  # options ARE the levels; legend is positional

  {"type": "noul", "options": ["true", "false"],
   "criteria": {"true": "...", "false": "..."}, "label": [p_yes, p_no]}

`criteria` is always optional: the API allows a bare option set, and some real
tasks have no meaningful description to give (RACE's options are the answer
text itself). Training on a mix teaches the model to use criteria when they are
there and cope when they are not.
"""

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# noul answers in TypeSafe are keyed true/false, and the reported probability is
# the one for "true" — so index 0 must be the affirmative, always.
TRUE_FALSE = ["true", "false"]
YES_WORDS = {"yes", "true", "y"}
NO_WORDS = {"no", "false", "n"}

PREFIX = "Read the state and answer the question with one option letter.\n\n<state>\n"


def kind_of(row):
    """choice | score | noul, from an explicit type or the shape of the row."""
    if row.get("type") in ("choice", "score", "noul"):
        return row["type"]
    if row.get("ordered"):
        return "score"
    opts = [str(o).strip().lower() for o in row["options"]]
    if len(opts) == 2 and any(o in YES_WORDS for o in opts) \
            and any(o in NO_WORDS for o in opts):
        return "noul"
    return "choice"


def normalize(row):
    """Stamp `type`, and put noul rows in true/false order with true first.

    Returns a new row; the answer index and soft label are permuted with the
    options, so a normalized row means exactly what the original did.
    """
    row = dict(row)
    kind = kind_of(row)
    row["type"] = kind
    if kind == "score":
        row["ordered"] = True
        return row
    if kind != "noul":
        return row

    opts = [str(o).strip().lower() for o in row["options"]]
    yes_at = next((i for i, o in enumerate(opts) if o in YES_WORDS), None)
    no_at = next((i for i, o in enumerate(opts) if o in NO_WORDS), None)
    if yes_at is None or no_at is None or yes_at == no_at:
        row["type"] = "choice"          # not actually a yes/no pair
        return row

    perm = [yes_at, no_at]
    criteria = row.get("criteria")
    if isinstance(criteria, dict) and criteria:
        row["criteria"] = {TRUE_FALSE[j]: criteria.get(row["options"][i])
                           for j, i in enumerate(perm)
                           if criteria.get(row["options"][i])}
    row["options"] = list(TRUE_FALSE)
    if "label" in row:
        row["label"] = [row["label"][i] for i in perm]
    if "answer" in row:
        row["answer"] = perm.index(row["answer"])
    return row


def criterion_for(row, option, position):
    """The description of one option: dict lookup for choice/noul, positional for score."""
    criteria = row.get("criteria")
    if not criteria:
        return None
    if isinstance(criteria, dict):
        return criteria.get(option)
    if isinstance(criteria, list) and position < len(criteria):
        # A score's levels and its criteria are the same ordered list; only
        # index into it when the row kept them apart.
        return criteria[position] if criteria[position] != option else None
    return None


def option_lines(row, options=None):
    """A. billing — Payments, invoicing, refunds"""
    options = row["options"] if options is None else options
    if len(options) > len(LETTERS):
        raise ValueError(f"max {len(LETTERS)} options, got {len(options)}")
    lines = []
    for i, option in enumerate(options):
        desc = criterion_for(row, option, i)
        lines.append(f"{LETTERS[i]}. {option}" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def question_block(row, options=None):
    """Everything after the state: the question, its options, their criteria.

    The header tells the model which primitive it is looking at. A score is the
    one case where option ORDER carries meaning, so it says so out loud.
    """
    header = ("Levels, lowest to highest:" if kind_of(row) == "score"
              else "Options:")
    return f"Question: {row['question']}\n{header}\n{option_lines(row, options)}"


def build_prompt(row, options=None):
    """The full teacher-facing prompt for one row."""
    return (f"{PREFIX}{row['state']}\n</state>\n\n{question_block(row, options)}\n\n"
            "Reply with exactly one option letter and nothing else.\nAnswer:")


# ------------------------------------------------------------------ answers

def answer(row, probs):
    """Shape a probability distribution into the API's answer for this type."""
    kind = kind_of(row)
    options = row["options"]
    if kind == "noul":
        return {"type": "noul", "noul": round(probs[0], 4)}

    confidence = round(1.0 - normalized_entropy(probs), 4)
    if kind == "score":
        return {"type": "score",
                "score": round(sum(i * p for i, p in enumerate(probs)), 4),
                "legend": {str(i): o for i, o in enumerate(options)},
                "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
                "confidence": confidence}
    return {"type": "choice",
            "choice": options[max(range(len(probs)), key=probs.__getitem__)],
            "probabilities": {o: round(p, 4) for o, p in zip(options, probs)},
            "confidence": confidence}


def normalized_entropy(probs):
    """0 = certain, 1 = uniform. Divided by log(n) so option counts compare."""
    import math
    n = len(probs)
    if n < 2:
        return 0.0
    h = -sum(p * math.log(p) for p in probs if p > 1e-12)
    return h / math.log(n)

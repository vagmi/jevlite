#!/usr/bin/env python3
"""
build_data.py — turn existing datasets into jev_lite training data.

Output rows match what jev_lite.py expects:
  {"type": ..., "state": ..., "question": ..., "options": [...], "answer": idx}
  plus "criteria" where the label set has a meaning worth stating, "ordered":
  true for score questions, and "task"/"source" for analysis. primitives.py
  decides the type and renders it; see its docstring for the three shapes.

The point of this file is TASK DIVERSITY. A model that has seen 500 different
kinds of question generalizes to a new one; a model that has seen 500 examples
of one question does not. So the eval split holds out WHOLE TASKS, never just
rows: that is the only way to measure "can it answer a question it never saw".

Subcommands
  sni    Super-NaturalInstructions (~1600 tasks) — the diversity backbone.
         Streams task files from GitHub; no clone needed.
  hf     MNLI / ANLI / BoolQ / RACE / Yelp / SST-2 via `datasets`.
  mix    Combine built files into one training set, soft labels winning.
  stats  Summarize a built file.

Rows come out typed as TypeSafe primitives: yes/no label sets become `noul`
(true first, so the reported probability is the one for true), scales become
`score`, everything else `choice`. SNI ships bare labels with no description,
so run `teacher.py criteria` over its output to fill those in.

Usage
  python build_data.py sni --out-dir data/ --max-per-task 40 --max-tasks 400
  python build_data.py hf  --out-dir data/ --sets mnli,boolq,race,yelp
  python build_data.py mix --out data/train.jsonl \\
      --sources data/sni.train.described.jsonl data/hf.train.jsonl \\
                data/synth.labelled.jsonl --overlay data/real.labelled.jsonl
  python build_data.py stats --input data/train.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import primitives

RAW = "https://raw.githubusercontent.com/allenai/natural-instructions/master"
MAX_OPTIONS = 25          # keep option lists short enough to reason over
MAX_OPTION_CHARS = 70     # an "option" longer than this is free-text, not a class
MAX_STATE_CHARS = 24_000  # ~6k tokens; long enough to matter, short enough to train


# ------------------------------------------------------------------ helpers

def fetch(path, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{RAW}/{path}", timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception:
            if attempt == retries - 1:
                raise
    return None


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def fingerprint(row):
    """Identity of an example, for dedupe across sources."""
    key = norm(row["state"])[:500] + "||" + norm(row["question"])
    return hashlib.sha1(key.encode()).hexdigest()


def clip(text, limit=MAX_STATE_CHARS):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...\n" + text[-(limit // 2):]


def make_row(state, question, options, answer, task, source, ordered=False,
             criteria=None):
    """Build and validate one row; returns None if it isn't usable.

    Normalizing here is what stamps the primitive type and puts yes/no rows
    into the true/false order the noul answer is reported in.
    """
    state, question = clip(state), norm(question)
    options = [norm(o) for o in options]
    if not state or not question or not (2 <= len(options) <= MAX_OPTIONS):
        return None
    if len(set(options)) != len(options):
        return None
    if any(not o or len(o) > MAX_OPTION_CHARS for o in options):
        return None
    if not 0 <= answer < len(options):
        return None
    row = {"state": state, "question": question, "options": options,
           "answer": answer, "task": task, "source": source}
    if ordered:
        row["ordered"] = True
    if criteria:
        row["criteria"] = {norm(k): norm(v) for k, v in criteria.items()}
    return primitives.normalize(row)


def write_splits(rows, out_dir, prefix, eval_frac, rng):
    """Split by TASK, not by row, so eval questions are genuinely unseen."""
    os.makedirs(out_dir, exist_ok=True)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    tasks = sorted(by_task)
    rng.shuffle(tasks)
    n_eval = max(1, int(len(tasks) * eval_frac)) if len(tasks) > 1 else 0
    eval_tasks = set(tasks[:n_eval])

    seen = set()
    counts = {"train": 0, "eval": 0}
    paths = {s: os.path.join(out_dir, f"{prefix}.{s}.jsonl") for s in counts}
    files = {s: open(p, "w") for s, p in paths.items()}
    try:
        for task in tasks:
            split = "eval" if task in eval_tasks else "train"
            for r in by_task[task]:
                fp = fingerprint(r)
                if fp in seen:        # same state+question twice = leakage
                    continue
                seen.add(fp)
                files[split].write(json.dumps(r, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for f in files.values():
            f.close()

    print(f"\n{prefix}: {counts['train']} train rows / {len(tasks) - n_eval} tasks, "
          f"{counts['eval']} eval rows / {n_eval} held-out tasks")
    for p in paths.values():
        print(f"  {p}")


# ----------------------------------------------- Super-NaturalInstructions

def letter_labels(options):
    """True when the labels are bare letters (A/B/C/D, (a), b. ...).

    Those tasks keep their real choices inside the input text and use the
    letter only as a pointer, so there is no option set to describe and the
    letters collide with our own A/B/C lettering. Not a choice question.
    """
    return all(re.fullmatch(r"\(?[A-Za-z][.)]?", str(o).strip()) for o in options)


def classification_options(instances, probe=400):
    """Decide whether a task is classification, and what its label set is.

    A task qualifies when its outputs form a small, repeating, short set —
    that is what separates "pick a label" from "write an answer".
    """
    outs = []
    for inst in instances[:probe]:
        out = inst.get("output") or []
        if len(out) != 1:       # several acceptable answers => free-text
            return None
        outs.append(norm(out[0]))

    counts = Counter(outs)
    if not (2 <= len(counts) <= MAX_OPTIONS):
        return None
    if any(len(o) > MAX_OPTION_CHARS for o in counts):
        return None
    # Every label must recur, or it's free-text that happens to be short.
    if sum(1 for c in counts.values() if c >= 2) < len(counts):
        return None
    # Guard against a near-constant label (a model can win by always guessing it).
    if max(counts.values()) / len(outs) > 0.9:
        return None
    return sorted(counts)


def convert_sni_task(name, max_per_task, rng):
    try:
        data = json.loads(fetch(f"tasks/{name}.json"))
    except Exception as e:
        return name, None, f"fetch failed: {type(e).__name__}"

    instances = data.get("Instances") or []
    if len(instances) < 8:
        return name, None, "too few instances"

    options = classification_options(instances)
    if options is None:
        return name, None, "not classification"
    if letter_labels(options):
        return name, None, "letter labels"

    definition = norm(" ".join(data.get("Definition") or []))
    if not definition:
        return name, None, "no definition"
    category = (data.get("Categories") or ["unknown"])[0]

    # The task definition IS the question: that is what makes it zero-shot.
    # Strip the part that tells a text model how to format its answer.
    question = re.split(r"(?i)\b(your answer should|output should|return only)\b",
                        definition)[0].strip() or definition

    picked = rng.sample(instances, min(max_per_task, len(instances)))
    rows = []
    for inst in picked:
        answer = norm(inst["output"][0])
        if answer not in options:
            continue
        row = make_row(inst["input"], question, options, options.index(answer),
                       task=name, source="sni")
        if row:
            row["category"] = category
            rows.append(row)

    if len(rows) < 4:
        return name, None, "too few valid rows"
    return name, rows, None


def cmd_sni(args):
    rng = random.Random(args.seed)
    names = []
    for split_file in ("train_tasks.txt", "test_tasks.txt"):
        try:
            names += [n.strip() for n in
                      fetch(f"splits/default/{split_file}").splitlines() if n.strip()]
        except Exception as e:
            print(f"warning: could not read {split_file}: {e}", file=sys.stderr)

    if not names:
        sys.exit("Could not list tasks. Check network access to raw.githubusercontent.com")

    names = sorted(set(names))
    rng.shuffle(names)
    if args.max_tasks:
        names = names[: args.max_tasks]
    print(f"scanning {len(names)} tasks with {args.workers} workers...")

    rows, kept, reasons = [], 0, Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_sni_task, n, args.max_per_task, rng) for n in names]
        for i, fut in enumerate(futures, 1):
            name, task_rows, why = fut.result()
            if task_rows:
                rows += task_rows
                kept += 1
            else:
                reasons[why] += 1
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} scanned, {kept} kept, {len(rows)} rows",
                      flush=True)

    print("\nskipped:", dict(reasons))
    if not rows:
        sys.exit("No usable tasks found.")
    write_splits(rows, args.out_dir, "sni", args.eval_frac, rng)


# ------------------------------------------------------------ HF adapters
# Each adapter yields (state, question, options, answer_idx, ordered, criteria).
#
# The criteria are written out by hand here because these label sets are fixed
# and well understood — no point paying a teacher to describe "entailment".
# RACE is the deliberate exception: its options ARE the candidate answers, so
# there is nothing to describe, and those rows teach the criteria-free case the
# API also allows.

NLI_CRITERIA = {
    "entailment": "The premise makes the hypothesis true.",
    "neutral": "The premise neither proves nor disproves the hypothesis.",
    "contradiction": "The premise makes the hypothesis false.",
}

def _mnli(split, lim):
    from datasets import load_dataset
    labels = ["entailment", "neutral", "contradiction"]
    q = ("Given the premise, is the hypothesis true, undetermined, or false? "
         "Answer entailment, neutral, or contradiction.")
    for ex in load_dataset("nyu-mll/glue", "mnli", split=split).select(range(lim)):
        state = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}"
        yield state, q, labels, ex["label"], False, NLI_CRITERIA


def _anli(split, lim):
    from datasets import load_dataset
    labels = ["entailment", "neutral", "contradiction"]
    q = "Does the premise entail, leave undetermined, or contradict the hypothesis?"
    for ex in load_dataset("facebook/anli", split=split).select(range(lim)):
        state = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}"
        yield state, q, labels, ex["label"], False, NLI_CRITERIA


def _boolq(split, lim):
    from datasets import load_dataset
    criteria = {"yes": "The passage supports the question.",
                "no": "The passage does not support the question."}
    for ex in load_dataset("google/boolq", split=split).select(range(lim)):
        yield (ex["passage"], ex["question"] + "?", ["yes", "no"],
               0 if ex["answer"] else 1, False, criteria)


def _race(split, lim):
    from datasets import load_dataset
    for ex in load_dataset("ehovy/race", "all", split=split).select(range(lim)):
        # No criteria: the options are the candidate answers themselves.
        yield (ex["article"], ex["question"], ex["options"],
               "ABCD".index(ex["answer"]), False, None)


def _yelp(split, lim):
    from datasets import load_dataset
    # A score's levels are its own criteria, so they describe the rating.
    stars = ["1 star — hated it", "2 stars — poor", "3 stars — mixed",
             "4 stars — good", "5 stars — loved it"]
    q = "How many stars out of 5 does this review give?"
    for ex in load_dataset("Yelp/yelp_review_full", split=split).select(range(lim)):
        yield ex["text"], q, stars, ex["label"], True, None  # ordered: never shuffled


def _sst2(split, lim):
    from datasets import load_dataset
    q = "Is the sentiment of this sentence positive or negative?"
    criteria = {"negative": "The sentence expresses dislike or criticism.",
                "positive": "The sentence expresses approval or praise."}
    for ex in load_dataset("stanfordnlp/sst2", split=split).select(range(lim)):
        yield ex["sentence"], q, ["negative", "positive"], ex["label"], False, criteria


HF_SETS = {
    "mnli":  (_mnli,  "train"),
    "anli":  (_anli,  "train_r3"),
    "boolq": (_boolq, "train"),
    "race":  (_race,  "train"),
    "yelp":  (_yelp,  "train"),
    "sst2":  (_sst2,  "train"),
}


def cmd_hf(args):
    rng = random.Random(args.seed)
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in HF_SETS]
    if unknown:
        sys.exit(f"Unknown sets {unknown}. Available: {sorted(HF_SETS)}")

    rows = []
    for name in wanted:
        fn, split = HF_SETS[name]
        print(f"loading {name}...", flush=True)
        try:
            n = 0
            for state, q, options, answer, ordered, criteria in fn(split, args.max_per_set):
                row = make_row(state, q, options, answer, task=name, source="hf",
                               ordered=ordered, criteria=criteria)
                if row:
                    rows.append(row)
                    n += 1
            print(f"  {name}: {n} rows")
        except Exception as e:
            print(f"  {name} failed: {type(e).__name__}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("Nothing loaded. Is `datasets` installed and the Hub reachable?")
    # Only a handful of tasks here, so split by row; SNI supplies task diversity.
    write_splits(rows, args.out_dir, "hf", args.eval_frac if len(wanted) > 2 else 0.0, rng)


# -------------------------------------------------------------------- mix

def cmd_mix(args):
    """Combine built files into one training set, soft labels winning.

    A row labelled by teacher.py carries the `_key` of the row it came from,
    so the labelled sample lands back on top of its own source rows instead of
    beside them — otherwise the same question trains twice, once calibrated
    and once as a 0/1 target, and the hard copy undoes the soft one.
    """
    rng = random.Random(args.seed)

    overlay = {}
    for path in args.overlay or []:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                if row.get("_key"):
                    overlay[row["_key"]] = row
    if overlay:
        print(f"{len(overlay)} labelled rows to overlay")

    rows, replaced = [], 0
    for path in args.sources:
        n = 0
        for i, line in enumerate(open(path)):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("_key") or f"{path}#{i}"
            if key in overlay:
                row = overlay.pop(key)
                replaced += 1
            rows.append(primitives.normalize(row))
            n += 1
        print(f"  {path}: {n} rows")
    rows += list(overlay.values())     # labelled rows whose source isn't listed

    if args.drop_flagged:
        before = len(rows)
        rows = [r for r in rows if not r.get("flags")]
        print(f"dropped {before - len(rows)} flagged rows")

    if args.max_per_task:
        kept, seen_task = [], Counter()
        for row in sorted(rows, key=lambda r: rng.random()):
            task = row.get("task", "?")
            if seen_task[task] < args.max_per_task:
                seen_task[task] += 1
                kept.append(row)
        print(f"capped at {args.max_per_task}/task: {len(rows)} -> {len(kept)} rows")
        rows = kept

    held = []
    if args.holdout:
        # Group by STATE: every question about one state goes to the same side,
        # or the eval set is answering questions about text it trained on.
        by_state = defaultdict(list)
        for row in rows:
            by_state[row["state"]].append(row)
        states = sorted(by_state)
        rng.shuffle(states)
        n_out = int(len(states) * args.holdout)
        out_states = set(states[:n_out])
        held = [r for s in out_states for r in by_state[s]]
        rows = [r for r in rows if r["state"] not in out_states]
        print(f"held out {len(held)} rows from {n_out} states")

    seen, out = set(), []
    for row in rows:
        fp = fingerprint(row)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(row)

    rng.shuffle(out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{len(out)} rows ({replaced} replaced by labelled versions, "
          f"{len(rows) - len(out)} duplicates dropped) -> {args.out}")

    if held:
        rng.shuffle(held)
        with open(args.holdout_out, "w") as f:
            for row in held:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(held)} held-out rows -> {args.holdout_out}")


# ------------------------------------------------------------------ stats

def cmd_stats(args):
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if not rows:
        sys.exit("empty file")

    tasks = Counter(r.get("task", "?") for r in rows)
    n_opts = Counter(len(r["options"]) for r in rows)
    soft = sum(1 for r in rows if "label" in r)
    chars = sorted(len(r["state"]) for r in rows)
    kinds = Counter(primitives.kind_of(r) for r in rows)
    # A score's levels describe themselves, so they always count as described.
    described = sum(1 for r in rows
                    if r.get("criteria") or primitives.kind_of(r) == "score")

    print(f"rows           {len(rows)}")
    print(f"distinct tasks {len(tasks)}")
    print(f"primitives     {dict(kinds)}")
    print(f"with criteria  {described} ({100 * described / len(rows):.1f}%)")
    print(f"soft-labelled  {soft} ({100 * soft / len(rows):.1f}%)")
    print(f"sources        {dict(Counter(r.get('source', '?') for r in rows))}")
    print(f"options/row    {dict(sorted(n_opts.items()))}")
    print(f"state chars    p50 {chars[len(chars) // 2]}  "
          f"p95 {chars[int(len(chars) * 0.95)]}  max {chars[-1]}")

    # A skewed answer index means the model can cheat by always picking one slot.
    idx = Counter(r["answer"] for r in rows if "answer" in r)
    if idx:
        top = max(idx.values()) / sum(idx.values())
        print(f"answer index   {dict(sorted(idx.items()))}  (top slot {100 * top:.0f}%)")
        if top > 0.5:
            print("  note: skewed — rely on jev_lite's option shuffling")
    print(f"biggest tasks  {tasks.most_common(5)}")


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out-dir", default="data")
        p.add_argument("--eval-frac", type=float, default=0.1,
                       help="fraction of TASKS held out for eval")
        p.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("sni", help="Super-NaturalInstructions from GitHub")
    common(s)
    s.add_argument("--max-per-task", type=int, default=40)
    s.add_argument("--max-tasks", type=int, default=0, help="0 = all ~1600")
    s.add_argument("--workers", type=int, default=8)

    h = sub.add_parser("hf", help="classic datasets via `datasets`")
    common(h)
    h.add_argument("--sets", default="mnli,boolq,race,yelp,sst2")
    h.add_argument("--max-per-set", type=int, default=2000)

    m = sub.add_parser("mix", help="combine built files into one training set")
    m.add_argument("--sources", nargs="+", required=True)
    m.add_argument("--overlay", nargs="*",
                   help="labelled files that replace their source rows by _key")
    m.add_argument("--out", required=True)
    m.add_argument("--drop-flagged", action="store_true",
                   help="drop rows teacher.py flagged ambiguous or gold-disagreeing")
    m.add_argument("--max-per-task", type=int, default=0,
                   help="cap rows per task so one big dataset can't dominate")
    m.add_argument("--holdout", type=float, default=0.0,
                   help="fraction of STATES to hold out (for sources split by state,"
                        " like synth; task-split sources already have their own eval)")
    m.add_argument("--holdout-out", default="holdout.jsonl")
    m.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("stats", help="summarize a built file")
    t.add_argument("--input", required=True)

    args = ap.parse_args()
    {"sni": cmd_sni, "hf": cmd_hf, "mix": cmd_mix, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()

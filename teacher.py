#!/usr/bin/env python3
"""
teacher.py — synthesize states/questions, and soft-label them with a teacher LLM.

Three stages, run in order:

  synth     Invent realistic program states (tickets, invoices, alerts, agent runs)
            and several typed questions about each. Produces UNLABELLED rows,
            one per TypeSafe primitive: choice, score, noul.
  criteria  Describe what each option means, for rows built from datasets that
            ship bare labels. The API always sends criteria, so the student has
            to be trained on them. Costs one call per distinct question, not
            per row.
  label     Ask a teacher model each question and record the full probability
            distribution as a soft label. Works on synth output OR on real rows
            from build_data.py (there it also cross-checks the gold answer).

Why soft labels: a 0/1 target teaches the model to be certain. Vote shares or
teacher probabilities teach it that "70% billing, 30% security" is the honest
answer — which is the whole point of a calibrated decision model.

Teacher: any OpenAI-compatible /chat/completions endpoint (llama.cpp, vLLM,
Ollama, Together, OpenAI). Prefer an open-weight teacher: most hosted providers
forbid using outputs to train competing models. Check your terms.

Reasoning: hybrid models (Qwen3.5/3.6) answer better when allowed to think, so
--think-budget lets them reason for a bounded number of tokens. `logprobs` mode
reads the distribution AT THE ANSWER TOKEN, after the reasoning closes — so the
probability is conditioned on the reasoning, and a letter mentioned mid-thought
is never mistaken for the answer.

ALWAYS run `doctor` first. It catches the failures that otherwise show up as an
empty output file hours later.

Usage
  # llama.cpp serving Qwen3.6-35B-A3B on a 24GB card
  llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL \
      --n-gpu-layers 99 --flash-attn on --ctx-size 16384 \
      --cache-type-k q8_0 --cache-type-v q8_0 --parallel 4

  export TEACHER_BASE=http://localhost:8080/v1 TEACHER_KEY=none
  M="--model qwen3.6-35b-a3b --think-budget 256"
  python teacher.py doctor $M
  python teacher.py synth $M --out raw.jsonl --n-states 200
  python teacher.py criteria $M --input data/sni.train.jsonl \
      --out data/sni.described.jsonl --cache data/criteria.json
  python teacher.py label $M --input raw.jsonl --out labelled.jsonl --debias
  python teacher.py label $M --input data/sni.train.jsonl --out sni.soft.jsonl \
      --debias --drop-flagged --resume
"""
import argparse
import json
import math
import os
import random
import re
import sys
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import primitives
from primitives import LETTERS

# Forcing variety is the whole game: a teacher left to its own devices writes
# the same support ticket 200 times. We sample a fresh combination per call.
DOMAINS = [
    ("a customer support ticket", ["billing dispute", "bug report", "feature request",
                                   "cancellation threat", "account access problem",
                                   "shipping delay", "refund request", "praise"]),
    ("a vendor invoice with line items", ["duplicate charge", "wrong tax rate",
                                          "missing PO number", "currency mismatch",
                                          "early-payment discount", "clean invoice"]),
    ("a security alert from a SIEM", ["impossible travel login", "credential stuffing",
                                      "data exfiltration", "benign admin activity",
                                      "expired certificate", "port scan"]),
    ("a transcript of an AI agent completing a task", ["task fully completed",
                                                       "test deleted instead of fixed",
                                                       "gave up partway", "wrong file edited",
                                                       "succeeded but left debug code"]),
    ("a code review diff with comments", ["SQL injection introduced", "style nitpicks only",
                                          "breaking API change", "missing test coverage",
                                          "good refactor"]),
    ("a production incident log", ["database connection pool exhausted", "bad deploy rollback",
                                   "third-party API outage", "slow memory leak", "false alarm"]),
    ("an internal chat thread between coworkers", ["scope disagreement", "deadline slip",
                                                   "handoff to another team",
                                                   "decision reached", "off-topic banter"]),
    ("a job applicant's cover letter and resume summary", ["strong fit", "career changer",
                                                           "overqualified", "keyword stuffing",
                                                           "gap in employment"]),
]
TONES = ["terse and technical", "rambling and emotional", "polite and formal",
         "frustrated and blunt", "ambiguous and incomplete", "cheerful",
         "written by a non-native speaker", "full of jargon and acronyms"]
LENGTHS = ["3-4 sentences", "a short paragraph", "two paragraphs with specifics",
           "a long detailed account with timestamps and IDs"]

SYNTH_PROMPT = """Write {article} about: {flavor}.
Style: {tone}. Length: {length}. Invent concrete names, IDs, dates and numbers.

Then write {n_q} questions that software would ask about it to decide what to do next.
Each question is one of three types:
- "noul"   is this true? Criteria describe what true and what false look like.
- "choice" pick one named option. Criteria give one short line per option.
- "score"  rate against ordered levels. Write the levels lowest first; each level
           IS its own criterion, so describe the level, don't just number it.

Rules:
- Each must be answerable from the state alone.
- Use each type at least twice, so all three are well represented.
- Criteria are one short line each and must never give away the answer.
- Make at least one question genuinely hard, where a reasonable person might hesitate.
- Do not include the answers.

Return ONLY JSON:
{{"state": "...", "questions": [
  {{"type": "noul", "question": "...",
    "criteria": {{"true": "...", "false": "..."}}}},
  {{"type": "choice", "question": "...",
    "criteria": {{"option name": "what this option means", "other option": "..."}}}},
  {{"type": "score", "question": "...",
    "levels": ["lowest level described", "middle", "highest level described"]}}
]}}"""


class Teacher:
    """Minimal OpenAI-compatible client (stdlib only)."""

    def __init__(self, base, key, model, timeout=180, extra=None):
        self.url = base.rstrip("/") + "/chat/completions"
        self.key = key
        self.model = model
        self.timeout = timeout
        self.extra = extra or {}
        self.calls = 0
        self._lock = threading.Lock()

    def chat(self, prompt, max_tokens=1200, temperature=0.0, logprobs=False, retries=4):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 20
        body.update(self.extra)   # thinking toggles, sampler overrides, etc.
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.key}"}

        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                with self._lock:
                    self.calls += 1
                return out["choices"][0]
            except Exception as e:
                last = e
                if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 404):
                    raise  # config error: retrying won't help
                threading.Event().wait(2 ** attempt)
        raise RuntimeError(f"teacher call failed after {retries} tries: {last}")


def parse_json(text):
    """Teachers wrap JSON in prose and code fences more often than not."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def stream_out(path, resume):
    """Append-only writer; returns (file, set_of_keys_already_done)."""
    done = set()
    if resume and os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("_key"))
                except json.JSONDecodeError:
                    continue
        print(f"resuming: {len(done)} rows already in {path}")
    return open(path, "a" if resume else "w"), done


# ------------------------------------------------------------------- synth

def synth_one(teacher, rng, n_q):
    article, flavors = rng.choice(DOMAINS)
    prompt = SYNTH_PROMPT.format(article=article, flavor=rng.choice(flavors),
                                 tone=rng.choice(TONES), length=rng.choice(LENGTHS),
                                 n_q=n_q)
    choice = teacher.chat(prompt, max_tokens=2600, temperature=1.0)
    obj = parse_json(choice["message"]["content"])
    if not obj or not obj.get("state") or not obj.get("questions"):
        return []

    rows = []
    for q in obj["questions"]:
        row = synth_row(q, str(obj["state"]), article)
        if row:
            rows.append(row)
    return rows


def synth_row(q, state, article):
    """Turn one generated question into a row, or None if it is malformed.

    The three types differ only in where the options come from: a noul's are
    fixed, a choice's are the keys of its criteria, a score's are its levels.
    """
    kind = str(q.get("type", "")).strip().lower()
    criteria = q.get("criteria")
    question = str(q.get("question", "")).strip()
    if not question:
        return None

    if kind == "score":
        levels = q.get("levels") or q.get("options")
        if not isinstance(levels, list):
            return None
        options, criteria = [str(o).strip() for o in levels], None
    elif kind == "noul":
        if not isinstance(criteria, dict):
            return None
        criteria = {k: str(v).strip() for k, v in criteria.items()
                    if str(k).strip().lower() in ("true", "false") and str(v).strip()}
        if len(criteria) != 2:
            return None
        options = list(primitives.TRUE_FALSE)
    elif kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            return None
        criteria = {str(k).strip(): str(v).strip() for k, v in criteria.items()}
        options = list(criteria)
    else:
        return None

    if not 2 <= len(options) <= 25:
        return None
    if len(set(options)) != len(options) or any(not o for o in options):
        return None

    row = {"type": kind, "state": state, "question": question, "options": options,
           "task": f"synth:{article}", "source": "synth"}
    if criteria:
        row["criteria"] = criteria
    if kind == "score":
        row["ordered"] = True
    return primitives.normalize(row)


def cmd_synth(args):
    teacher = Teacher(args.base, args.key, args.model, extra=parse_extra(args))
    rng = random.Random(args.seed)
    out, done = stream_out(args.out, args.resume)
    lock = threading.Lock()
    written = [0]

    def work(i):
        try:
            rows = synth_one(teacher, random.Random(args.seed * 100003 + i), args.questions)
        except Exception as e:
            print(f"  state {i} failed: {type(e).__name__}: {e}", file=sys.stderr)
            return
        with lock:
            for j, row in enumerate(rows):
                row["_key"] = f"synth-{i}-{j}"
                if row["_key"] in done:
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written[0] += 1
            out.flush()
            if i % 20 == 0:
                print(f"  {i}/{args.n_states} states, {written[0]} questions", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, range(args.n_states)))
    finally:
        out.close()
    print(f"\nwrote {written[0]} unlabelled rows to {args.out} ({teacher.calls} teacher calls)")
    print(f"next: python teacher.py label --input {args.out} --out labelled.jsonl")


# ------------------------------------------------------------------- label

def build_prompt(row, options=None):
    """Delegated to primitives.py so the teacher judges the prompt the student
    will be trained and served on — criteria included."""
    return primitives.build_prompt(row, options)


def dist_from_logprobs(choice, n):
    """Read probabilities at the ANSWER token, after any reasoning.

    Letting the model reason and then reading the distribution where it commits
    to a letter gives a probability conditioned on that reasoning — better
    calibrated than snap-judging token 0, and it works with thinking left on.
    """
    try:
        content = choice["logprobs"]["content"]
    except (KeyError, TypeError):
        return None
    if not content:
        return None

    # Hit the token cap: whatever it was saying, it never got to the answer.
    if choice.get("finish_reason") == "length":
        return None

    # The reasoning tokens sit in this same stream, ahead of the answer, so the
    # letter must be read after the closing tag — never before it.
    start, closed = 0, False
    for i, item in enumerate(content):
        if re.search(r"</(think|reasoning)>", item.get("token", ""), re.I):
            start, closed = i + 1, True   # keep going: take the LAST close

    # Did this reply reason at all? Servers signal it two ways: a separate
    # reasoning_content field (llama.cpp, vLLM) or an inline opening tag. The
    # opening tag is often injected by the chat template and never echoed back,
    # so the field is the reliable signal.
    reasoned = bool((choice.get("message") or {}).get("reasoning_content")) or \
        any(re.search(r"<(think|reasoning)>", it.get("token", ""), re.I)
            for it in content[:4])
    if reasoned and not closed:
        # Cut off mid-thought: every letter below is reasoning, not an answer.
        return None

    for item in content[start:]:
        tok = item.get("token", "").strip().upper()
        if len(tok) != 1 or tok not in LETTERS[:n]:
            continue
        probs = [0.0] * n
        for alt in item.get("top_logprobs") or []:
            a = alt["token"].strip().upper()
            if len(a) == 1 and a in LETTERS[:n]:
                probs[LETTERS.index(a)] += math.exp(alt["logprob"])
        total = sum(probs)
        return [p / total for p in probs] if total > 1e-6 else None
    return None


def extract_letter(text, n):
    """First standalone option letter, ignoring any reasoning preamble.

    An UNTERMINATED <think> means the reply was cut off mid-thought, so there is
    no answer yet. Returning a letter found inside the reasoning would silently
    invent a label, so we return None and let the row count as no_valid_answer.
    """
    text = re.sub(r"<(think|reasoning)>.*?</\1>", " ", str(text), flags=re.S | re.I)
    if re.search(r"<(think|reasoning)>", text, re.I):
        return None          # truncated mid-reasoning; raise --vote-tokens
    m = re.search(rf"\b([{LETTERS[:n]}])\b", text.upper())
    return LETTERS.index(m.group(1)) if m else None


def dist_from_votes(teacher, prompt, n, samples, temperature, max_tokens=4):
    votes = Counter()
    for _ in range(samples):
        choice = teacher.chat(prompt, max_tokens=max_tokens, temperature=temperature)
        idx = extract_letter(choice["message"]["content"], n)
        if idx is not None:
            votes[idx] += 1
    total = sum(votes.values())
    if not total:
        return None
    return [votes[i] / total for i in range(n)]


def teacher_dist(teacher, args, row):
    """One distribution over the row's options, optionally debiased."""
    options = row["options"]

    def once(opts):
        prompt = build_prompt(row, opts)
        budget = answer_tokens(args)
        if args.mode == "logprobs":
            return dist_from_logprobs(
                teacher.chat(prompt, budget, 0.0, logprobs=True), len(opts))
        return dist_from_votes(teacher, prompt, len(opts), args.samples,
                               args.temperature, budget)

    forward = once(options)
    if forward is None:
        return None
    # A score's levels run low to high and the prompt says so; reversing them
    # asks a different question, so scales are never debiased this way.
    if not args.debias or primitives.kind_of(row) == "score":
        return forward
    # Same question, options reversed: averaging cancels most position bias.
    back = once(list(reversed(options)))
    if back is None:
        return forward
    back = list(reversed(back))
    return [(a + b) / 2 for a, b in zip(forward, back)]


def entropy_ratio(p):
    """0 = certain, 1 = uniform. Normalized so option counts stay comparable."""
    n = len(p)
    if n < 2:
        return 0.0
    h = -sum(x * math.log(x) for x in p if x > 1e-12)
    return h / math.log(n)


def cmd_label(args):
    teacher = Teacher(args.base, args.key, args.model, extra=parse_extra(args))
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    out, done = stream_out(args.out, args.resume)
    lock = threading.Lock()
    tally = Counter()
    processed = [0]

    def tick():
        """Caller must hold the lock."""
        processed[0] += 1
        if processed[0] % 50 == 0 or processed[0] == len(rows):
            print(f"  {processed[0]}/{len(rows)} labelled", flush=True)

    def work(i_row):
        i, row = i_row
        row = primitives.normalize(row)
        key = row.get("_key") or f"{args.input}#{i}"
        if key in done:
            return
        try:
            probs = teacher_dist(teacher, args, row)
        except Exception as e:
            print(f"  row {i} failed: {type(e).__name__}: {e}", file=sys.stderr)
            with lock:
                tally["error"] += 1
                tick()
            return
        if probs is None:
            with lock:
                tally["no_valid_answer"] += 1
                tick()
            return

        ent = entropy_ratio(probs)
        gold = row.get("answer")
        flags = []
        if gold is not None and max(range(len(probs)), key=probs.__getitem__) != gold:
            flags.append("disagrees_with_gold")
        if ent > args.max_entropy:
            flags.append("ambiguous")

        new = dict(row)
        new.pop("answer", None)
        # Blend toward gold when we have it: keeps the label anchored to truth
        # while the teacher supplies the shape of the uncertainty.
        if gold is not None and args.gold_weight > 0:
            probs = [(1 - args.gold_weight) * p + args.gold_weight * (1.0 if j == gold else 0.0)
                     for j, p in enumerate(probs)]
        new["label"] = [round(p, 5) for p in probs]
        new["_key"] = key
        new["teacher_entropy"] = round(ent, 4)
        if gold is not None:
            new["gold_answer"] = gold
        if flags:
            new["flags"] = flags

        with lock:
            for f in flags:
                tally[f] += 1
            if flags and args.drop_flagged:
                tally["dropped"] += 1
            else:
                tally["kept"] += 1
                out.write(json.dumps(new, ensure_ascii=False) + "\n")
                out.flush()
            tick()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, enumerate(rows)))
    finally:
        out.close()

    print(f"\n{dict(tally)}")
    print(f"{teacher.calls} teacher calls -> {args.out}")
    if tally.get("disagrees_with_gold"):
        pct = 100 * tally["disagrees_with_gold"] / max(len(rows), 1)
        print(f"teacher disagreed with gold on {pct:.1f}% of rows. "
              f"{'Inspect those — they are usually bad gold or a vague question.' if pct > 20 else ''}")


# ----------------------------------------------------------------- criteria

CRITERIA_PROMPT = """A classifier reads some text and must pick one of these labels.

The question it is asked: {question}
The labels: {labels}

Write one short line per label saying what makes that label the right choice.
Describe what the label means — never describe a particular text, never hint
that one label is more likely, and keep each line under 15 words.

Return ONLY JSON with exactly these keys and nothing else: {labels}"""


def criteria_for(teacher, question, options, budget):
    """Ask the teacher what each label means. Returns a dict or None."""
    prompt = CRITERIA_PROMPT.format(question=question, labels=json.dumps(options))
    obj = parse_json(teacher.chat(prompt, max_tokens=max(budget, 400),
                                  temperature=0.0)["message"]["content"])
    if not isinstance(obj, dict):
        return None
    # Accept only a complete answer: a partial map would leave some options
    # described and others bare, which is a hint about which one to pick.
    out = {}
    for o in options:
        v = obj.get(o) or obj.get(str(o).lower()) or obj.get(str(o).title())
        if not isinstance(v, str) or not v.strip():
            return None
        out[o] = norm_line(v)
    return out


def norm_line(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cmd_criteria(args):
    """Attach option descriptions to rows that have none.

    Criteria belong to a TASK, not a row: every row of an SNI task shares one
    question and one label set, so 371 tasks cost 371 calls, not 11k.
    """
    teacher = Teacher(args.base, args.key, args.model, extra=parse_extra(args))
    rows = [json.loads(l) for l in open(args.input) if l.strip()]

    cache = {}
    if args.cache and os.path.exists(args.cache):
        cache = json.load(open(args.cache))
        print(f"loaded {len(cache)} cached criteria from {args.cache}")

    def signature(row):
        return json.dumps([row["question"], row["options"]], ensure_ascii=False)

    todo = []
    for row in map(primitives.normalize, rows):
        if row.get("criteria") or primitives.kind_of(row) == "score":
            continue          # a score's levels already describe themselves
        sig = signature(row)
        if sig not in cache and sig not in todo:
            todo.append(sig)

    print(f"{len(rows)} rows, {len(todo)} distinct questions need criteria")
    budget = answer_tokens(args)
    lock = threading.Lock()
    tally = Counter()

    def work(sig):
        question, options = json.loads(sig)
        try:
            got = criteria_for(teacher, question, options, budget)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}", file=sys.stderr)
            got = None
        with lock:
            tally["written" if got else "failed"] += 1
            if got:
                cache[sig] = got
            n = sum(tally.values())
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} described", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))

    if args.cache:
        json.dump(cache, open(args.cache, "w"), ensure_ascii=False, indent=1)

    kept = 0
    with open(args.out, "w") as f:
        for row in rows:
            row = primitives.normalize(row)
            if not row.get("criteria") and primitives.kind_of(row) != "score":
                got = cache.get(signature(row))
                if got:
                    row["criteria"] = got
                    kept += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{dict(tally)}; {kept}/{len(rows)} rows gained criteria -> {args.out}")


# ------------------------------------------------------------------ doctor

DOCTOR_ROW = {
    "state": "Ticket #9001: I was billed twice for my annual plan and want one charge reversed.",
    "question": "Which team should handle this ticket?",
    "options": ["billing", "technical support", "sales"],
}


def cmd_doctor(args):
    """Ten seconds of checks that save hours of silently-empty output.

    The killer case is a hybrid reasoning model (Qwen3.5/3.6, and others):
    logprobs mode reads the FIRST token, which will be <think>, so every row
    comes back unlabelled. Catch it here, not after a night of labelling.
    """
    teacher = Teacher(args.base, args.key, args.model, extra=parse_extra(args))
    prompt = build_prompt(DOCTOR_ROW)
    n = len(DOCTOR_ROW["options"])
    print(f"endpoint {teacher.url}\nmodel    {teacher.model}")
    if teacher.extra:
        print(f"extra    {json.dumps(teacher.extra)}")
    print()

    budget = answer_tokens(args)
    print(f"budget   {budget} generation tokens\n")

    # 1) reachable at all?
    import time
    t0 = time.time()
    try:
        choice = teacher.chat(prompt, max_tokens=budget, temperature=0.0, logprobs=True)
    except Exception as e:
        sys.exit(f"FAIL  endpoint unreachable: {type(e).__name__}: {e}")
    elapsed = time.time() - t0
    text = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning_content") or ""
    print(f"PASS  endpoint reachable ({elapsed:.1f}s for one call)")
    print(f"      reply: {(reasoning + text)[:160]!r}")

    # 2) reasoning present, and did it finish?
    # finish_reason is the honest signal: a reply the server truncated never
    # reached its answer, however tidy the text that came back looks.
    if choice.get("finish_reason") == "length" or (
            re.search(r"<(think|reasoning)>", text, re.I)
            and not re.search(r"</(think|reasoning)>", text, re.I)):
        print(f"FAIL  reasoning was CUT OFF at {budget} tokens — it never answered.")
        print("      -> raise --max-answer-tokens, or lower --think-budget")
        print("      (rows like this are refused, not guessed — expect no_valid_answer)")
    elif reasoning or re.search(r"</(think|reasoning)>", text, re.I):
        print("PASS  model reasoned and then answered")
    else:
        print("PASS  no reasoning preamble")

    # 3) does it commit to a letter?
    idx = extract_letter(text, n)
    print(f"{'PASS' if idx is not None else 'FAIL'}  letter extraction: "
          f"{DOCTOR_ROW['options'][idx] if idx is not None else 'no letter found'}")

    # 4) probabilities readable at the answer token?
    probs = dist_from_logprobs(choice, n)
    if probs:
        print(f"PASS  logprobs mode works: "
              f"{ {o: round(p, 3) for o, p in zip(DOCTOR_ROW['options'], probs)} }")
    else:
        print("FAIL  no option-letter logprobs found at the answer position.")
        print("      -> server may not return per-token top_logprobs; use --mode vote")

    # 5) position bias: same question, options reversed
    rev_prompt = build_prompt(DOCTOR_ROW, list(reversed(DOCTOR_ROW["options"])))
    if probs:
        back = dist_from_logprobs(teacher.chat(rev_prompt, budget, 0.0, logprobs=True), n)
    else:
        back = dist_from_votes(teacher, rev_prompt, n, 3, 0.8, budget)
        probs = dist_from_votes(teacher, prompt, n, 3, 0.8, budget)
    if probs and back:
        fwd_pick = max(range(n), key=probs.__getitem__)
        back_pick = n - 1 - max(range(n), key=back.__getitem__)
        same = fwd_pick == back_pick
        print(f"{'PASS' if same else 'WARN'}  position bias: "
              f"{'same answer both orders' if same else 'answer FLIPS with option order — use --debias'}")

    # 6) what this will cost at scale
    per_row = elapsed * (2 if args.debias else 1) * (args.samples if args.mode == "vote" else 1)
    print(f"\n{teacher.calls} calls made.")
    print(f"est. {per_row:.1f}s/row single-stream in --mode {args.mode}"
          f"{' --debias' if args.debias else ''}: "
          f"{per_row * 10000 / 3600:.1f}h for 10k rows.")
    print(f"      divide by your server's real concurrency "
          f"(llama.cpp --parallel N, vLLM batching) and use --workers to match.")


def parse_extra(args):
    """Translate --think-budget into whatever the server understands.

    -1  leave the server's default alone
     0  no reasoning at all
    >0  reason, but stop after N tokens (Qwen3.5/3.6 honour reasoning_budget)
    """
    extra = {}
    budget = getattr(args, "think_budget", -1)
    if budget == 0:
        # Belt and braces: servers disagree on which field turns it off, and
        # enable_thinking alone is known not to be enough on llama.cpp.
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        extra["enable_thinking"] = False
        extra["reasoning_budget"] = 0
    elif budget > 0:
        extra["chat_template_kwargs"] = {"enable_thinking": True}
        extra["reasoning_budget"] = budget
    if getattr(args, "extra", None):
        try:
            extra.update(json.loads(args.extra))
        except json.JSONDecodeError as e:
            sys.exit(f"--extra is not valid JSON: {e}")
    return extra


def answer_tokens(args):
    """Room for the reasoning plus the letter that follows it."""
    if getattr(args, "max_answer_tokens", 0):
        return args.max_answer_tokens
    budget = getattr(args, "think_budget", -1)
    if budget == 0:
        return 4
    return (budget if budget > 0 else 512) + 64


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--base", default=os.environ.get("TEACHER_BASE",
                                                        "http://localhost:8000/v1"))
        p.add_argument("--key", default=os.environ.get("TEACHER_KEY", "none"))
        p.add_argument("--model", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--resume", action="store_true", help="append to --out, skip done rows")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--think-budget", type=int, default=-1, metavar="N",
                       help="reasoning tokens before answering: 0 = off, "
                            "N = capped (256 is a good middle), -1 = server default")
        p.add_argument("--max-answer-tokens", type=int, default=0,
                       help="override total generation cap (default: think budget + 64)")
        p.add_argument("--extra", help='raw JSON merged into each request body, e.g. '
                                       '\'{"top_k": 20, "min_p": 0}\'')

    s = sub.add_parser("synth", help="invent states + questions (unlabelled)")
    common(s)
    s.add_argument("--out", required=True)
    s.add_argument("--n-states", type=int, default=200)
    s.add_argument("--questions", type=int, default=5, help="questions per state")

    l = sub.add_parser("label", help="soft-label rows with the teacher")
    common(l)
    l.add_argument("--input", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--mode", choices=["logprobs", "vote"], default="logprobs",
                   help="logprobs = 1 call/row (cheap); vote = N samples (any endpoint)")
    l.add_argument("--samples", type=int, default=8, help="vote mode only")
    l.add_argument("--temperature", type=float, default=0.8, help="vote mode only")
    l.add_argument("--debias", action="store_true",
                   help="also ask with options reversed and average")
    l.add_argument("--max-entropy", type=float, default=0.85,
                   help="above this the question is probably vague, not hard")
    l.add_argument("--gold-weight", type=float, default=0.5,
                   help="blend toward the dataset's gold answer (0 = pure teacher)")
    l.add_argument("--drop-flagged", action="store_true")

    c = sub.add_parser("criteria", help="describe each option, for rows that lack criteria")
    common(c)
    c.add_argument("--input", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cache", help="JSON map of question -> criteria, reused across files")

    d = sub.add_parser("doctor", help="check the endpoint before labelling anything")
    common(d)
    d.add_argument("--mode", choices=["logprobs", "vote"], default="logprobs")
    d.add_argument("--samples", type=int, default=8)
    d.add_argument("--debias", action="store_true")

    args = ap.parse_args()
    {"synth": cmd_synth, "label": cmd_label, "criteria": cmd_criteria,
     "doctor": cmd_doctor}[args.cmd](args)


if __name__ == "__main__":
    main()

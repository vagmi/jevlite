#!/usr/bin/env python3
"""
jev_lite.py — QLoRA fine-tune of Gemma 4 into a "System One"-style decision model.

Input : state + question + options
Output: a probability distribution over the options, read from ONE forward pass.
        No generation, so the answer can't fall outside the option set.

How it works
  The prompt lists options as letters (A, B, C...) and ends with "Answer:".
  We take the next-token logits for just those letter tokens, softmax over them,
  and train with soft-label cross-entropy. Soft labels (e.g. teacher vote shares)
  teach calibration, not just the right answer.

Data format (JSONL, one example per line)
  {"state": "...", "question": "...", "options": ["true", "false"],
   "type": "noul", "answer": 0}
  {"state": "...", "question": "...", "options": ["billing", "tech", "sales"],
   "type": "choice", "criteria": {"billing": "Payments, invoicing, refunds", ...},
   "label": [0.7, 0.2, 0.1]}                                   # soft label
  {"state": "...", "question": "Urgency?", "options": ["Calm", "Annoyed", "Angry"],
   "type": "score", "answer": 2, "ordered": true}              # levels; never shuffled

  `type` and `criteria` are optional and inferred when absent — but the API
  sends criteria, so train with them. primitives.py owns that rendering.

Usage
  pip install -U torch transformers peft bitsandbytes accelerate
  python jev_lite.py train --train train.jsonl --eval eval.jsonl --out adapter/
  python jev_lite.py predict --adapter adapter/ --input questions.jsonl
"""
import argparse
import json
import math
import random

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          get_cosine_schedule_with_warmup)

import primitives
from primitives import LETTERS

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SKIP_MODULES = ("vision", "audio", "multi_modal")  # train the text model only


# ----------------------------------------------------------------------------- data

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def target_dist(ex):
    n = len(ex["options"])
    if "label" in ex:
        t = torch.tensor(ex["label"], dtype=torch.float32)
        assert len(t) == n, f"label length {len(t)} != {n} options"
        return t / t.sum()
    t = torch.zeros(n)
    t[ex["answer"]] = 1.0
    return t


def shuffle_options(options, target, rng):
    """Shuffle option order so the model can't learn 'A is usually right'."""
    perm = list(range(len(options)))
    rng.shuffle(perm)
    return [options[i] for i in perm], target[perm]


class Encoder:
    """Builds token ids for a row and finds the option-letter token ids."""

    def __init__(self, tok, max_len):
        self.tok = tok
        self.max_len = max_len
        self.letter_ids = [self._single_token(c) for c in LETTERS]

    def _single_token(self, c):
        for s in (" " + c, c):
            ids = self.tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        raise ValueError(f"Letter {c!r} is not a single token for this tokenizer")

    def __call__(self, row, options=None):
        """`options` overrides the row's order (shuffling); criteria follow it."""
        suffix = f"\n</state>\n\n{primitives.question_block(row, options)}\nAnswer:"

        p = self.tok.encode(primitives.PREFIX, add_special_tokens=False)
        s = self.tok.encode(row["state"], add_special_tokens=False)
        q = self.tok.encode(suffix, add_special_tokens=False)

        budget = self.max_len - 1 - len(p) - len(q)
        if budget < 0:
            raise ValueError("question + options alone exceed max_len")
        if len(s) > budget:  # too long: keep the head and tail of the state
            head = budget // 2
            s = s[:head] + s[len(s) - (budget - head):]

        ids = [self.tok.bos_token_id] + p + s + q
        n = len(row["options"] if options is None else options)
        return torch.tensor([ids]), self.letter_ids[:n]


# ---------------------------------------------------------------------------- model

def load_base(model_id, attn):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    kwargs = dict(quantization_config=bnb_cfg, dtype=torch.bfloat16,
                  device_map={"": 0}, attn_implementation=attn)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        # Gemma 4 is multimodal; some versions only register the image-text class.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model


def option_logprobs(model, enc, row, options=None):
    """One forward pass -> log-probs over the options only."""
    ids, opt_ids = enc(row, options)
    ids = ids.to(model.device)
    try:
        # Only compute logits for the last position (Gemma's vocab is huge).
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    logits_to_keep=1, use_cache=False)
    except TypeError:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = out.logits[0, -1].float()
    return F.log_softmax(logits[opt_ids], dim=-1)


# ----------------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, enc, rows, bins=10):
    was_training = model.training
    model.eval()
    n = correct = 0
    nll = brier = 0.0
    confs, hits = [], []
    for ex in rows:
        try:
            lp = option_logprobs(model, enc, ex).cpu()
        except ValueError:
            continue
        t = target_dist(ex)
        p = lp.exp()
        hit = float(p.argmax() == t.argmax())
        n += 1
        correct += hit
        nll += -(t * lp).sum().item()
        brier += ((p - t) ** 2).sum().item()
        confs.append(p.max().item())
        hits.append(hit)

    # Expected calibration error: does 80% confidence mean 80% accuracy?
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(hits[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            ece += len(idx) / max(n, 1) * abs(acc - conf)

    if was_training:
        model.train()
    n = max(n, 1)
    return {"n": n, "acc": correct / n, "nll": nll / n, "brier": brier / n, "ece": ece}


# ---------------------------------------------------------------------------- train

def train(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = Encoder(tok, args.max_len)
    # normalize now: stamps `type` and puts noul rows in true/false order, so a
    # file built before either existed still trains as the API will serve it.
    train_rows = [primitives.normalize(r) for r in load_jsonl(args.train)]
    eval_rows = [primitives.normalize(r) for r in load_jsonl(args.eval)] if args.eval else []

    model = load_base(args.model, args.attn)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})

    targets = [name for name, mod in model.named_modules()
               if isinstance(mod, torch.nn.Linear)
               and name.split(".")[-1] in LORA_TARGETS
               and not any(k in name for k in SKIP_MODULES)]
    if not targets:
        raise RuntimeError("No LoRA target modules found; check module names.")

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=targets, bias="none"))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(train_rows) * args.epochs / args.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total_steps)), total_steps)

    model.train()
    step = micro = 0
    running = 0.0

    def optimizer_step():
        nonlocal step, running
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            print(f"step {step}/{total_steps}  loss {running / args.log_every:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
            running = 0.0
        if eval_rows and step % args.eval_every == 0:
            print(f"  eval @ {step}: {evaluate(model, enc, eval_rows)}", flush=True)

    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        for ex in train_rows:
            options, t = ex["options"], target_dist(ex)
            if args.shuffle_options and primitives.kind_of(ex) != "score":
                options, t = shuffle_options(options, t, rng)
            try:
                lp = option_logprobs(model, enc, ex, options)
            except ValueError as e:
                print(f"skipping example: {e}")
                continue
            loss = -(t.to(lp.device) * lp).sum()  # soft-label cross-entropy
            (loss / args.grad_accum).backward()
            running += loss.item() / args.grad_accum
            micro += 1
            if micro % args.grad_accum == 0:
                optimizer_step()
        print(f"epoch {epoch + 1} done")

    if micro % args.grad_accum:
        optimizer_step()

    if eval_rows:
        print(f"final eval: {evaluate(model, enc, eval_rows)}")
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


# -------------------------------------------------------------------------- predict

def predict(args):
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.adapter)
    enc = Encoder(tok, args.max_len)
    model = PeftModel.from_pretrained(load_base(args.model, args.attn), args.adapter)
    model.eval()

    with torch.no_grad():
        for ex in load_jsonl(args.input):
            ex = primitives.normalize(ex)
            p = option_logprobs(model, enc, ex).exp()
            # One answer per row, shaped by its primitive: a noul reports the
            # probability that it is true, a score its expected level.
            out = primitives.answer(ex, p.tolist())
            out["question"] = ex["question"]
            print(json.dumps(out, ensure_ascii=False))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="google/gemma-4-E4B-it")
        p.add_argument("--max-len", type=int, default=8192,
                       help="training context; 4k-16k fits a 24 GB card")
        p.add_argument("--attn", default="sdpa", help="sdpa or flash_attention_2")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--train", required=True)
    t.add_argument("--eval")
    t.add_argument("--out", default="jev-lite-adapter")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--grad-accum", type=int, default=16, help="effective batch size")
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--no-shuffle-options", dest="shuffle_options", action="store_false")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("predict")
    common(p)
    p.add_argument("--adapter", required=True)
    p.add_argument("--input", required=True)

    args = ap.parse_args()
    train(args) if args.cmd == "train" else predict(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
serve.py — the trained adapter behind the TypeSafe System One API, on torch.

Speaks the wire protocol in reference/typesafe-sdk-python, so the official
client talks to this server unmodified. api.py owns that contract; this file
owns the model:

  POST /v1/systemone   answer several questions about one state
  GET  /v1/models      what this server will answer to

  export TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=whatever

Every answer is ONE forward pass over the option letters — the model cannot
answer outside the criteria it was given, so no validation of its output is
needed. Questions in a request are independent: each sees the same state and
nothing of the others.

Rendering goes through primitives.py, the same module that built the training
prompts. That is the whole point: a question arriving here is formatted
exactly as its training rows were, criteria included.

This backend runs the 4-bit NF4 base the adapter was actually trained against,
so its probabilities need no calibration correction. serve_vllm.py trades that
for roughly 3x the throughput and a fitted temperature — see its docstring.

Usage
  pip install fastapi uvicorn
  python serve.py                                   # vagmi/jev-lite from the Hub
  python serve.py --adapter jev-lite-adapter        # or a local training output
  python serve.py --host 0.0.0.0 --port 8000

Auth: set JEV_API_KEY to require `Authorization: Bearer <key>`. Unset, the
server accepts any caller — fine on localhost, not on a network.
"""
import argparse
import threading
import time

import torch
import uvicorn
from starlette.concurrency import run_in_threadpool

import primitives
from api import create_app, resolve_adapter
from jev_lite import Encoder, load_base, option_logprobs


class TorchBackend:
    """The adapter, and the lock that keeps one GPU answering one thing at a time."""

    name = "torch-4bit"

    def __init__(self, base_model, adapter, max_len, attn):
        from peft import PeftModel
        from transformers import AutoTokenizer

        adapter = resolve_adapter(adapter)
        print(f"loading {base_model} + {adapter} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(adapter)
        self.enc = Encoder(tok, max_len)
        self.model = PeftModel.from_pretrained(load_base(base_model, attn), adapter)
        self.model.eval()
        self.lock = threading.Lock()
        print("ready", flush=True)

    @torch.no_grad()
    def _answer(self, rows):
        answers, tokens = [], 0
        with self.lock:
            for row in rows:
                ids, _ = self.enc(row)
                tokens += int(ids.shape[-1])
                probs = option_logprobs(self.model, self.enc, row).exp().tolist()
                answers.append(primitives.answer(row, probs))
        return answers, tokens

    async def answer(self, rows):
        """Off the event loop: the forward passes are blocking and serialized."""
        started = time.time()
        result = await run_in_threadpool(self._answer, rows)
        print(f"{len(rows)} question(s) in {time.time() - started:.2f}s", flush=True)
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="vagmi/jev-lite",
                    help="Hub id or local directory")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    backend = TorchBackend(args.model, args.adapter, args.max_len, args.attn)
    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

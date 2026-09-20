#!/usr/bin/env python3
"""
serve_vllm.py — the same System One API, served by vLLM.

Why a second server: vLLM's continuous batching answers concurrent questions in
shared forward passes. Over the 1898-row eval set it ran 66.9 rows/s against
19.4 rows/s for the torch path — the questions in one request, and across
requests in flight, are batched together instead of queueing behind a lock.

Three things this file has to get exactly right, each of which cost a run to
find out:

  1. Never merge the adapter. A transformers load/save round-trip drops
     k_proj/v_proj/k_norm for layers 24-41 — Gemma 4 E4B shares KV across them,
     so transformers never instantiates those modules, while vLLM's loader
     still demands the weights. Runtime LoRA over the intact base avoids it.

  2. Feed it our own token ids. Letting vLLM tokenize the prompt string moved
     16.4% of argmaxes; building the ids exactly as jev_lite's Encoder does cuts
     that to 5.2%, and the rest is arithmetic, not text.

  3. Correct the temperature. The adapter was trained with QLoRA against a
     4-bit NF4 base and vLLM serves bf16, so the distribution it produces is
     sharper than the one that was trained. Accuracy is unaffected (0.807 vs
     0.799) but ECE doubles, and confidence IS p_max now. A single temperature
     of 1.4, fitted on half the eval set and measured on the other half, brings
     ECE to 0.033 — below the 4-bit baseline's 0.043. Pass --temperature 1.0 to
     serve the raw distribution.

Runs in the vLLM venv, which pins its own torch:

  uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm
  .venv-vllm/bin/python serve_vllm.py                             # from the Hub
  .venv-vllm/bin/python serve_vllm.py --adapter jev-lite-adapter  # or local

Auth: set JEV_API_KEY to require `Authorization: Bearer <key>`, same as serve.py.

Shutting it down: vLLM runs its engine in a CHILD process, so killing this one
by pid orphans the engine and it keeps the GPU. Stop the process group, or send
SIGINT instead:

  kill -INT <pid>          # clean: uvicorn shuts the engine down with it
  kill -- -<pgid>          # or take the whole group
  nvidia-smi               # confirm the memory came back
"""
import argparse
import asyncio
import math
import os
import uuid

import uvicorn
from transformers import AutoTokenizer

import primitives
from api import create_app, resolve_adapter

# Fitted on half the held-out rows, measured on the other half. See the module
# docstring; this is a property of serving a QLoRA adapter on a bf16 base, not
# of the model itself.
DEFAULT_TEMPERATURE = 1.4


class VllmBackend:
    """One vLLM engine, answering option-letter questions with no generation."""

    name = "vllm"

    def __init__(self, base, adapter, max_len, gpu_fraction, temperature):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.lora.request import LoRARequest

        self.temperature = temperature
        # vLLM's LoRARequest needs a real directory, so a Hub id is fetched first.
        adapter = os.path.abspath(resolve_adapter(adapter))
        self.tok = AutoTokenizer.from_pretrained(adapter)
        self.letter_ids = [self._letter_id(c) for c in primitives.LETTERS]
        self.lora = LoRARequest("jev", 1, adapter)

        print(f"loading {base} + {adapter} under vLLM ...", flush=True)
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=base, dtype="bfloat16", max_model_len=max_len,
            gpu_memory_utilization=gpu_fraction,
            enable_lora=True, max_lora_rank=16, max_loras=1))
        print(f"ready (temperature {temperature})", flush=True)

    def _letter_id(self, c):
        for s in (" " + c, c):
            ids = self.tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        raise ValueError(f"Letter {c!r} is not a single token for this tokenizer")

    def encode(self, row):
        """Byte-for-byte what jev_lite's Encoder builds, BOS included."""
        suffix = f"\n</state>\n\n{primitives.question_block(row)}\nAnswer:"
        return ([self.tok.bos_token_id]
                + self.tok.encode(primitives.PREFIX, add_special_tokens=False)
                + self.tok.encode(row["state"], add_special_tokens=False)
                + self.tok.encode(suffix, add_special_tokens=False))

    async def _one(self, row, ids):
        from vllm import SamplingParams, TokensPrompt

        n = len(row["options"])
        # Masking to the option letters is what makes this a decision and not a
        # generation: there is no token it could emit that isn't an option.
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=n,
                                allowed_token_ids=self.letter_ids[:n])
        final = None
        async for out in self.engine.generate(
                TokensPrompt(prompt_token_ids=ids), params,
                request_id=f"jev-{uuid.uuid4().hex[:16]}", lora_request=self.lora):
            final = out

        probs = [0.0] * n
        for tid, info in (final.outputs[0].logprobs[0] or {}).items():
            if tid in self.letter_ids[:n]:
                probs[self.letter_ids.index(tid)] = math.exp(info.logprob)
        total = sum(probs)
        probs = [p / total for p in probs] if total > 1e-9 else [1 / n] * n
        if self.temperature != 1.0:
            probs = primitives.temper(probs, self.temperature)
        return primitives.answer(row, probs)

    async def answer(self, rows):
        """All questions go in flight together; vLLM batches what it can."""
        encoded = [self.encode(r) for r in rows]
        answers = await asyncio.gather(*(self._one(r, ids)
                                         for r, ids in zip(rows, encoded)))
        return list(answers), sum(len(ids) for ids in encoded)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="vagmi/jev-lite",
                    help="Hub id or local directory")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-fraction", type=float, default=0.88)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="calibration correction for bf16 serving; 1.0 = raw")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    backend = VllmBackend(args.model, args.adapter, args.max_len,
                          args.gpu_fraction, args.temperature)
    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

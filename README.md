# jevlite

Build a **System One** decision model: something that reads a state, reads a typed question
about it, and returns a calibrated probability distribution over the allowed answers.

Because the answer is read from the logits at the option letters, the model **cannot**
answer outside the options it was given. No parsing, no retries, no "as an AI language
model". It answers the three [TypeSafe](https://docs.typesafe.ai) primitives:


| type | question | returns |
|---|---|---|
| `choice` | pick one of several named options | the option, probabilities, confidence |
| `score` | rate against ordered levels | expected level, legend, probabilities, confidence |
| `noul` | is this true? | a single probability |

The trained adapter is at **[vagmi/jev-lite](https://huggingface.co/vagmi/jev-lite)**.


### 4. Serve

Both servers speak the TypeSafe wire API (`POST /v1/systemone`, `GET /v1/models`) and share
their contract via `api.py`, so the official SDK drives either one unmodified.

```bash
python serve.py                        # torch 4-bit, pulls vagmi/jev-lite from the Hub
.venv-vllm/bin/python serve_vllm.py    # vLLM bf16, same adapter

python serve.py --adapter jev-lite-adapter   # or point at a local training output
```

Both take `--adapter` as a Hub id or a local directory; the default is the published
adapter, so a fresh checkout serves without training anything first.

```python
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul
# TYPESAFE_BASE_URL=http://localhost:8000

with TypeSafeClient() as client:
    r = client.system_one(
        state="Help! My payouts have been failing for 3 days.",
        questions={
            "is_urgent": Noul(instructions="Does this convey urgency?"),
            "department": Choice(instructions="Which team should handle this?",
                                 criteria={"billing": "Payments, invoicing, refunds",
                                           "technical": "Bugs, outages, integrations"}),
            "frustration": Score(instructions="How frustrated is the customer?",
                                 criteria=["Calm", "Frustrated", "Very angry"]),
        })
    print(r.nouls["is_urgent"].noul, r.choices["department"].choice)
```


| backend | 1 question | 3 questions | 16 concurrent |
|---|---|---|---|
| torch 4-bit | 40.5 ms | 119.1 ms | serializes behind a lock |
| vLLM bf16 | 19.0 ms | 27.7 ms | 667 questions/s |


Set `JEV_API_KEY` to require `Authorization: Bearer <key>`. Unset, the server accepts any
caller — fine on localhost, not on a network.

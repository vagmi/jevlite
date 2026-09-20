#!/usr/bin/env python3
"""
api.py — the TypeSafe System One wire contract, minus the model.

Everything here is backend-agnostic: the request schema, the translation from
an API question into the row shape jev_lite was trained on, and the routes.
serve.py plugs in a torch backend, serve_vllm.py plugs in vLLM. Two servers,
one definition of the protocol — because two hand-maintained copies of a wire
contract drift, and the drift shows up as a client bug nobody can reproduce.

A backend is any object with:

    name  -> str
    async answer(rows) -> (answers, input_token_count)

where `rows` are normalized primitives.py rows and each answer is whatever
primitives.answer() produced for it. resolve_adapter() lives here too, since
both backends need the same answer to "where are the weights".
"""
import json
import os
import uuid
from typing import Annotated, Any, Literal, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, RootModel

import primitives

JSONContent = Union[str, dict[str, Any], list[Any]]
REQUEST_ID_HEADER = "x-typesafe-request-id"

# "jev-latest" is the SDK's default model, so it has to resolve to something or
# every default-constructed client 404s.
MODELS = [
    {"name": "jev-latest",
     "description": "Gemma 4 E4B tuned on typed decision questions; "
                    "one forward pass over the option letters.",
     "release_date": "2026-09-19"},
    {"name": "jev-lite-gemma4-e4b",
     "description": "Pinned name for the same adapter.",
     "release_date": "2026-09-19"},
]
MODEL_NAMES = {m["name"] for m in MODELS}


ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def resolve_adapter(ref: str) -> str:
    """A local directory, or a Hub id to fetch — always returns a local path.

    peft resolves Hub ids itself, but vLLM's LoRARequest wants a real directory,
    so both go through this and get the same thing.
    """
    if os.path.isdir(ref):
        return ref
    from huggingface_hub import snapshot_download

    print(f"fetching adapter {ref} from the Hub ...", flush=True)
    path = snapshot_download(ref, allow_patterns=ADAPTER_FILES)
    print(f"  -> {path}", flush=True)
    return path


# ------------------------------------------------------------------ schema
# Mirrors typesafe_sdk._schemas.models. FastAPI rejects a malformed request
# with exactly the {"detail": [{loc, msg, type, ...}]} body the SDK parses,
# so the validation contract comes for free from declaring these.

class NoulCriteria(BaseModel):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(..., min_length=1)


class Question(RootModel[Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion],
                                   Field(discriminator="type")]]):
    pass


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(..., min_length=1)


# -------------------------------------------------------------- conversion

def as_text(content: JSONContent | None) -> str | None:
    """Instructions and criteria may arrive as text, an object, or an array."""
    if content is None:
        return None
    if isinstance(content, str):
        return content.strip() or None
    return json.dumps(content, ensure_ascii=False)


# `instructions` is optional in the schema, so a question can arrive as pure
# criteria. The model still needs something in the question slot.
DEFAULT_INSTRUCTIONS = {
    "noul": "Is this true?",
    "choice": "Which option applies?",
    "score": "Which level applies?",
}


def invalid(name, kind, field, msg, code="value_error", value=None):
    detail = {"loc": ["body", "questions", name, kind, field], "msg": msg, "type": code}
    if value is not None:
        detail["input"] = value
    return HTTPException(422, detail=[detail])


def to_row(name: str, question: Question, state: str) -> dict:
    """Turn one API question into the row shape jev_lite was trained on."""
    q = question.root
    kind = q.type
    row = {"type": kind, "state": state,
           "question": as_text(q.instructions) or DEFAULT_INSTRUCTIONS[kind]}

    if kind == "choice":
        options = list(q.criteria)
        if len(options) < 2:
            raise invalid(name, kind, "criteria", "A choice needs at least two criteria.",
                          "too_short", q.criteria)
        row["options"] = options
        described = {k: as_text(v) for k, v in q.criteria.items()}
        row["criteria"] = {k: v for k, v in described.items() if v}
    elif kind == "score":
        # A level's description IS its label: position sets the score, so the
        # model reads the rubric itself rather than a bare number.
        row["options"] = [as_text(c) or f"level {i}" for i, c in enumerate(q.criteria)]
        row["ordered"] = True
    else:
        row["options"] = list(primitives.TRUE_FALSE)
        if q.criteria:
            described = {"true": as_text(q.criteria.true),
                         "false": as_text(q.criteria.false)}
            row["criteria"] = {k: v for k, v in described.items() if v}

    if len(row["options"]) > len(primitives.LETTERS):
        raise invalid(name, kind, "criteria",
                      f"At most {len(primitives.LETTERS)} criteria are supported.",
                      "too_long")
    # Duplicate labels would make the answer ambiguous: two options, one name.
    if len(set(row["options"])) != len(row["options"]):
        raise invalid(name, kind, "criteria", "Criteria must be distinct.")
    return primitives.normalize(row)


# -------------------------------------------------------------------- app

def require_key(authorization: Annotated[str | None, Header()] = None):
    expected = os.environ.get("JEV_API_KEY")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(401, detail="Invalid API key.")


def create_app(backend) -> FastAPI:
    """Wrap a backend in the System One HTTP contract."""
    app = FastAPI(title=f"jev-lite ({backend.name})")

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        """Every response carries an id, which is what the SDK reports on failure."""
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = f"req_{uuid.uuid4().hex[:24]}"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        # A list detail is the validation shape; anything else the SDK reads as
        # a message. Both paths are handled by the client's extract_message.
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    def list_models():
        return {"models": MODELS}

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": backend.name}

    @app.post("/v1/systemone", dependencies=[Depends(require_key)])
    async def system_one(body: SystemOneRequest):
        if body.model not in MODEL_NAMES:
            raise HTTPException(404, detail=f"Unknown model {body.model!r}. "
                                            f"Available: {', '.join(sorted(MODEL_NAMES))}.")
        state = as_text(body.state) or ""
        names = list(body.questions)
        rows = [to_row(name, body.questions[name], state) for name in names]
        answers, input_tokens = await backend.answer(rows)
        return {
            "model": body.model,
            "answers": dict(zip(names, answers)),
            # One answer is read from a single position, so a question costs
            # exactly one output token however many options it lists.
            "usage": {"input_tokens": input_tokens, "output_tokens": len(rows)},
        }

    return app

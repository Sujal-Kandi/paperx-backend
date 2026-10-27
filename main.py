"""
PaperX /ask endpoint (v2: claim-level evidence).

The model returns CLAIMS, each with its own verbatim quote. In code we check
every quote really appears in the retrieved chunk, drop claims that fail, and
build the answer ONLY from claims that survived. No verified claim => refusal.

Setup:
    pip install fastapi uvicorn groq python-dotenv
    .env file containing:  GROQ_API_KEY=your_key
    uvicorn main:app --reload
"""
import json
import os
import re
import time
import unicodedata
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

from ingest import DATA_DIR, get_collection, ingest, search

DEBUG = os.getenv("PAPERX_DEBUG") == "1"  # prints raw model output on every refusal
MODEL = "openai/gpt-oss-120b"  # or "llama-3.3-70b-versatile"
client = Groq()

@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(exist_ok=True)
    if get_collection().count() == 0:
        ingest()  # fresh server: download the PDF and index it
    search("warm up", k=1)  # load the embedding model before the first real question
    yield


app = FastAPI(title="PaperX", lifespan=lifespan)

# Comma-separated list, e.g. ALLOWED_ORIGINS="https://paperx.vercel.app,http://localhost:3000"
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple per-IP rate limit so a public link can't drain the Groq key.
HITS = defaultdict(deque)
LIMIT, WINDOW = 12, 60  # 12 questions per minute per visitor


def rate_limit(request: Request) -> None:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now = time.time()
    q = HITS[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= LIMIT:
        raise HTTPException(status_code=429, detail="Too many questions, try again in a minute.")
    q.append(now)


SYSTEM = """You answer questions about ONE research paper using ONLY the numbered excerpts provided.
Build the answer as 1 to 3 claims. Each claim is ONE sentence and carries its own evidence.
Rules:
1. Use only facts stated in the excerpts. Never use outside knowledge.
2. For each claim give the excerpt number and a verbatim quote (at least 6 and under 30 words) copied exactly from that excerpt. The quote must directly contain the key fact, number, or reason stated in the claim.
3. Do not state a reason, number, or mechanism unless one of your quotes contains it.
4. If the excerpts do not contain the answer, set "supported" to false and return no claims.
Return ONLY JSON: {"supported": bool, "claims": [{"text": str, "excerpt": int, "quote": str}]}"""


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=300)


def clean(s: str) -> str:
    """Unicode-normalize (no-break spaces/hyphens, ligatures) and collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def norm(s: str) -> str:
    s = clean(s).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return s.lower()


REFUSAL = {
    "answer": "The paper doesn't clearly say this in the passages I found.",
    "supported": False,
    "citations": [],
}


def ask(req: AskRequest):
    hits = search(req.question, k=6)
    context = "\n\n".join(
        f"[{i + 1}] (page {h['meta']['page']}) {h['text']}" for i, h in enumerate(hits)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {req.question}"},
        ],
    )

    raw = resp.choices[0].message.content
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        if DEBUG:
            print(f"[REFUSED: bad JSON] {req.question}\n  raw: {raw}")
        return REFUSAL

    texts, cites = [], []
    for c in out.get("claims", []):
        try:
            hit = hits[int(c["excerpt"]) - 1]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        quote = clean(c.get("quote", ""))
        if len(quote.split()) >= 6 and norm(quote) in norm(hit["text"]):
            claim = clean(c.get("text", ""))
            texts.append(claim)
            cites.append(
                {
                    "claim": claim,
                    "page": hit["meta"]["page"],
                    "para_idx": hit["meta"]["para_idx"],
                    "quote": quote,
                    "bbox": hit["meta"]["bbox"],
                }
            )

    if not out.get("supported") or not cites:
        if DEBUG:
            print(f"[REFUSED] {req.question}\n  supported={out.get('supported')}, "
                  f"verified claims={len(cites)}\n  raw: {raw}")
        return REFUSAL
    return {"answer": " ".join(texts), "supported": True, "citations": cites}


@app.post("/ask")
def ask_route(req: AskRequest, request: Request):
    rate_limit(request)
    return ask(req)


@app.get("/health")
def health():
    return {"status": "ok"}
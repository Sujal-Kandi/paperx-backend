"""
PaperX ingestion: PDF -> paragraph chunks with page metadata -> ChromaDB.

Setup:
    pip install pymupdf chromadb requests
    python ingest.py            # downloads the paper and ingests it
    python ingest.py --test     # runs a few test queries and prints citations
"""
import argparse
import pathlib
import re

import chromadb
import fitz  # PyMuPDF
import requests
from rank_bm25 import BM25Okapi

PAPER_ID = "attention-is-all-you-need"
ARXIV_URL = "https://arxiv.org/pdf/1706.03762"
DATA_DIR = pathlib.Path(__file__).parent / "data"
PDF_PATH = DATA_DIR / f"{PAPER_ID}.pdf"
CHROMA_DIR = str(DATA_DIR / "chroma")

MIN_CHARS = 60  # drop headers, page numbers, stray fragments


def download_pdf() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if PDF_PATH.exists():
        return
    r = requests.get(ARXIV_URL, timeout=60)
    r.raise_for_status()
    PDF_PATH.write_bytes(r.content)
    print(f"Downloaded {PDF_PATH}")


def clean(text: str) -> str:
    text = text.replace("-\n", "")          # re-join hyphenated line breaks
    text = re.sub(r"\s*\n\s*", " ", text)    # newlines -> spaces
    return re.sub(r"\s{2,}", " ", text).strip()


def extract_chunks(pdf_path: pathlib.Path) -> list[dict]:
    """One chunk per text block (~paragraph), keeping page + position."""
    chunks = []
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc):
        # sort=True gives reading order (helps with two-column layouts)
        blocks = page.get_text("blocks", sort=True)
        para_idx = 0
        for x0, y0, x1, y1, text, _block_no, block_type in blocks:
            if block_type != 0:  # skip images
                continue
            text = clean(text)
            if len(text) < MIN_CHARS:
                continue
            para_idx += 1
            chunks.append(
                {
                    "id": f"{PAPER_ID}-p{page_idx + 1}-para{para_idx}",
                    "text": text,
                    "meta": {
                        "paper_id": PAPER_ID,
                        "page": page_idx + 1,  # 1-indexed, matches the PDF viewer
                        "para_idx": para_idx,
                        "kind": "caption"
                        if re.match(r"^(Figure|Table)\s*\d", text)
                        else "body",
                        "bbox": f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}",
                    },
                }
            )
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Uses Chroma's default embedding model (all-MiniLM-L6-v2). Swap later if needed.
    return client.get_or_create_collection(name="papers")


def ingest() -> None:
    download_pdf()
    chunks = extract_chunks(PDF_PATH)
    col = get_collection()
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["meta"] for c in chunks],
    )
    print(f"Ingested {len(chunks)} chunks across {chunks[-1]['meta']['page']} pages")


_LEX = {}


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lexical_index(include_captions: bool):
    """BM25 index over all chunks, built once from what is already in Chroma."""
    if include_captions not in _LEX:
        where = (
            {"paper_id": PAPER_ID}
            if include_captions
            else {"$and": [{"paper_id": PAPER_ID}, {"kind": "body"}]}
        )
        data = get_collection().get(where=where, include=["documents", "metadatas"])
        _LEX[include_captions] = (
            BM25Okapi([_tok(d) for d in data["documents"]]),
            data["ids"],
            data["documents"],
            data["metadatas"],
        )
    return _LEX[include_captions]


def search(query: str, k: int = 4, include_captions: bool = False) -> list[dict]:
    """Hybrid search: embeddings (meaning) + BM25 (exact words), merged with
    reciprocal rank fusion. Fixes misses on exact terms like 'inner-layer'."""
    col = get_collection()
    where = (
        {"paper_id": PAPER_ID}
        if include_captions
        else {"$and": [{"paper_id": PAPER_ID}, {"kind": "body"}]}
    )
    res = col.query(query_texts=[query], n_results=20, where=where)
    sem_ids = res["ids"][0]
    items = {
        i: {"text": d, "meta": m, "distance": dist}
        for i, d, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    }

    bm25, ids, docs, metas = _lexical_index(include_captions)
    scores = bm25.get_scores(_tok(query))
    lex_ids = []
    for i in sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:20]:
        if scores[i] <= 0:
            break
        lex_ids.append(ids[i])
        items.setdefault(ids[i], {"text": docs[i], "meta": metas[i], "distance": None})

    fused = {}
    for ranking in (sem_ids, lex_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    return [items[c] for c in sorted(fused, key=fused.get, reverse=True)[:k]]


TEST_QUERIES = [
    "Why is the dot product scaled by the square root of d_k?",
    "How many attention heads does the base model use?",
    "What is the positional encoding formula?",
    "What BLEU score did the big model get on English-to-German?",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        for q in TEST_QUERIES:
            print(f"\nQ: {q}")
            for hit in search(q, k=2):
                m = hit["meta"]
                print(f"  [p{m['page']}, para {m['para_idx']}] {hit['text'][:160]}...")
    else:
        ingest()
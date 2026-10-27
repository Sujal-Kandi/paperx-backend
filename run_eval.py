"""
PaperX eval. Run from the folder that holds main.py (no server needed):
    python run_eval.py

For answerable questions it checks, in code (no human judging):
  - answered:   the system did not refuse
  - correct:    the answer contains the expected fact
  - cited:      at least one VERIFIED citation quote contains the fact
                (and is on an expected page, where one is set)
For unanswerable questions it checks that the system refused.

Results are also saved to eval_results.json so you can read every failure.
"""
import json
import pathlib
import time
import unicodedata

from main import AskRequest, ask

HERE = pathlib.Path(__file__).resolve().parent
CASES = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
PAUSE = 3  # seconds between calls, to stay under Groq rate limits


def n(s):
    """Normalize invisible unicode variants so matching is about content, not characters."""
    return " ".join(unicodedata.normalize("NFKC", s).split()).lower()


def as_list(x):
    return x if isinstance(x, list) else [x]


def call(q):
    for attempt in range(2):
        try:
            return ask(AskRequest(question=q))
        except Exception as e:  # rate limit or network hiccup: wait, retry once
            if attempt == 1:
                return {"answer": f"ERROR: {e}", "supported": False, "citations": [], "error": True}
            time.sleep(10)


rows = []
for i, c in enumerate(CASES, 1):
    res = call(c["q"])
    ans = n(res["answer"])
    row = {"q": c["q"], "answerable": c["answerable"], "supported": res["supported"], "answer": res["answer"],
           "citations": res["citations"]}
    if c["answerable"]:
        row["answered"] = res["supported"]
        row["correct"] = any(n(k) in ans for k in c["answer_any"])
        row["cited"] = any(
            any(n(nd) in n(x["quote"]) for nd in as_list(c["quote_has"])) and (not c["pages"] or x["page"] in c["pages"])
            for x in res["citations"]
        )
        row["pass"] = row["answered"] and row["correct"] and row["cited"]
    else:
        row["pass"] = not res["supported"]
    rows.append(row)
    print(f"{i:>2}. {'PASS' if row['pass'] else 'FAIL'}  {c['q']}")
    time.sleep(PAUSE)

yes = [r for r in rows if r["answerable"]]
no = [r for r in rows if not r["answerable"]]
cnt = lambda xs, k: sum(1 for r in xs if r[k])

print("\n=== PaperX eval ===")
print(f"Answerable ({len(yes)}): answered {cnt(yes,'answered')}/{len(yes)}, "
      f"correct {cnt(yes,'correct')}/{len(yes)}, citation supports fact {cnt(yes,'cited')}/{len(yes)}, "
      f"all three {cnt(yes,'pass')}/{len(yes)}")
print(f"Unanswerable ({len(no)}): correctly refused {cnt(no,'pass')}/{len(no)}")
print(f"Overall pass: {cnt(rows,'pass')}/{len(rows)}")

(HERE / "eval_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nDetails saved to eval_results.json")
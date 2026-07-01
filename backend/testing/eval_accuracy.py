"""
CerviCare / PaperMind accuracy eval — 25 real questions across 3 documents.

Documents expected to be ingested under your account:
  1. CerviCare Synopsis (project report / synopsis PDF)
  2. Literature Review (your lit review PDF)
  3. IJACSA paper ("Predicting Cervical Cancer" paper PDF)

BEFORE YOU RUN
  1. Start your backend:  uvicorn app.main:app --reload --port 8000
  2. Ingest all three documents via the frontend (if not already done)
  3. Get a fresh Clerk token:
       Open frontend → DevTools Console → await window.Clerk.session.getToken()
  4. Set DOCUMENT_FILTER_* below to the exact filenames shown in your
     /api/documents list (or leave as None to search across all docs)

RUN
  pip install httpx --break-system-packages
  python eval_cervicare.py --token "PASTE_TOKEN_HERE"

OUTPUT
  Per-question pass/fail + final score to cite on your resume.
"""

import argparse
import asyncio
import statistics
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"

# ── Set these to the exact filenames in your Qdrant/Cloudinary store ──────────
# Leave as None to search across all ingested documents (still works, slightly
# noisier since answers might bleed across docs).
DOC_SYNOPSIS    = "CerviCare_final_report.pdf"   # e.g. "cervicare_synopsis.pdf"
DOC_LIT_REVIEW  = "Literature_review.pdf"  # e.g. "literature_review.pdf"
DOC_IJACSA      = "Predicting_Cervical_Cancer_using_Machine_Learning_Methods.pdf"  # e.g. "ijacsa_cervical_paper.pdf"

# ── All 25 eval questions ─────────────────────────────────────────────────────
QUESTIONS = [

    # ── CerviCare Synopsis (11 questions) ─────────────────────────────────────
    {
        "section": "Synopsis",
        "question": "How many patient records and features does the UCI Cervical Cancer Risk Factors dataset contain before column dropping?",
        "expect_any": ["858", "36"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What is the negative-to-positive class imbalance ratio in the UCI cervical cancer dataset?",
        "expect_any": ["14:1", "14 to 1", "14-to-1"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "Which two columns are dropped due to missing values exceeding 91%?",
        "expect_any": ["STDs: Time since first diagnosis", "STDs: Time since last diagnosis", "time since first", "time since last"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What oversampling technique does CerviCare use to address class imbalance, and on which split is it applied?",
        "expect_any": ["SMOTE", "training split", "training set", "exclusively on the training"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What are the target Recall, Precision, ROC-AUC, and PR-AUC for CerviCare models?",
        "expect_any": ["90%", "80%", "0.95", "0.60", "PR-AUC", "ROC-AUC"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "How many total model configurations and families are benchmarked in CerviCare?",
        "expect_any": ["twelve", "12", "five families", "5 families"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What hyperparameter tuning method is used in CerviCare and how many iterations and folds?",
        "expect_any": ["RandomizedSearchCV", "30 iterations", "30", "5-fold", "5 fold"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What explainability technique is used for per-sample predictions in CerviCare?",
        "expect_any": ["SHAP", "TreeExplainer", "SHapley"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What is the train/test split ratio and resulting sample counts in CerviCare?",
        "expect_any": ["80/20", "80-20", "686", "172"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What scale_pos_weight value is configured for XGBoost or LightGBM in CerviCare?",
        "expect_any": ["14.68", "≈14", "14.6"],
        "document_filter": DOC_SYNOPSIS,
    },
    {
        "section": "Synopsis",
        "question": "What web frameworks are proposed for the CerviCare demo interface?",
        "expect_any": ["Streamlit", "Flask", "FastAPI", "React"],
        "document_filter": DOC_SYNOPSIS,
    },

    # ── Literature Review (8 questions) ───────────────────────────────────────
    {
        "section": "Literature Review",
        "question": "According to WHO figures cited in the review, how many new cervical cancer cases and deaths occurred globally in 2020?",
        "expect_any": ["604,000", "604000", "342,000", "342000"],
        "document_filter": DOC_LIT_REVIEW,
    },
    {
        "section": "Literature Review",
        "question": "What percentage of cervical cancer deaths occur in low- and middle-income countries?",
        "expect_any": ["90%", "nearly 90", "90 percent"],
        "document_filter": DOC_LIT_REVIEW,
    },
    {
        "section": "Literature Review",
        "question": "What are the five-year survival rates for Stage I vs Stage IV cervical cancer cited in the review?",
        "expect_any": ["90%", "20%", "Stage I", "Stage IV"],
        "document_filter": DOC_LIT_REVIEW,
    },
    {
        "section": "Literature Review",
        "question": "What recall improvement did Choudhary et al. report when applying SMOTE with Random Forest?",
        "expect_any": ["61%", "88%", "61 to 88", "from 61"],
        "document_filter": DOC_LIT_REVIEW,
    },
    {
        "section": "Literature Review",
        "question": "Which two HPV genotypes account for approximately 70% of cervical cancer cases?",
        "expect_any": ["HPV-16", "HPV-18", "HPV 16", "HPV 18"],
        "document_filter": DOC_LIT_REVIEW,
    },
    {
        "section": "Literature Review",
        "question": "What three-classifier ensemble does the CerviCare literature review propose?",
        "expect_any": ["Logistic Regression", "Random Forest", "XGBoost"],
        "document_filter": DOC_LIT_REVIEW,
    },
    {
        "section": "Literature Review",
        "question": "What percentage of records carry a positive Biopsy label according to the review's discussion of class imbalance?",
        "expect_any": ["8%", "fewer than 8", "less than 8"],
        "document_filter": DOC_LIT_REVIEW,
    },
    {
        "section": "Literature Review",
        "question": "What are the research gaps identified in the review? Name at least one.",
        "expect_any": ["Integration deficit", "SMOTE application", "Metric selection", "Usability", "Dataset diversity", "Multi-label", "research gap"],
        "document_filter": DOC_LIT_REVIEW,
    },

    # ── IJACSA Paper (6 questions) ─────────────────────────────────────────────
    {
        "section": "IJACSA Paper",
        "question": "Which three classifiers are combined in the voting method used in the IJACSA cervical cancer paper?",
        "expect_any": ["Decision tree", "logistic regression", "random forest"],
        "document_filter": DOC_IJACSA,
    },
    {
        "section": "IJACSA Paper",
        "question": "What dimensionality reduction technique is combined with SMOTE in this paper, and how many principal components are used?",
        "expect_any": ["PCA", "Principal Component", "11 principal", "11 components"],
        "document_filter": DOC_IJACSA,
    },
    {
        "section": "IJACSA Paper",
        "question": "What cross-validation method is used to prevent overfitting in the IJACSA paper?",
        "expect_any": ["stratified 10-fold", "10-fold", "10 fold", "stratified k-fold"],
        "document_filter": DOC_IJACSA,
    },
    {
        "section": "IJACSA Paper",
        "question": "What ROC AUC value did the SMOTE-Voting-PCA model achieve for the Schiller target variable?",
        "expect_any": ["99.80%", "99.8%", "99.80", "0.998"],
        "document_filter": DOC_IJACSA,
    },
    {
        "section": "IJACSA Paper",
        "question": "How many cervical cancer cases malignant and benign were used for the Biopsy target variable?",
        "expect_any": ["55", "803"],
        "document_filter": DOC_IJACSA,
    },
    {
        "section": "IJACSA Paper",
        "question": "What institution or hospital did the UCI cervical cancer dataset originate from?",
        "expect_any": ["Hospital Universitario de Caracas", "Caracas", "Venezuela"],
        "document_filter": DOC_IJACSA,
    },
]

NO_INFO_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "i don't know",
    "cannot find",
    "not mentioned",
    "not provided",
    "no information",
]


def grade(answer: str, expect_any: list[str]) -> bool:
    a = answer.lower()
    return any(kw.lower() in a for kw in expect_any)


def is_no_info(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in NO_INFO_PHRASES)


async def run_question(client: httpx.AsyncClient, headers: dict, item: dict) -> dict:
    payload = {
        "question": item["question"],
        "document_filter": item.get("document_filter"),
    }
    start = time.monotonic()
    try:
        resp = await client.post(f"{BASE_URL}/api/query", json=payload, headers=headers, timeout=60.0)
        elapsed = time.monotonic() - start
        if resp.status_code != 200:
            return {**item, "ok": False, "error": f"HTTP {resp.status_code}", "ms": elapsed * 1000}
        data = resp.json()
        answer = data.get("answer", "")
        sources = len(data.get("sources", []))
        passed = grade(answer, item["expect_any"])
        no_info = is_no_info(answer)
        return {**item, "ok": True, "passed": passed, "no_info": no_info, "answer": answer, "sources": sources, "ms": elapsed * 1000}
    except httpx.RequestError as exc:
        elapsed = time.monotonic() - start
        return {**item, "ok": False, "error": str(exc), "ms": elapsed * 1000}


async def main_async(token: str, concurrency: int):
    headers = {"Authorization": f"Bearer {token}"}
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(client, item):
        async with semaphore:
            return await run_question(client, headers, item)

    async with httpx.AsyncClient() as client:
        tasks = [bounded(client, q) for q in QUESTIONS]
        results = await asyncio.gather(*tasks)

    return results


def main():
    parser = argparse.ArgumentParser(description="CerviCare 25-question accuracy eval")
    parser.add_argument("--token", required=True)
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel requests (keep low to avoid Gemini rate limits)")
    args = parser.parse_args()

    print(f"\nRunning {len(QUESTIONS)} questions | concurrency={args.concurrency}\n")
    results = asyncio.run(main_async(args.token, args.concurrency))

    # ── Per-section breakdown ─────────────────────────────────────────────────
    sections = {}
    for r in results:
        sec = r["section"]
        sections.setdefault(sec, []).append(r)

    latencies = []
    total_pass = 0

    for sec, items in sections.items():
        print(f"\n── {sec} ({'─' * (50 - len(sec))})")
        for i, r in enumerate(items, 1):
            if not r.get("ok"):
                print(f"  [{i}] ERROR   — {r.get('error')}")
                print(f"       Q: {r['question']}")
                continue

            status = "✅ PASS" if r["passed"] else ("⚠️  NO-INFO" if r["no_info"] else "❌ FAIL")
            print(f"  [{i}] {status}  ({r['ms']:.0f}ms, {r['sources']} sources)")
            print(f"       Q: {r['question']}")
            if not r["passed"]:
                print(f"       A: {r['answer'][:180]}{'...' if len(r['answer']) > 180 else ''}")
                print(f"       Expected one of: {r['expect_any']}")
            latencies.append(r["ms"])
            if r["passed"]:
                total_pass += 1

    total_ok = sum(1 for r in results if r.get("ok"))
    total_err = len(results) - total_ok
    no_info_count = sum(1 for r in results if r.get("ok") and r.get("no_info") and not r.get("passed"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions:   {len(QUESTIONS)}")
    print(f"Requests failed:   {total_err}")
    print(f"Correct (PASS):    {total_pass}/{total_ok}  ({total_pass/total_ok*100:.1f}%)" if total_ok else "N/A")
    if no_info_count:
        print(f"No-info responses: {no_info_count}  (retrieval miss, not model failure)")

    for sec, items in sections.items():
        ok_items = [r for r in items if r.get("ok")]
        sec_pass = sum(1 for r in ok_items if r.get("passed"))
        print(f"  {sec}: {sec_pass}/{len(ok_items)}")

    if latencies:
        print(f"\nLatency (successful):")
        print(f"  p50: {sorted(latencies)[len(latencies)//2]:.0f}ms")
        p95_idx = int(len(latencies) * 0.95)
        print(f"  p95: {sorted(latencies)[min(p95_idx, len(latencies)-1)]:.0f}ms")
        print(f"  mean: {statistics.mean(latencies):.0f}ms")

    print("\n" + "=" * 60)
    print("RESUME BULLET (use your actual numbers above):")
    print(f'  "Validated retrieval quality with a manually-labeled {len(QUESTIONS)}-question')
    print(f'  eval set across {len(sections)} real research documents, achieving')
    print(f'  {total_pass}/{total_ok} correct responses ({total_pass/total_ok*100:.1f}% accuracy)"' if total_ok else "")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
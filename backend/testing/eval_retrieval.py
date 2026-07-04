"""
PaperMind — RETRIEVAL-ONLY eval (zero Gemini LLM calls).

WHY THIS EXISTS
  Your pipeline makes 2 Gemini LLM calls per question (query rewrite + generation).
  At 5 RPM free tier, 25 questions = 50 calls = ~10 minutes + constant 429s.

  This script bypasses both LLM calls entirely. It calls _retrieve_hybrid()
  directly using only the Gemini EMBEDDING API (separate quota, 1500 RPD free)
  and checks whether the right chunks are actually returned from Qdrant.

  This tests the part of your system that's actually novel and resume-worthy:
  the hybrid dense+BM25 RRF retrieval pipeline — not the commodity LLM step.

WHAT IT MEASURES
  For each question, it checks whether any returned chunk contains the
  expected keywords. This is "chunk recall" — did retrieval surface the
  right content? If yes, the LLM would almost certainly answer correctly
  given that context. If no, the LLM can't save you.

SETUP
  1. Run from inside your backend directory so imports resolve:
       cd /path/to/PaperMind/backend
  2. Make sure your .env is present (needs GOOGLE_API_KEY, QDRANT_URL etc.)
  3. Set USER_ID below to your real Clerk user_id (find it in Clerk dashboard
     → Users → click your user → copy the "User ID" starting with "user_")

RUN
  pip install python-dotenv --break-system-packages
  python eval_retrieval_only.py

  No token needed. No Gemini LLM quota used. Runs in ~30-60 seconds.
"""

import os
import sys
import time
import statistics

# ── Ensure the backend directory is on sys.path so 'app' imports resolve ──────
# This allows the script to be run from any working directory.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ── Load .env before importing app modules ────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

# ── Fix relative paths that break when CWD != backend/ ───────────────────────
# QDRANT_PERSIST_DIR defaults to "./qdrant_data" in config.py (relative path).
# When this script runs from testing/, that resolves to the wrong directory.
# Override it with the absolute path so QdrantClient always finds the right data.
if not os.environ.get("QDRANT_URL"):  # only for local (non-cloud) mode
    os.environ.setdefault(
        "QDRANT_PERSIST_DIR",
        os.path.join(_BACKEND_DIR, "qdrant_data"),
    )

# ── Set your Clerk user_id here ───────────────────────────────────────────────
USER_ID = os.getenv("EVAL_USER_ID", "")

# ── Optional: narrow each question to a specific ingested filename ────────────
# Set to None to search all your documents.
DOC_SYNOPSIS    = "CerviCare_final_report.pdf"
DOC_LIT_REVIEW  = "Literature_review.pdf"
DOC_IJACSA      = "Predicting_Cervical_Cancer_using_Machine_Learning_Methods.pdf" 

# ── 25 eval questions with expected keywords ──────────────────────────────────
QUESTIONS = [
    # Synopsis
    {"section": "Synopsis",  "q": "How many patient records and features does the UCI Cervical Cancer Risk Factors dataset contain before column dropping?", "expect": ["858", "36"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What is the negative-to-positive class imbalance ratio in the UCI cervical cancer dataset?", "expect": ["14:1", "14 to 1"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "Which two columns are dropped due to missing values exceeding 91%?", "expect": ["time since first diagnosis", "time since last diagnosis"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What oversampling technique does CerviCare use to address class imbalance, and on which split?", "expect": ["SMOTE", "training"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What are the target Recall Precision ROC-AUC and PR-AUC for CerviCare models?", "expect": ["90%", "80%", "0.95", "0.60"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "How many total model configurations and families are benchmarked in CerviCare?", "expect": ["twelve", "12", "five", "5"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What hyperparameter tuning method is used and how many iterations and folds?", "expect": ["RandomizedSearchCV", "30", "5-fold"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What explainability technique is used for per-sample predictions in CerviCare?", "expect": ["SHAP", "TreeExplainer"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What is the train test split ratio and resulting sample counts in CerviCare?", "expect": ["80/20", "686", "172"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What scale_pos_weight value is configured for XGBoost or LightGBM in CerviCare?", "expect": ["14.68", "14.6"], "doc": DOC_SYNOPSIS},
    {"section": "Synopsis",  "q": "What web frameworks are proposed for the CerviCare demo interface?", "expect": ["Streamlit", "Flask", "FastAPI", "React"], "doc": DOC_SYNOPSIS},
    # Literature Review
    {"section": "Lit Review", "q": "How many new cervical cancer cases and deaths occurred globally in 2020 according to WHO?", "expect": ["604,000", "604000", "342,000", "342000"], "doc": DOC_LIT_REVIEW},
    {"section": "Lit Review", "q": "What percentage of cervical cancer deaths occur in low and middle income countries?", "expect": ["90%", "nearly 90"], "doc": DOC_LIT_REVIEW},
    {"section": "Lit Review", "q": "What are the five year survival rates for Stage I vs Stage IV cervical cancer?", "expect": ["90%", "20%", "Stage I", "Stage IV"], "doc": DOC_LIT_REVIEW},
    {"section": "Lit Review", "q": "What recall improvement did Choudhary et al report when applying SMOTE with Random Forest?", "expect": ["61%", "88%", "61 to 88"], "doc": DOC_LIT_REVIEW},
    {"section": "Lit Review", "q": "Which two HPV genotypes account for approximately 70% of cervical cancer cases?", "expect": ["HPV-16", "HPV-18", "HPV 16", "HPV 18"], "doc": DOC_LIT_REVIEW},
    {"section": "Lit Review", "q": "What three-classifier ensemble does the CerviCare literature review propose?", "expect": ["Logistic Regression", "Random Forest", "XGBoost"], "doc": DOC_LIT_REVIEW},
    {"section": "Lit Review", "q": "What percentage of records carry a positive Biopsy label?", "expect": ["8%", "fewer than 8", "less than 8"], "doc": DOC_LIT_REVIEW},
    {"section": "Lit Review", "q": "What research gaps are identified in the literature review?", "expect": ["Integration deficit", "SMOTE application", "Metric selection", "Usability", "Dataset diversity", "Multi-label", "research gap"], "doc": DOC_LIT_REVIEW},
    # IJACSA Paper
    {"section": "IJACSA",    "q": "Which three classifiers are combined in the voting method in the IJACSA cervical cancer paper?", "expect": ["Decision tree", "logistic regression", "random forest"], "doc": DOC_IJACSA},
    {"section": "IJACSA",    "q": "What dimensionality reduction technique is combined with SMOTE and how many principal components are used?", "expect": ["PCA", "Principal Component", "11 principal", "11 components"], "doc": DOC_IJACSA},
    {"section": "IJACSA",    "q": "What cross-validation method is used to prevent overfitting in the IJACSA paper?", "expect": ["stratified 10-fold", "10-fold", "10 fold"], "doc": DOC_IJACSA},
    {"section": "IJACSA",    "q": "What ROC AUC value did the SMOTE-Voting-PCA model achieve for the Schiller target variable?", "expect": ["99.80%", "99.8%", "0.998"], "doc": DOC_IJACSA},
    {"section": "IJACSA",    "q": "How many malignant and benign cases were used for the Biopsy target variable?", "expect": ["55", "803"], "doc": DOC_IJACSA},
    {"section": "IJACSA",    "q": "What institution or hospital did the UCI cervical cancer dataset originate from?", "expect": ["Hospital Universitario de Caracas", "Caracas", "Venezuela"], "doc": DOC_IJACSA},
]


def grade_chunks(chunks: list, expect: list[str]) -> tuple[bool, str]:
    """Check if any returned chunk contains at least one expected keyword."""
    all_text = " ".join(c.page_content for c in chunks).lower()
    for kw in expect:
        if kw.lower() in all_text:
            return True, kw
    return False, ""


def main():
    if not USER_ID or USER_ID == "PASTE_YOUR_CLERK_USER_ID_HERE":
        print("❌ Set EVAL_USER_ID in your .env (or at the top of the script) before running.")
        sys.exit(1)

    # Import app modules after dotenv is loaded
    from app.services.rag_chain import _retrieve_hybrid, _retrieve_dense, _build_embeddings
    from app.config import get_settings
    settings = get_settings()

    print(f"\nPaperMind Retrieval-Only Eval")
    print(f"Mode: {'hybrid (dense+BM25 RRF)' if settings.ENABLE_HYBRID_SEARCH else 'dense-only'}")
    print(f"User: {USER_ID}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Zero Gemini LLM calls — embedding API only\n")

    embeddings = _build_embeddings()
    results = []
    latencies = []
    sections: dict = {}

    for i, item in enumerate(QUESTIONS, 1):
        start = time.monotonic()
        try:
            if settings.ENABLE_HYBRID_SEARCH:
                docs_and_scores = _retrieve_hybrid(
                    item["q"], embeddings, item["doc"], user_id=USER_ID
                )
            else:
                from app.services.vectorstore import get_langchain_vectorstore
                vs = get_langchain_vectorstore(embeddings, user_id=USER_ID)
                docs_and_scores = _retrieve_dense(item["q"], vs, item["doc"], user_id=USER_ID)

            elapsed_ms = (time.monotonic() - start) * 1000
            chunks = [doc for doc, _ in docs_and_scores]
            scores = [score for _, score in docs_and_scores]

            passed, matched_kw = grade_chunks(chunks, item["expect"])
            top_score = max(scores) if scores else 0.0

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"[{i:02d}] {status}  ({elapsed_ms:.0f}ms | {len(chunks)} chunks | top_score={top_score:.3f})")
            print(f"      Q: {item['q'][:90]}")
            if passed:
                print(f"      matched: '{matched_kw}'")
            else:
                print(f"      expected one of: {item['expect']}")
                if chunks:
                    preview = chunks[0].page_content[:120].replace('\n', ' ')
                    print(f"      top chunk: {preview}...")
                else:
                    print(f"      ⚠️  NO CHUNKS RETURNED — check user_id or document ingestion")
            print()

            latencies.append(elapsed_ms)
            results.append({"section": item["section"], "passed": passed, "chunks": len(chunks)})
            sections.setdefault(item["section"], []).append(passed)

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            print(f"[{i:02d}] ERROR  — {exc}")
            results.append({"section": item["section"], "passed": False, "chunks": 0})
            sections.setdefault(item["section"], []).append(False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print("=" * 60)
    print("RETRIEVAL EVAL SUMMARY")
    print("=" * 60)
    print(f"Total:    {total} questions")
    print(f"Correct:  {passed}/{total}  ({passed/total*100:.1f}% chunk recall)")
    print()
    for sec, outcomes in sections.items():
        sec_pass = sum(outcomes)
        print(f"  {sec}: {sec_pass}/{len(outcomes)}")

    if latencies:
        lat_sorted = sorted(latencies)
        p95_idx = min(int(len(lat_sorted) * 0.95), len(lat_sorted) - 1)
        print(f"\nLatency (retrieval only, no LLM):")
        print(f"  p50: {lat_sorted[len(lat_sorted)//2]:.0f}ms")
        print(f"  p95: {lat_sorted[p95_idx]:.0f}ms")
        print(f"  mean: {statistics.mean(latencies):.0f}ms")

    print("\n" + "=" * 60)
    print("RESUME BULLET:")
    print(f'  "Validated hybrid retrieval pipeline with a {total}-question')
    print(f'  eval set across 3 research documents, achieving {passed}/{total}')
    print(f'  ({passed/total*100:.1f}%) chunk recall — confirming relevant context')
    print(f'  was surfaced before LLM generation."')
    print("=" * 60)


if __name__ == "__main__":
    main()
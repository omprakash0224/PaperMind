"""
PaperMind load test — concurrent requests against GET /api/documents,
reporting p50/p95/p99 latency and error rate.

WHY /api/documents AND NOT /api/query
  /api/query makes 2 Gemini LLM calls per request — you will hit rate limits
  instantly on the free tier. GET /api/documents hits Qdrant directly with
  zero LLM calls, so you can fire as many concurrent requests as you want.
  It still tests the real things worth claiming on a resume:
    - FastAPI async request handling under concurrency
    - Qdrant multi-tenant filtered scroll performance
    - JWT auth middleware throughput (every request goes through Clerk JWKS)

BEFORE YOU RUN
  1. Start your backend:  uvicorn app.main:app --reload --port 8000
  2. Make sure you have at least a few documents ingested
  3. Get a 1-hour Clerk token and pass it via --token

RUN
  pip install httpx --break-system-packages
  python load_test.py --token "PASTE_TOKEN_HERE" --requests 100 --concurrency 50

OUTPUT
  p50/p95/p99 latency, throughput, error rate — real numbers for your resume.
"""

import argparse
import asyncio
import statistics
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"


async def fire_request(client: httpx.AsyncClient, url: str, headers: dict) -> dict:
    start = time.monotonic()
    try:
        resp = await client.get(url, headers=headers, timeout=30.0)
        elapsed_ms = (time.monotonic() - start) * 1000
        return {"ok": resp.status_code == 200, "status": resp.status_code, "ms": elapsed_ms}
    except httpx.RequestError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {"ok": False, "status": None, "ms": elapsed_ms, "error": str(exc)}


async def run_load_test(base_url: str, token: str, total_requests: int, concurrency: int):
    url = f"{base_url}/api/documents"
    headers = {"Authorization": f"Bearer {token}"}
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client):
        async with semaphore:
            return await fire_request(client, url, headers)

    print(f"Firing {total_requests} requests at GET {url} (max {concurrency} concurrent)\n")

    wall_start = time.monotonic()
    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
    wall_elapsed = time.monotonic() - wall_start

    return results, wall_elapsed


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description="PaperMind GET /api/documents load test")
    parser.add_argument("--token", required=True, help="Clerk bearer JWT (1-hour token recommended)")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--requests", type=int, default=100, help="Total requests to fire (default: 100)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests (default: 50)")
    args = parser.parse_args()

    results, wall_elapsed = asyncio.run(
        run_load_test(args.base_url, args.token, args.requests, args.concurrency)
    )

    latencies_ok = [r["ms"] for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    status_codes: dict = {}
    for r in errors:
        key = str(r.get("status") or "connection_error")
        status_codes[key] = status_codes.get(key, 0) + 1

    total = len(results)
    success = len(latencies_ok)

    print("=" * 60)
    print("LOAD TEST RESULTS  —  GET /api/documents")
    print("=" * 60)
    print(f"Endpoint:            GET /api/documents (Qdrant, no LLM)")
    print(f"Total requests:      {total}")
    print(f"Concurrency:         {args.concurrency}")
    print(f"Successful (200):    {success}")
    print(f"Failed:              {len(errors)}")
    if status_codes:
        print(f"  Failure breakdown: {status_codes}")
    print(f"Wall-clock time:     {wall_elapsed:.2f}s")
    throughput = total / wall_elapsed if wall_elapsed > 0 else float("inf")
    print(f"Throughput:          {throughput:.1f} req/s")
    print()

    if latencies_ok:
        p50 = percentile(latencies_ok, 50)
        p95 = percentile(latencies_ok, 95)
        p99 = percentile(latencies_ok, 99)
        print(f"p50 latency:         {p50:.0f}ms")
        print(f"p95 latency:         {p95:.0f}ms")
        print(f"p99 latency:         {p99:.0f}ms")
        print(f"min / max:           {min(latencies_ok):.0f}ms / {max(latencies_ok):.0f}ms")
        print(f"mean:                {statistics.mean(latencies_ok):.0f}ms")
    else:
        print("No successful requests — check your token and that the backend is running.")

    if status_codes.get("401"):
        print("\n  401 errors — token may have expired. Get a fresh 1-hour token and rerun.")

    print("=" * 60)

    if latencies_ok:
        # Reuse p95 already computed above — no need to recalculate
        print("\nRESUME BULLET:")
        print(f'  "Load tested GET /api/documents at {args.concurrency} concurrent requests,')
        print(f'  achieving p95 latency of {p95:.0f}ms and {throughput:.0f} req/s throughput')
        print(f'  across {total} requests with {success/total*100:.0f}% success rate,')
        print(f'  measured via a Python asyncio/httpx benchmark."')
    print("=" * 60)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
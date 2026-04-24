"""
DocPilot Architecture Benchmark — Standalone (No DB / No Server Required)
===========================================================================
Measures ACTUAL latency and throughput for:
  1. Monolith style  — Direct in-process Python call (no network, in-memory)
  2. Microservice style — HTTP call over loopback (serialization + TCP overhead)

The HTTP target is a tiny embedded FastAPI server that spins up in a background
thread using uvicorn — no external process or real DB needed.

HOW TO RUN:
    poetry run python scratch_benchmark.py

WHAT IT MEASURES:
  NFR 1 — Response Time: mean, median, p95, p99 over 100 iterations
  NFR 2 — Throughput: concurrent requests/second with 10 workers

Results are printed to console AND saved to benchmark_results.json
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
ITERATIONS  = 100   # serial runs per mode
CONCURRENCY = 10    # parallel workers for throughput test
HTTP_PORT   = 19876 # loopback port for the embedded mock server

# ---------------------------------------------------------------------------
# Shared fake data: simulates what a real repository returns
# ---------------------------------------------------------------------------
FAKE_DOCUMENTS = [
    {
        "id": str(uuid4()),
        "title": f"Document {i}",
        "filename": f"doc_{i}.pdf",
        "page_count": i + 1,
    }
    for i in range(20)   # 20 documents — realistic size
]


# ===========================================================================
# PART A: In-process (Monolith) — pure Python, no I/O
# ===========================================================================

class InMemoryRepository:
    """Simulates the Monolith pattern: DocumentRepository accessed in-process."""

    def __init__(self, documents: list[dict]) -> None:
        self._documents = documents

    def list_documents(self) -> list[dict]:
        # Simulate a tiny in-memory scan (what the real Repo does after
        # SQLAlchemy ORM hydrates rows from the connection pool cache)
        return [doc.copy() for doc in self._documents]


_REPO = InMemoryRepository(FAKE_DOCUMENTS)


def monolith_call() -> float:
    """Timed single in-process call. Returns elapsed milliseconds."""
    t0 = time.perf_counter()
    _ = _REPO.list_documents()
    return (time.perf_counter() - t0) * 1_000


def run_monolith_benchmark() -> dict:
    print("\n[1/4] Running in-process (Monolith) benchmark...")
    monolith_call()  # warm-up

    timings = [monolith_call() for _ in range(ITERATIONS)]
    print(f"      Done -- {ITERATIONS} iterations completed.")
    return _compute_stats("Modular Monolith (In-Process)", timings)


# ===========================================================================
# PART B: HTTP loopback (Microservice simulation) — full TCP + JSON cycle
# ===========================================================================

def _start_mock_server() -> None:
    """Start a minimal FastAPI server in a background daemon thread."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    mock_app = FastAPI()

    @mock_app.get("/documents")
    def get_documents():
        # Mirror exactly what the real /api/documents endpoint does:
        # hydrate objects → serialize to JSON → return response
        return JSONResponse(content=FAKE_DOCUMENTS)

    config = uvicorn.Config(
        mock_app,
        host="127.0.0.1",
        port=HTTP_PORT,
        log_level="error",   # suppress request logs during benchmark
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


def _wait_for_server(timeout: float = 10.0) -> bool:
    """Poll until the server is accepting connections."""
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{HTTP_PORT}/documents", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.1)
    return False


def http_call(client) -> float:
    """Timed single HTTP call. Returns elapsed milliseconds."""
    t0 = time.perf_counter()
    resp = client.get(f"http://127.0.0.1:{HTTP_PORT}/documents")
    resp.raise_for_status()
    _ = resp.json()   # include deserialization cost — real microservice must do this
    return (time.perf_counter() - t0) * 1_000


def run_http_benchmark() -> dict:
    import httpx

    print("\n[2/4] Starting embedded mock HTTP server (Microservice simulation)...")
    server_thread = threading.Thread(target=_start_mock_server, daemon=True)
    server_thread.start()

    if not _wait_for_server():
        print("  [ERROR] Mock server failed to start. Exiting.")
        sys.exit(1)
    print(f"      Server ready on port {HTTP_PORT}.")

    print(f"[3/4] Running HTTP benchmark...")
    with httpx.Client() as client:
        http_call(client)  # warm-up
        timings = [http_call(client) for _ in range(ITERATIONS)]

    print(f"      Done -- {ITERATIONS} iterations completed.")
    return _compute_stats("Microservice (HTTP Loopback)", timings)


# ===========================================================================
# PART C: Throughput — concurrent requests per second
# ===========================================================================

def run_throughput_benchmark() -> dict:
    import httpx

    TOTAL_CALLS = CONCURRENCY * 5  # e.g., 10 workers x 5 = 50 total calls

    # ---- Monolith throughput ----
    print(f"\n[4/4] Throughput test -- {CONCURRENCY} concurrent workers x 5 batches...")

    def mono_worker(_): _REPO.list_documents()

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(mono_worker, range(TOTAL_CALLS)))
    mono_rps = round(TOTAL_CALLS / (time.perf_counter() - wall_start), 2)
    print(f"      Monolith:    {TOTAL_CALLS} calls -> {mono_rps} RPS")

    # ---- HTTP throughput ----
    def http_worker(_):
        with httpx.Client() as c:
            c.get(f"http://127.0.0.1:{HTTP_PORT}/documents")

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(http_worker, range(TOTAL_CALLS)))
    http_rps = round(TOTAL_CALLS / (time.perf_counter() - wall_start), 2)
    print(f"      HTTP (uSvc): {TOTAL_CALLS} calls -> {http_rps} RPS")

    return {
        "monolith_rps": mono_rps,
        "http_rps": http_rps,
        "total_calls": TOTAL_CALLS,
        "concurrency": CONCURRENCY,
    }


# ===========================================================================
# Stats + Reporting
# ===========================================================================

def _compute_stats(label: str, timings: list[float]) -> dict:
    s = sorted(timings)
    n = len(s)
    return {
        "label": label,
        "n": n,
        "mean_ms":   round(statistics.mean(timings), 4),
        "median_ms": round(statistics.median(timings), 4),
        "stdev_ms":  round(statistics.stdev(timings), 4),
        "min_ms":    round(min(timings), 4),
        "max_ms":    round(max(timings), 4),
        "p95_ms":    round(s[int(0.95 * n) - 1], 4),
        "p99_ms":    round(s[int(0.99 * n) - 1], 4),
    }


def print_report(mono: dict, http: dict, throughput: dict) -> None:
    sep = "=" * 72
    overhead_mean   = round(http["mean_ms"]   - mono["mean_ms"], 4)
    overhead_median = round(http["median_ms"] - mono["median_ms"], 4)
    overhead_p95    = round(http["p95_ms"]    - mono["p95_ms"], 4)
    overhead_p99    = round(http["p99_ms"]    - mono["p99_ms"], 4)
    speedup         = round(http["mean_ms"] / mono["mean_ms"], 1)
    rps_gain        = round(throughput["monolith_rps"] / throughput["http_rps"], 1)

    print(f"\n{sep}")
    print("  DocPilot Architecture Benchmark -- ACTUAL Results")
    print(f"  Iterations per mode: {ITERATIONS}  |  Concurrency: {CONCURRENCY}")
    print(sep)
    print(f"  {'Metric':<24} {'Monolith':>12} {'HTTP (uSvc)':>12} {'Delta (ms)':>12}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*12}")

    rows = [
        ("Mean Latency",    mono["mean_ms"],   http["mean_ms"],   overhead_mean),
        ("Median Latency", mono["median_ms"], http["median_ms"], overhead_median),
        ("P95 Latency",    mono["p95_ms"],    http["p95_ms"],    overhead_p95),
        ("P99 Latency",    mono["p99_ms"],    http["p99_ms"],    overhead_p99),
        ("Std Dev",        mono["stdev_ms"],  http["stdev_ms"],  None),
        ("Min",            mono["min_ms"],    http["min_ms"],    None),
        ("Max",            mono["max_ms"],    http["max_ms"],    None),
    ]
    for name, m_val, h_val, delta in rows:
        d_str = f"{delta:>+.4f}" if delta is not None else ""
        print(f"  {name + ' (ms)':<24} {m_val:>12.4f} {h_val:>12.4f} {d_str:>12}")

    print(f"  {sep}")
    print(f"  {'Throughput (RPS)':<24} {throughput['monolith_rps']:>12} {throughput['http_rps']:>12}")
    print(sep)
    print(sep)
    print(f"\n  >> HTTP adds {overhead_mean:.4f} ms mean overhead  (x{speedup} slower)")
    print(f"  >> HTTP adds {overhead_p95:.4f} ms at P95")
    print(f"  >> Monolith achieves {rps_gain}x higher throughput under concurrency")
    print(sep)


def save_results(mono: dict, http: dict, throughput: dict) -> None:
    out = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"iterations": ITERATIONS, "concurrency": CONCURRENCY},
        "monolith": mono,
        "http_microservice": http,
        "throughput": throughput,
        "summary": {
            "http_overhead_mean_ms":   round(http["mean_ms"]   - mono["mean_ms"],   4),
            "http_overhead_p95_ms":    round(http["p95_ms"]    - mono["p95_ms"],    4),
            "http_overhead_p99_ms":    round(http["p99_ms"]    - mono["p99_ms"],    4),
            "speedup_factor":          round(http["mean_ms"]   / mono["mean_ms"],   2),
            "monolith_throughput_rps": throughput["monolith_rps"],
            "http_throughput_rps":     throughput["http_rps"],
            "throughput_gain_factor":  round(throughput["monolith_rps"] / throughput["http_rps"], 2),
        },
    }
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {out_path}\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  DocPilot Architecture Benchmark")
    print("  Monolith  = Python in-process function call (zero network)")
    print("  HTTP uSvc = REST call over TCP loopback (serialization + network)")
    print("=" * 72)

    mono_stats = run_monolith_benchmark()
    http_stats = run_http_benchmark()
    throughput = run_throughput_benchmark()

    print_report(mono_stats, http_stats, throughput)
    save_results(mono_stats, http_stats, throughput)

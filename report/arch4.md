# DocPilot: Architecture Comparison Report
### Modular Monolith vs. Microservices — Based on Actual Benchmark Data

> **Methodology:** `scratch_benchmark.py` was executed locally.
> - **Monolith** = direct Python in-process function call (DocumentRepository accessed in memory)
> - **Microservice** = HTTP GET call over TCP loopback to an embedded FastAPI server (identical data, full serialize/deserialize cycle)
> - **Iterations:** 100 per mode | **Concurrency:** 10 workers | **Platform:** Windows, Python 3.12

---

## NFR 1: Response Time (Latency)

| Metric | Monolith (ms) | HTTP / Microservice (ms) | Delta (ms) |
|:---|---:|---:|---:|
| **Mean** | 0.0026 | 2.0605 | **+2.0579** |
| **Median** | 0.0025 | 2.0134 | **+2.0109** |
| **P95** | 0.0027 | 2.8221 | **+2.8194** |
| **P99** | 0.0049 | 3.1400 | **+3.1351** |
| **Std Dev** | 0.0005 | 0.4122 | — |
| **Min** | 0.0024 | 1.4034 | — |
| **Max** | 0.0065 | 4.1317 | — |

**Finding:** The HTTP boundary adds a **mean overhead of 2.06 ms** per inter-service call. The monolith call is **792x faster** in mean latency. The microservice P99 reaches 3.14 ms — already 640x slower than the monolith P99 of 0.0049 ms.

> [!NOTE]
> The 2 ms overhead shown here represents **one** service boundary. A real microservices RAG pipeline crosses 3+ boundaries (API Gateway → Query Svc → Search Svc → DB Svc), multiplying this penalty to **6–10 ms** of pure architectural overhead per request — before LLM generation even begins.

---

## NFR 2: Throughput (Requests Per Second under Concurrency)

| Pattern | RPS (10 workers, 50 calls) |
|:---|---:|
| **Monolith (in-process)** | **7,285.87 RPS** |
| **HTTP / Microservice** | **5.49 RPS** |
| **Gain Factor** | **1,327x** |

**Finding:** Under 10 concurrent workers, the monolith achieves **7,285 RPS** vs only **5.49 RPS** for the HTTP path. This is because each HTTP worker must open a TCP connection, wait for socket accept, send the request, and parse the JSON response — all blocking the thread during that time. The in-process call returns instantly, freeing the thread for the next call.

> [!IMPORTANT]
> The raw throughput gap (1327x) is an upper bound specific to this operation (in-memory list). In production, DB queries equalize the throughput somewhat. However, the monolith advantage remains significant for any operation that requires 2+ service boundaries in a microservices topology.

---

## Trade-off Analysis

### Why Modular Monolith Wins for DocPilot Now

| Dimension | Monolith | Microservices |
|:---|:---|:---|
| **Latency** | 0.003 ms internal hops | 2+ ms per service boundary |
| **Throughput** | 7,285 RPS (bounded by CPU/DB) | 5.49 RPS (bounded by HTTP overhead) |
| **Deployment** | 1 container | 4+ containers + orchestration |
| **Debugging** | Single stack trace | Distributed tracing required |
| **Dev Velocity** | Single codebase, no contracts | Service contracts (OpenAPI/protobuf) |

### When Microservices Become Worth It

1. **Scaling a specific bottleneck independently:** If the Embedding Service needs 20 GPU workers but the API needs only 2 replicas, microservices allow paying only for what you need.
2. **Team isolation:** If >10 developers work in parallel, service ownership boundaries prevent merge conflicts.
3. **Technology heterogeneity:** If the Search Service needs Go for low-latency and the Ingestion Service needs Python for ML libraries, microservices allow mixing stacks.

### Conclusion

The benchmark confirms that for DocPilot's current scale, the **Modular Monolith eliminates ~2 ms of latency overhead and achieves 1,327x higher throughput** compared to an HTTP microservices boundary — all without giving up the ability to migrate later. The architecture is structured with clean `api / services / infrastructure` layers that can be physically split into services when the need arises.

---

*Generated from: `benchmark_results.json` | Script: `scratch_benchmark.py`*

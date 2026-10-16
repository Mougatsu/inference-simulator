# inference-simulator

A local simulation of a self-hosted vLLM inference server: **1× H100 80GB SXM serving Llama-3.1-8B-Instruct (BF16)**.
No model runs. Timing comes from a small model of the GPU's resources (compute, memory bandwidth, KV cache memory, batch slots), so latency, queueing, batching and utilization behave the way a real deployment would.

```
pip install -r requirements.txt
uvicorn server:app --port 8000

curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt_id": 123}'
curl localhost:8000/metrics

locust --host http://localhost:8000                                   # UI at http://localhost:8089
locust --host http://localhost:8000 --headless -u 128 -r 32 -t 60s   # headless
```

| File | What it is |
|---|---|
| `server.py` | `POST /generate`, `GET /metrics`, and the simulated engine (about 200 lines) |
| `data/prompts.jsonl` | Dataset: 1000 records |
| `data/prepare_dataset.py` | Rebuilds the dataset from the source |
| `locustfile.py` | Stress test: each Locust user is one client sending requests back to back |

## Dataset

There are 1000 records sampled (seed 42) from [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0).
These are real human-written prompts, and the categories match the company's use cases: `summarization`, `information_extraction`, and `closed_qa` (document understanding over a supplied context), plus `open_qa`, `general_qa` and `classification` (support-style questions).

Each record is `{id, category, prompt, response}`. The prompt is the instruction followed by the context, if there is one.
**The completion length comes from the real human response**, so the output-length distribution is real and not made up. That matters because output length drives decode time.

| | p50 | p99 | max |
|---|---|---|---|
| prompt tokens | 15 | 969 | 1748 |
| completion tokens | 40 | 540 | 1592 |

## Part 1: Inference simulation

**Tokenizer:** [tiktoken](https://github.com/openai/tiktoken) `cl100k_base`. Llama 3's 128k vocabulary is built on cl100k's 100k BPE merges plus 28k extra tokens, so the counts land within a few percent of the real model's tokenizer. Using it avoids downloading the gated Llama tokenizer files. Earlier versions used a `len(text) / 4` heuristic, which can be off by 10–20% per request.

**Engine:** a single asyncio task owns the "GPU" and runs **continuous batching** the way vLLM's scheduler does. Each loop iteration is one engine step. The stages are commented in `server.py`:

| Stage | What happens | Time model |
|---|---|---|
| 0. Tokenize | Count prompt tokens (CPU) | Negligible |
| 1. Schedule | Move waiting requests into the running batch between steps. The limits are `MAX_NUM_SEQS=256` and free KV cache | – |
| 2. Prefill | Run all newly admitted prompts in one pass and emit their first token. Compute-bound | `1ms + prompt_tokens / 20k tok/s` |
| 3. Decode | One token for *every* running sequence. Memory-bandwidth-bound: weights are read once per step for the whole batch, plus each sequence's KV cache | `1ms + (16GB weights + KV_tokens × 128KiB) / 3.35 TB/s` |
| 4. Complete | Sequences that reached their length leave the batch and free their KV | – |
| 5. Detokenize | Return the dataset's reference answer as the "generated" content | Negligible |

Where the constants come from (top of `server.py`):
- **Decode:** at batch size 1, each step takes about 5.8ms (≈170 tok/s), which is in line with real 8B-on-H100 numbers. Because the weight read is shared by the whole batch, batching multiplies throughput, and the per-token cost rises slowly as KV traffic grows.
- **KV cache:** 2 (K,V) × 32 layers × 8 KV heads × 128 head dim × 2 bytes = **128 KiB per token**. The pool is (80GB × 0.9 − 16GB weights − 4GB activations) ≈ **397k tokens**.
- **Prefill:** 2 × 8B FLOPs per token at about 35% MFU of 989 TFLOPS gives about 20k tok/s.
- **Prefill blocks decode:** prefill steps pause decoding for the running batch, the way vLLM behaves without chunked prefill. This is visible as higher TPOT under load.

**Response:** `content`, `usage.{prompt_tokens, completion_tokens, total_tokens}`, and `timing.{ttft_s, e2e_s}`.
**Errors:** an unknown `prompt_id` returns 404. More than 1000 queued requests returns 429, so the server applies backpressure instead of letting the queue grow without limit.

**Simplifications (on purpose):**
- The KV cache for prompt plus full output is reserved at admission, so there is no paging or preemption. With an 8B model on 80GB, the 256-sequence limit is reached long before the KV cache fills. For a 70B model, KV would become the bottleneck and preemption would need modeling.
- There is no chunked prefill and no prefix caching.
- Timing runs on a virtual GPU clock, so OS timer jitter (about 15ms on Windows) doesn't add up over many steps.

### What it shows (Locust, this laptop)

| Users | req/s | output tok/s | mean TPOT | median E2E | GPU busy |
|---|---|---|---|---|---|
| 1 | 1.65 | ~170 | 5.8 ms | 280 ms | ~89% |
| 128 | 88 | ~7,900 | ~14 ms | 660 ms | ~100% |

Going from 1 to 128 users gives about 45× the throughput, while per-token latency only rises about 2.5×. This is the main economic argument for self-hosting with batching. Above about 256 concurrent users, requests start queueing, and TTFT and E2E grow without bound. That point is where a second GPU is needed.

## Part 2: Observability (`GET /metrics`)

The output is in Prometheus text format, so it can be scraped as-is. The metric names mirror vLLM's own `vllm:*` metrics, so the same dashboards and alerts carry over to production.

| Metric | Type | Why |
|---|---|---|
| `sim_time_to_first_token_seconds` | histogram | The latency users *feel* first. It is queue wait plus prefill |
| `sim_time_per_output_token_seconds` | histogram | Streaming speed. It degrades with batch size and prefill interference |
| `sim_e2e_request_latency_seconds` | histogram | Total latency, the number to compare against the external provider |
| `sim_request_queue_time_seconds` | histogram | Separates "the GPU is slow" from "the GPU is full" |
| `sim_num_requests_running` / `_waiting` | gauge | Batch occupancy and backlog. Waiting > 0 for a sustained period means saturation |
| `sim_kv_cache_usage_ratio` | gauge | Memory pressure. Near 1.0, vLLM preempts and throughput falls off a cliff |
| `sim_gpu_busy_seconds_total` | counter | `rate()` gives GPU utilization (the fraction of wall time spent executing steps) |
| `sim_decode_batch_size` | histogram | Batching efficiency. A small batch at high load points to a scheduling problem |
| `sim_prompt_tokens_total`, `sim_generation_tokens_total` | counter | Token throughput. This is the cost basis compared with per-token API pricing |
| `sim_requests_total{status}` | counter | Success, rejected (429), and not-found counts |
| `sim_kv_cache_capacity_tokens` | gauge | Context for the usage ratio |

Choices:
- **Histograms rather than averages,** because latency SLOs are about p95/p99 and averages hide the tail. The buckets cover 10ms to 60s.
- **TTFT and TPOT are tracked separately** because they come from different bottlenecks (compute versus bandwidth), and the fixes differ too: chunked prefill or more GPUs versus a smaller batch, quantization or speculative decoding.
- **Token counters** are what make a cost comparison with the old API bill possible: GPU-hours ÷ tokens.

### What to alert on

| Alert | Condition (example) | Meaning / action |
|---|---|---|
| TTFT SLO breach | p95 TTFT > 2s for 5m | Users are waiting. Usually caused by queueing: scale out |
| TPOT degradation | p95 TPOT > 50ms for 10m | Streaming feels slow. The batch is too large or prefill is interfering: cap `max_num_seqs` or enable chunked prefill |
| Sustained queue | `waiting > 0` for 5m, or `waiting > 50` | Saturated. This is the leading signal to add a replica, firing before latency alerts |
| KV cache pressure | `kv_cache_usage_ratio > 0.9` for 5m | Preemption is coming. Reduce concurrency or max context, or add memory or GPUs |
| Rejections | `rate(requests_total{status="rejected"}) > 0` | Load is being shed and users are getting errors. Page someone |
| Error rate | non-ok / total > 1% | Something is broken |
| Low utilization | `rate(gpu_busy) < 0.3` for 1h during business hours | Paying for an idle H100. Consolidate the workload or shrink the fleet |

For dashboards, I'd also watch the following (no alerts needed):
- request rate by category
- prompt and output token distributions, to catch workload drift such as longer documents
- tokens per GPU-second, which is the efficiency trend

In production, I'd add hardware metrics from DCGM (SM utilization, HBM usage, temperature, ECC/XID errors) and per-tenant labels for chargeback.

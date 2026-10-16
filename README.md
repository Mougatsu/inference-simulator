# inference-simulator

A local simulation of a self-hosted vLLM deployment: **2× H200 141GB, each GPU running a full replica of [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) (BF16)**, with requests load-balanced across the two.
No model runs. Timing comes from a small model of each GPU's resources (compute, memory bandwidth, KV cache memory, batch slots) and of the model's real architecture, so latency, queueing, batching and utilization behave the way the real deployment would.

## Quick start

**You need:** Python 3.11+ and Docker Desktop (Docker is only needed for the dashboard).

**1. Install**
```
pip install -r requirements.txt
```

**2. Start the simulator** (terminal 1, leave it running)
```
uvicorn server:app --host 0.0.0.0 --port 8000
```

**3. Start the dashboard** (terminal 2)
```
docker compose -f monitoring/docker-compose.yml up -d
```
- **Grafana:** http://localhost:3000 opens the "LLM Inference Simulator" dashboard, no login needed.
- **Prometheus alerts:** http://localhost:9090/alerts

**4. Send one request**
```bash
# macOS / Linux / Git Bash
curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt_id": 123}'
```
```powershell
# Windows PowerShell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/generate -ContentType 'application/json' -Body '{"prompt_id": 123}'
```
The response contains the content, token counts, which GPU served it, and the timing. Raw metrics are at http://localhost:8000/metrics.

**5. Put it under load**
```
locust --host http://localhost:8000
```
Open http://localhost:8089 and start a test, then watch Grafana:

| Users | What you'll see |
|---|---|
| **128** | Normal load: both GPUs busy, latency low, no alerts |
| **800** | Overload, past the 2 × 256 batch limit. Requests queue, TTFT climbs, and `QueueBacklog` and `TTFTHigh` fire within about a minute |

To run without the UI: `locust --host http://localhost:8000 --headless -u 800 -r 200 -t 120s`

**6. Stop**
Press Ctrl+C in terminal 1, then run:
```
docker compose -f monitoring/docker-compose.yml down
```

| File | What it is |
|---|---|
| `server.py` | `POST /generate`, `GET /metrics`, the router and the simulated engines (about 250 lines) |
| `data/prompts.jsonl` | Dataset: 1000 records |
| `data/prepare_dataset.py` | Rebuilds the dataset from the source |
| `locustfile.py` | Stress test: each Locust user is one client sending requests back to back |
| `monitoring/` | Prometheus with the alert rules, and a Grafana dashboard (Docker Compose) |
| `SYSTEM_DESIGN.md` | Part 3: the production scaling memo |

## The deployment being simulated

This is the real vLLM command for the hardware the simulator models. Its numbers (batch limit, memory utilization, precision) are the same constants used in `server.py`:

```bash
vllm serve google/gemma-4-26B-A4B-it \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 4}' \
  --reasoning-parser gemma4 \
  --served-model-name gemma4-fast
```

| Flag | Why |
|---|---|
| `--data-parallel-size 2` | One full replica per H200, and vLLM load-balances between them. The simulator's router does the same |
| `--tensor-parallel-size 1` | The 26B MoE (~50 GB BF16) fits on one GPU, so splitting it would only add communication between GPUs |
| `--gpu-memory-utilization 0.90` | Leaves ~72 GB per GPU for KV cache after weights (`KV_CAPACITY_BYTES`) |
| `--max-num-seqs 256` | Batch limit per replica (`MAX_NUM_SEQS`) |
| `--max-model-len 32768` | Caps context to bound KV use per request. Gemma 4 supports up to 256K |
| `--enable-chunked-prefill` | Long prompts don't stall decoding for the whole batch |
| `--enable-prefix-caching` | Reuses KV for shared system prompts and repeated documents |
| `--async-scheduling` | Overlaps CPU scheduling with GPU execution |
| `--limit-mm-per-prompt '{"image": 4}'` | Allows scanned pages or images in a request, for document understanding |
| `--reasoning-parser gemma4` | Separates thinking from the answer when thinking is enabled |

For function-calling support tooling, add `--tool-call-parser gemma4 --enable-auto-tool-choice`.

## Dataset

There are 1000 records sampled (seed 42) from [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0).
These are real human-written prompts, and the categories match the company's use cases: `summarization`, `information_extraction`, and `closed_qa` (document understanding over a supplied context), plus `open_qa`, `general_qa` and `classification` (support-style questions).

Each record is `{id, category, prompt, response}`. The prompt is the instruction followed by the context, if there is one.
**The completion length comes from the real human response**, so the output-length distribution is real and not made up. That matters because output length drives decode time.

| | p50 | p99 | max |
|---|---|---|---|
| prompt tokens | 15 | 1020 | 1988 |
| completion tokens | 38 | 561 | 1672 |

## Part 1: Inference simulation

**Tokenizer:** `tokens = max(1, round(len(text) / 4.5))`.
- **Where 4.5 comes from:** I calibrated the ratio by running **Gemma 4's own tokenizer** over this dataset; it averages 4.52 characters per token. Tokenizers are model-specific, so the formula is calibrated against the model being simulated.
- **Why not the usual `len/4`:** it overcounts this dataset by about 12%.
- **Limits:** the formula is accurate on average, not per request. Compared with Gemma's tokenizer, the median request is off by about 11%, and 1 in 10 by more than 30%. Text full of names or non-English words packs fewer characters into each token. For example, the opening of the Bismarck prompt (id 636) averages under 3 characters per token. In production you don't estimate: vLLM returns exact `usage` counts from the served model's own tokenizer.

**Router:** each request goes to the replica with the fewest outstanding requests (running plus waiting). That's what vLLM's data-parallel load balancer does. The response includes the `gpu` that served it.

**Engine:** each GPU is an asyncio task running **continuous batching** the way vLLM's scheduler does. Each loop iteration is one engine step. The stages are commented in `server.py`:

| Stage | What happens | Time model |
|---|---|---|
| 0. Tokenize | Count prompt tokens (CPU) | Negligible |
| 1. Schedule | Move waiting requests into the running batch between steps. The limits are `MAX_NUM_SEQS=256` and free KV cache | – |
| 2. Prefill | Run all newly admitted prompts in one pass and emit their first token. Compute-bound | `1ms + prompt_tokens / 30k tok/s` |
| 3. Decode | One token for *every* running sequence. Memory-bandwidth-bound: the weights the batch's experts need, plus every sequence's KV cache | `1ms + (dense 4.7GB + experts 45.7GB × touched + KV) / 3.84 TB/s` |
| 4. Complete | Sequences that reached their length leave the batch and free their KV | – |
| 5. Detokenize | Return the dataset's reference answer as the "generated" content | Negligible |

Where the constants come from (top of `server.py`). The model numbers come from Gemma 4's Hugging Face `config.json`:
- **MoE weights:** 25.2B total and 3.8B active parameters, with 128 experts per layer and 8 routed per token. That works out to about 2.4B always-on parameters (attention, embeddings, vision encoder) and about 22.8B in experts.
  - **Small batches:** a single sequence reads only 8/128 of the expert weights per step, about 2.9 ms/token (~340 tok/s).
  - **Large batches:** the share of experts touched is `1 − (1 − 8/128)^batch`. By about 64 sequences almost every expert is read on every step, so decode is bound by reading the full ~50 GB.
  - **The tradeoff:** MoE saves compute, but at high batch it doesn't save memory bandwidth.
- **KV cache, hybrid attention:** 25 of the 30 layers use a **1024-token sliding window** (8 KV heads × 256 dim = 200 KiB/token, capped at the window). 5 layers use full attention (2 KV heads × 512 dim = 20 KiB/token, growing with context).
  - **Short prompts:** a short request costs about 220 KiB/token.
  - **Long documents:** past 1K tokens, each extra token costs only 20 KiB, so long documents are much cheaper than a dense model's KV would be.
  - **Pool:** (141 GB × 0.9 − 50 GB weights − 4 GB activations) ≈ **72 GB per GPU**.
- **Bandwidth:** H200 peak is 4.8 TB/s; the simulator assumes **80% is achievable**. That's the main calibration knob.
- **Prefill:** 2 × 3.8B FLOPs per token at about 25% MFU of 989 TFLOPS (MoE routing lowers efficiency) gives about 30k tok/s.
- **Prefill blocks decode:** a prefill step pauses decoding for that replica's batch. This is a conservative stand-in for vLLM's chunked prefill, which interleaves the two. It shows up as higher TPOT under load.

**Response:** `content`, `usage.{prompt_tokens, completion_tokens, total_tokens}`, `gpu`, and `timing.{ttft_s, e2e_s}`.
**Errors:** an unknown `prompt_id` returns 404. More than 1000 queued requests (across both GPUs) returns 429, so the server applies backpressure instead of letting the queue grow without limit.

**Simplifications (on purpose):**
- **No paging or preemption:** the KV cache for prompt plus full output is reserved at admission.
  - **This dataset:** requests are short (about 165 tokens), so a full batch of 256 uses about 13% of the KV pool. The 256 batch limit is what binds.
  - **4K-token documents:** 256 sequences would need about 73 GB, more than the pool. KV would then become the limit and `KVCachePressure` would fire.
- No chunked prefill and no prefix caching.
- K and V are counted separately on the full-attention layers. The config's `attention_k_eq_v` may halve that, so KV numbers are slightly conservative.
- Timing runs on a virtual GPU clock, so OS timer jitter (about 15ms on Windows) doesn't add up over many steps.

### What it shows (Locust, this laptop)

| Users | req/s | output tok/s | mean TPOT | median E2E | GPUs busy |
|---|---|---|---|---|---|
| 1 | 3.7 | ~300 | 3.0 ms | 120 ms | GPU0 ~89%, GPU1 idle |
| 128 | 85 | ~6,700 | ~18 ms | 680 ms | both ~94% |
| 512 | 211 | ~15,900 | ~28 ms | 1.1 s | both saturated |
| 800 | 231 | – | – | 2.3 s | saturated; 288 waiting, `QueueBacklog` and `TTFTHigh` firing |

Going from 1 to 512 users gives about **57× the throughput**, while per-token latency rises about 9×.
- **Why latency rises more than for a dense model:** a single MoE sequence is very cheap, but a full batch reads every expert. Each token still streams at about 36 tok/s per user, well inside the 50 ms TPOT alert threshold.
- **The limit:** above about **512 concurrent users** (2 × 256), requests queue, and TTFT and E2E grow without bound. That's the point where a third GPU is needed.

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
| `sim_kv_cache_capacity_bytes` | gauge | KV pool size per GPU, as context for the usage ratio |

Per-replica gauges and GPU busy carry a `gpu` label, so a single hot or idle GPU is visible. Latency histograms are fleet-wide, which is what users experience.

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
| Low utilization | `rate(gpu_busy) < 0.3` for 1h during business hours | Paying for idle H200s. Consolidate the workload or shrink the fleet |

These rules are implemented in [`monitoring/alerts.yml`](monitoring/alerts.yml). Their `for` durations are shortened so the alerts fire within a demo; each rule's `production_for` annotation holds the production value.

### Dashboard (Prometheus + Grafana)

```
uvicorn server:app --host 0.0.0.0 --port 8000        # 0.0.0.0 so the Prometheus container can scrape it
docker compose -f monitoring/docker-compose.yml up -d
locust --host http://localhost:8000 --headless -u 800 -r 200 -t 120s   # past the 2 x 256 batch limit
```

**Grafana** (http://localhost:3000) opens the "LLM Inference Simulator" dashboard, with no login:
- an alerts table showing pending and firing alerts
- headline stats: requests/s, tokens/s, GPU busy, KV cache, running, waiting and 429s
- latency, queue, GPU, KV cache and error charts, with each alert's threshold drawn on the chart

**Prometheus** (http://localhost:9090/alerts) shows the raw alert rule states.

With 800 users, `QueueBacklog` and `TTFTHigh` fire within about a minute. `GPUUnderutilized` fires when the simulator sits idle. `KVCachePressure` doesn't fire with this short-prompt dataset, because the 256 batch limit is reached first. It would fire with 4K-token documents, as explained above.

For dashboards, I'd also watch the following (no alerts needed):
- request rate by category
- prompt and output token distributions, to catch workload drift such as longer documents
- tokens per GPU-second, which is the efficiency trend

In production, I'd add hardware metrics from DCGM (SM utilization, HBM usage, temperature, ECC/XID errors) and per-tenant labels for chargeback.

## Part 3: Scaling to production

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md), the memo on reaching 10,000 req/min. It covers:
- **models:** Gemma 4 26B A4B (fast tier) and gpt-oss-120b (deep tier)
- **hardware:** 3 nodes of 8× H200
- **the rest:** request flow through a LiteLLM gateway, load-aware routing, the observability pipeline, scaling challenges, and reliability

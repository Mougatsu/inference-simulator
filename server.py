"""Simulated vLLM inference server: 1x H100 80GB serving Llama-3.1-8B-Instruct (BF16).

Run: uvicorn server:app --port 8000
"""
import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Hardware / model profile. Every timing below is derived from these numbers.
# ---------------------------------------------------------------------------
HBM_BANDWIDTH = 3.35e12                      # H100 SXM memory bandwidth, bytes/s
WEIGHT_BYTES = 16e9                          # 8B params * 2 bytes (BF16)
KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2    # (K+V) * layers * kv_heads * head_dim * 2 bytes = 128 KiB
# vLLM gpu_memory_utilization=0.9 -> 72GB, minus weights, minus ~4GB activations/CUDA graphs = KV cache pool
KV_CAPACITY_TOKENS = int((80e9 * 0.9 - WEIGHT_BYTES - 4e9) / KV_BYTES_PER_TOKEN)  # ~396k tokens
MAX_NUM_SEQS = 256                           # vLLM default: max sequences in one batch
PREFILL_TOKENS_PER_S = 20_000                # compute-bound: 16 GFLOP/token at ~35% MFU of 989 TFLOPS
STEP_OVERHEAD_S = 0.001                      # scheduler + kernel launch per engine step
MAX_QUEUE = 1000                             # backpressure: reject with 429 beyond this


def count_tokens(text: str) -> int:
    # Tokenizer approximation: ~4 characters per token, the usual rule of thumb for English BPE.
    return max(1, round(len(text) / 4))


# ---------------------------------------------------------------------------
# Dataset: loaded into memory once at startup.
# ---------------------------------------------------------------------------
DATASET = {r["id"]: r for r in map(json.loads, open(Path(__file__).parent / "data" / "prompts.jsonl", encoding="utf-8"))}

# ---------------------------------------------------------------------------
# Metrics (names follow vLLM's own /metrics so dashboards carry over to production).
# ---------------------------------------------------------------------------
LAT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60)
REQUESTS = Counter("sim_requests_total", "Requests by outcome", ["status"])
PROMPT_TOKENS = Counter("sim_prompt_tokens_total", "Prefill tokens processed")
GEN_TOKENS = Counter("sim_generation_tokens_total", "Tokens generated")
GPU_BUSY = Counter("sim_gpu_busy_seconds_total", "Seconds the GPU spent executing steps; rate() = utilization")
RUNNING = Gauge("sim_num_requests_running", "Sequences in the running batch")
WAITING = Gauge("sim_num_requests_waiting", "Requests queued for admission")
KV_USAGE = Gauge("sim_kv_cache_usage_ratio", "Fraction of KV cache blocks reserved")
QUEUE_TIME = Histogram("sim_request_queue_time_seconds", "Arrival -> admitted to batch", buckets=LAT_BUCKETS)
TTFT = Histogram("sim_time_to_first_token_seconds", "Arrival -> first token", buckets=LAT_BUCKETS)
TPOT = Histogram("sim_time_per_output_token_seconds", "Mean inter-token latency per request",
                 buckets=(0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.25))
E2E = Histogram("sim_e2e_request_latency_seconds", "Arrival -> last token", buckets=LAT_BUCKETS)
BATCH = Histogram("sim_decode_batch_size", "Sequences per decode step", buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256))
KV_CAPACITY = Gauge("sim_kv_cache_capacity_tokens", "KV cache pool size in tokens")
KV_CAPACITY.set(KV_CAPACITY_TOKENS)


class Seq:
    def __init__(self, record):
        self.record = record
        self.prompt_tokens = count_tokens(record["prompt"])
        # Output length comes from the dataset's real reference answer, so the workload's
        # output-length distribution is realistic rather than invented.
        self.completion_tokens = count_tokens(record["response"])
        self.kv_tokens = self.prompt_tokens + self.completion_tokens
        self.generated = 0
        self.arrival = time.perf_counter()
        self.done = asyncio.get_running_loop().create_future()


waiting: deque[Seq] = deque()
running: list[Seq] = []
kv_reserved = 0
wake = asyncio.Event()
gpu_clock = 0.0


async def gpu(seconds: float):
    """Occupy the (single) GPU for `seconds`. Sleeps to a deadline on a virtual clock, so OS
    timer jitter (~15ms on Windows) doesn't accumulate: late wakeups shorten the next sleep."""
    global gpu_clock
    gpu_clock += seconds
    GPU_BUSY.inc(seconds)
    await asyncio.sleep(max(0.0, gpu_clock - time.perf_counter()))


async def engine_loop():
    """Continuous batching, modeled on vLLM's scheduler: one loop owns the GPU, and every
    iteration is one engine step over the whole batch."""
    global kv_reserved, gpu_clock
    while True:
        if not waiting and not running:
            wake.clear()
            await wake.wait()
            gpu_clock = time.perf_counter()  # GPU was idle: virtual clock restarts at "now"

        # --- Stage 1: scheduling / admission -----------------------------------------
        # New requests join the running batch between steps (no waiting for the batch to drain),
        # bounded by the max batch size and free KV cache.
        # ponytail: reserves KV for the full output up front, so preemption never happens. vLLM pages
        # blocks lazily and preempts on overflow; model that if KV pressure becomes the question.
        admitted = []
        while waiting and len(running) + len(admitted) < MAX_NUM_SEQS \
                and kv_reserved + waiting[0].kv_tokens <= KV_CAPACITY_TOKENS:
            s = waiting.popleft()
            kv_reserved += s.kv_tokens
            QUEUE_TIME.observe(max(0.0, gpu_clock - s.arrival))
            admitted.append(s)

        if admitted:
            # --- Stage 2: prefill ------------------------------------------------------
            # The whole prompt goes through the model in one pass to fill the KV cache. This is
            # compute-bound, so time scales with prompt tokens. Prefill emits the first token.
            # Like vLLM without chunked prefill, the step pauses decoding for the running batch.
            n = sum(s.prompt_tokens for s in admitted)
            await gpu(STEP_OVERHEAD_S + n / PREFILL_TOKENS_PER_S)
            PROMPT_TOKENS.inc(n)
            for s in admitted:
                s.generated = 1
                s.first_token = gpu_clock
                TTFT.observe(gpu_clock - s.arrival)
            GEN_TOKENS.inc(len(admitted))
            running.extend(admitted)
        else:
            # --- Stage 3: decode -------------------------------------------------------
            # One new token for every sequence in the batch. Decode is memory-bandwidth-bound:
            # each step streams all weights from HBM once (shared by the whole batch, which is
            # why batching helps) plus each sequence's KV cache (which grows every token).
            kv_tokens = sum(s.prompt_tokens + s.generated for s in running)
            await gpu(STEP_OVERHEAD_S + (WEIGHT_BYTES + kv_tokens * KV_BYTES_PER_TOKEN) / HBM_BANDWIDTH)
            BATCH.observe(len(running))
            GEN_TOKENS.inc(len(running))
            for s in running:
                s.generated += 1

        # --- Stage 4: completion -------------------------------------------------------
        # Sequences that hit their stop condition (EOS / length) leave the batch and free KV.
        for s in [s for s in running if s.generated >= s.completion_tokens]:
            running.remove(s)
            kv_reserved -= s.kv_tokens
            E2E.observe(gpu_clock - s.arrival)
            if s.completion_tokens > 1:
                TPOT.observe((gpu_clock - s.first_token) / (s.completion_tokens - 1))
            s.done.set_result(gpu_clock)

        RUNNING.set(len(running))
        WAITING.set(len(waiting))
        KV_USAGE.set(kv_reserved / KV_CAPACITY_TOKENS)


app = FastAPI(title="LLM Inference Simulator")
log = logging.getLogger("uvicorn.error")  # uvicorn's own logger, so lines show in the server output


@app.on_event("startup")
async def start_engine():
    app.state.engine = asyncio.create_task(engine_loop())


class GenerateRequest(BaseModel):
    prompt_id: int


@app.post("/generate")
async def generate(req: GenerateRequest):
    record = DATASET.get(req.prompt_id)
    if record is None:
        REQUESTS.labels("not_found").inc()
        raise HTTPException(404, f"prompt_id must be 0..{len(DATASET) - 1}")
    if len(waiting) >= MAX_QUEUE:
        REQUESTS.labels("rejected").inc()
        raise HTTPException(429, "queue full")

    # --- Stage 0: tokenization (CPU, negligible next to GPU time) ---------------------
    seq = Seq(record)
    # ponytail: truncated so one request = one log line; contexts run up to ~9k chars
    log.info("prompt_id=%d tokens=%d prompt=%r", req.prompt_id, seq.prompt_tokens, record["prompt"][:200])
    waiting.append(seq)
    WAITING.set(len(waiting))
    wake.set()
    finished = await seq.done

    # --- Stage 5: detokenization -------------------------------------------------------
    # The simulation "generates" the dataset's reference answer; only its length drives timing.
    REQUESTS.labels("ok").inc()
    return {
        "prompt_id": req.prompt_id,
        "content": record["response"],
        "usage": {
            "prompt_tokens": seq.prompt_tokens,
            "completion_tokens": seq.completion_tokens,
            "total_tokens": seq.kv_tokens,
        },
        "timing": {
            "ttft_s": round(seq.first_token - seq.arrival, 4),
            "e2e_s": round(finished - seq.arrival, 4),
        },
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

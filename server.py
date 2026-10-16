"""Simulated vLLM deployment: 2x H200 141GB, each GPU a full replica of google/gemma-4-26B-A4B-it (BF16).

Equivalent real deployment: `vllm serve google/gemma-4-26B-A4B-it --data-parallel-size 2 ...` (see README).
Run: uvicorn server:app --host 0.0.0.0 --port 8000
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
# Hardware profile: NVIDIA H200 SXM.
# ---------------------------------------------------------------------------
NUM_GPUS = 2
HBM_BYTES = 141e9
HBM_BANDWIDTH = 4.8e12 * 0.8     # peak 4.8 TB/s; ~80% is achievable in practice. Calibration knob.
GPU_MEMORY_UTILIZATION = 0.9     # vLLM default

# ---------------------------------------------------------------------------
# Model profile: Gemma 4 26B A4B, numbers from its Hugging Face config.json.
# ---------------------------------------------------------------------------
# MoE: 128 experts per layer, 8 routed per token. 25.2B total / 3.8B active params solves to
# ~2.4B always-on (attention, embeddings, vision encoder) + ~22.8B in experts.
DENSE_BYTES = 2.37e9 * 2         # BF16
EXPERT_BYTES = 22.83e9 * 2
NUM_EXPERTS, TOP_K = 128, 8
# Attention: 30 layers = 25 sliding-window (last 1024 tokens, 8 KV heads x 256 dim)
#                       + 5 full-attention (2 KV heads x 512 dim).
SLIDING_WINDOW = 1024
KV_SLIDING_PER_TOKEN = 25 * 2 * 8 * 256 * 2   # (K+V) * BF16 = 200 KiB/token, capped at the window
KV_FULL_PER_TOKEN = 5 * 2 * 2 * 512 * 2       # 20 KiB/token, grows with context
# ponytail: counts K and V separately on full layers; config has attention_k_eq_v, which may halve it.
KV_CAPACITY_BYTES = HBM_BYTES * GPU_MEMORY_UTILIZATION - DENSE_BYTES - EXPERT_BYTES - 4e9  # ~4GB activations -> ~72GB
MAX_NUM_SEQS = 256               # vLLM default: max sequences in one replica's batch
PREFILL_TOKENS_PER_S = 30_000    # compute-bound: 2 * 3.8B FLOP/token at ~25% MFU (MoE routing) of 989 TFLOPS
STEP_OVERHEAD_S = 0.001          # scheduler + kernel launch per engine step
MAX_QUEUE = 1000                 # backpressure across both replicas: 429 beyond this


def count_tokens(text: str) -> int:
    # Tokenizer approximation: 4.5 chars/token, calibrated against Gemma 4's own tokenizer over this
    # dataset (4.52). Right on average; off per request (names, non-English text, code).
    return max(1, round(len(text) / 4.5))


def kv_bytes(tokens: int) -> float:
    """KV cache one sequence holds at a given context length. Sliding layers stop growing past the window."""
    return KV_FULL_PER_TOKEN * tokens + KV_SLIDING_PER_TOKEN * min(tokens, SLIDING_WINDOW)


# ---------------------------------------------------------------------------
# Dataset: loaded into memory once at startup.
# ---------------------------------------------------------------------------
DATASET = {r["id"]: r for r in map(json.loads, open(Path(__file__).parent / "data" / "prompts.jsonl", encoding="utf-8"))}

# ---------------------------------------------------------------------------
# Metrics (names follow vLLM's own /metrics so dashboards carry over to production).
# Per-replica state carries a `gpu` label; latency histograms are fleet-wide.
# ---------------------------------------------------------------------------
LAT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60)
REQUESTS = Counter("sim_requests_total", "Requests by outcome", ["status"])
PROMPT_TOKENS = Counter("sim_prompt_tokens_total", "Prefill tokens processed")
GEN_TOKENS = Counter("sim_generation_tokens_total", "Tokens generated")
GPU_BUSY = Counter("sim_gpu_busy_seconds_total", "Seconds the GPU spent executing steps; rate() = utilization", ["gpu"])
RUNNING = Gauge("sim_num_requests_running", "Sequences in the running batch", ["gpu"])
WAITING = Gauge("sim_num_requests_waiting", "Requests queued for admission", ["gpu"])
KV_USAGE = Gauge("sim_kv_cache_usage_ratio", "Fraction of KV cache reserved", ["gpu"])
QUEUE_TIME = Histogram("sim_request_queue_time_seconds", "Arrival -> admitted to batch", buckets=LAT_BUCKETS)
TTFT = Histogram("sim_time_to_first_token_seconds", "Arrival -> first token", buckets=LAT_BUCKETS)
TPOT = Histogram("sim_time_per_output_token_seconds", "Mean inter-token latency per request",
                 buckets=(0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.25))
E2E = Histogram("sim_e2e_request_latency_seconds", "Arrival -> last token", buckets=LAT_BUCKETS)
BATCH = Histogram("sim_decode_batch_size", "Sequences per decode step", buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256))
KV_CAPACITY = Gauge("sim_kv_cache_capacity_bytes", "KV cache pool size per GPU in bytes")
KV_CAPACITY.set(KV_CAPACITY_BYTES)


class Seq:
    def __init__(self, record):
        self.record = record
        self.prompt_tokens = count_tokens(record["prompt"])
        # Output length comes from the dataset's real reference answer, so the workload's
        # output-length distribution is realistic rather than invented.
        self.completion_tokens = count_tokens(record["response"])
        self.kv_reserve = kv_bytes(self.prompt_tokens + self.completion_tokens)
        self.generated = 0
        self.arrival = time.perf_counter()
        self.done = asyncio.get_running_loop().create_future()


class Engine:
    """One vLLM replica owning one GPU. Continuous batching, modeled on vLLM's scheduler:
    every loop iteration is one engine step over the whole batch."""

    def __init__(self, gpu_id: int):
        self.gpu = str(gpu_id)
        self.waiting: deque[Seq] = deque()
        self.running: list[Seq] = []
        self.kv_reserved = 0.0
        self.wake = asyncio.Event()
        self.clock = 0.0
        for metric in (GPU_BUSY, RUNNING, WAITING, KV_USAGE):
            metric.labels(self.gpu)  # export every GPU from the start, so idle ones show 0, not "no data"

    def load(self) -> int:
        return len(self.waiting) + len(self.running)

    def submit(self, seq: Seq):
        self.waiting.append(seq)
        WAITING.labels(self.gpu).set(len(self.waiting))
        self.wake.set()

    async def run_on_gpu(self, seconds: float):
        """Occupy this GPU for `seconds`. Sleeps to a deadline on a virtual clock, so OS timer
        jitter (~15ms on Windows) doesn't accumulate: late wakeups shorten the next sleep."""
        self.clock += seconds
        GPU_BUSY.labels(self.gpu).inc(seconds)
        await asyncio.sleep(max(0.0, self.clock - time.perf_counter()))

    async def loop(self):
        while True:
            if not self.waiting and not self.running:
                self.wake.clear()
                await self.wake.wait()
                self.clock = time.perf_counter()  # GPU was idle: virtual clock restarts at "now"

            # --- Stage 1: scheduling / admission ---------------------------------------
            # New requests join the running batch between steps (no waiting for the batch to drain),
            # bounded by the max batch size and free KV cache.
            # ponytail: reserves KV for the full output up front, so preemption never happens. vLLM pages
            # blocks lazily and preempts on overflow; model that if KV pressure becomes the question.
            admitted = []
            while self.waiting and len(self.running) + len(admitted) < MAX_NUM_SEQS \
                    and self.kv_reserved + self.waiting[0].kv_reserve <= KV_CAPACITY_BYTES:
                s = self.waiting.popleft()
                self.kv_reserved += s.kv_reserve
                QUEUE_TIME.observe(max(0.0, self.clock - s.arrival))
                admitted.append(s)

            if admitted:
                # --- Stage 2: prefill --------------------------------------------------
                # The whole prompt goes through the model in one pass to fill the KV cache. This is
                # compute-bound, so time scales with prompt tokens. Prefill emits the first token.
                # The step pauses decoding for the running batch: a conservative stand-in for vLLM's
                # chunked prefill, which interleaves the two.
                n = sum(s.prompt_tokens for s in admitted)
                await self.run_on_gpu(STEP_OVERHEAD_S + n / PREFILL_TOKENS_PER_S)
                PROMPT_TOKENS.inc(n)
                for s in admitted:
                    s.generated = 1
                    s.first_token = self.clock
                    TTFT.observe(self.clock - s.arrival)
                GEN_TOKENS.inc(len(admitted))
                self.running.extend(admitted)
            else:
                # --- Stage 3: decode ---------------------------------------------------
                # One new token for every sequence in the batch. Decode is memory-bandwidth-bound:
                # each step streams weights from HBM once for the whole batch (why batching helps),
                # plus every sequence's KV cache.
                # MoE: a token only touches 8 of 128 experts per layer, so a small batch reads a small
                # slice of the expert weights; by ~64 sequences almost every expert is hit each step.
                touched = 1 - (1 - TOP_K / NUM_EXPERTS) ** len(self.running)
                weights_read = DENSE_BYTES + EXPERT_BYTES * touched
                kv_read = sum(kv_bytes(s.prompt_tokens + s.generated) for s in self.running)
                await self.run_on_gpu(STEP_OVERHEAD_S + (weights_read + kv_read) / HBM_BANDWIDTH)
                BATCH.observe(len(self.running))
                GEN_TOKENS.inc(len(self.running))
                for s in self.running:
                    s.generated += 1

            # --- Stage 4: completion ---------------------------------------------------
            # Sequences that hit their stop condition (EOS / length) leave the batch and free KV.
            for s in [s for s in self.running if s.generated >= s.completion_tokens]:
                self.running.remove(s)
                self.kv_reserved -= s.kv_reserve
                E2E.observe(self.clock - s.arrival)
                if s.completion_tokens > 1:
                    TPOT.observe((self.clock - s.first_token) / (s.completion_tokens - 1))
                s.done.set_result(self.clock)

            RUNNING.labels(self.gpu).set(len(self.running))
            WAITING.labels(self.gpu).set(len(self.waiting))
            KV_USAGE.labels(self.gpu).set(self.kv_reserved / KV_CAPACITY_BYTES)


ENGINES = [Engine(i) for i in range(NUM_GPUS)]
app = FastAPI(title="LLM Inference Simulator")
log = logging.getLogger("uvicorn.error")  # uvicorn's own logger, so lines show in the server output


@app.on_event("startup")
async def start_engines():
    app.state.tasks = [asyncio.create_task(e.loop()) for e in ENGINES]


class GenerateRequest(BaseModel):
    prompt_id: int


@app.post("/generate")
async def generate(req: GenerateRequest):
    record = DATASET.get(req.prompt_id)
    if record is None:
        REQUESTS.labels("not_found").inc()
        raise HTTPException(404, f"prompt_id must be 0..{len(DATASET) - 1}")
    if sum(len(e.waiting) for e in ENGINES) >= MAX_QUEUE:
        REQUESTS.labels("rejected").inc()
        raise HTTPException(429, "queue full")

    # --- Stage 0: tokenization (CPU, negligible next to GPU time) ---------------------
    seq = Seq(record)
    # ponytail: truncated so one request = one log line; contexts run up to ~9k chars
    log.info("prompt_id=%d tokens=%d prompt=%r", req.prompt_id, seq.prompt_tokens, record["prompt"][:200])

    # --- Routing: least outstanding requests, like vLLM's data-parallel load balancer ---
    engine = min(ENGINES, key=Engine.load)
    engine.submit(seq)
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
            "total_tokens": seq.prompt_tokens + seq.completion_tokens,
        },
        "gpu": int(engine.gpu),
        "timing": {
            "ttft_s": round(seq.first_token - seq.arrival, 4),
            "e2e_s": round(finished - seq.arrival, 4),
        },
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

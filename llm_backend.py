#!python
"""
LLM Backend abstraction for benchmark scripts.

Supports:
  - anthropic_batch: Anthropic Batch API (default, async, cheap)
  - anthropic_direct: Anthropic Messages API (synchronous)
  - openai: OpenAI-compatible API (ollama, vllm, lmstudio, etc.)

Usage:
    backend = LLM_CreateBackend("openai", model="qwen2.5:32b",
                                 base_url="http://localhost:11434/v1")
    results = backend.run_requests(requests)
"""

import json
import time
import sys

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


class AnthropicBatchBackend:
    """Anthropic Batch API — async, 50% cheaper."""

    def __init__(self, model, judge_model=None, chunk_size=5, poll_interval=30,
                 max_retries=5):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.judgeModel = judge_model or model
        self.chunkSize = chunk_size
        self.pollInterval = poll_interval
        self.maxRetries = max_retries
        self.name = "anthropic_batch"

    def run_requests(self, requests):
        """Submit requests as batches and wait for results.

        requests: list of dicts with 'custom_id' and 'params' keys.
        Returns: dict {custom_id: {"status": str, "text": str}}
        """
        batches = []
        for i in range(0, len(requests), self.chunkSize):
            chunk = requests[i:i + self.chunkSize]
            for attempt in range(self.maxRetries):
                try:
                    batch = self.client.messages.batches.create(requests=chunk)
                    print(f"  Batch submitted: {batch.id} ({len(chunk)} requests) "
                          f"[{i+1}-{i+len(chunk)}/{len(requests)}]")
                    batches.append(batch)
                    break
                except Exception as e:
                    if attempt < self.maxRetries - 1:
                        wait = 10 * (2 ** attempt)
                        print(f"  Batch submit failed ({attempt+1}/{self.maxRetries}): {e}")
                        print(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  FATAL: {self.maxRetries} attempts exhausted.")
                        raise

        results = {}
        for batch in batches:
            results.update(self._wait_batch(batch.id))
        return results

    def _wait_batch(self, batchId):
        startTime = time.time()
        while True:
            status = self.client.messages.batches.retrieve(batchId)
            counts = status.request_counts
            total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
            done = counts.succeeded + counts.errored + counts.canceled + counts.expired
            elapsed = time.time() - startTime
            print(f"\r  Batch {batchId[:20]}...: {done}/{total} "
                  f"(ok={counts.succeeded} err={counts.errored}) [{elapsed:.0f}s]    ",
                  end="")
            if status.processing_status == "ended":
                print()
                break
            time.sleep(self.pollInterval)

        results = {}
        for result in self.client.messages.batches.results(batchId):
            cid = result.custom_id
            if result.result.type == "succeeded":
                text = ""
                for block in result.result.message.content:
                    if block.type == "text":
                        text = block.text
                        break
                results[cid] = {"status": "succeeded", "text": text}
            else:
                results[cid] = {"status": result.result.type,
                                "text": f"[{result.result.type}]"}

        succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
        print(f"  Batch complete: {succeeded}/{len(results)} succeeded")
        return results


class OpenAIBackend:
    """OpenAI-compatible API — synchronous, works with local LLMs."""

    def __init__(self, model, judge_model=None, base_url="http://localhost:11434/v1",
                 api_key="ollama", max_retries=3, timeout=None, num_ctx=32768):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.judgeModel = judge_model or model
        self.maxRetries = max_retries
        self.timeout = timeout
        self.numCtx = num_ctx
        self.name = "openai"
        self.baseUrl = base_url

    def _call_one(self, req, idx=0, total=1):
        """Execute a single request. Returns (custom_id, result_dict)."""
        cid = req["custom_id"]
        params = req["params"]

        messages = []
        if "system" in params:
            messages.append({"role": "system", "content": params["system"]})
        for msg in params.get("messages", []):
            messages.append({"role": msg["role"], "content": msg["content"]})

        model = params.get("model", self.model)
        maxTokens = params.get("max_tokens", 4096)

        for attempt in range(self.maxRetries):
            try:
                extraBody = {}
                if self.numCtx and "ollama" in self.baseUrl:
                    extraBody["options"] = {"num_ctx": self.numCtx}
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=maxTokens,
                    temperature=0,
                    extra_body=extraBody if extraBody else None,
                )
                text = response.choices[0].message.content or ""
                print(f"\r  [{idx+1}/{total}] {cid}: OK ({len(text)} chars)    ",
                      end="")
                return cid, {"status": "succeeded", "text": text}
            except Exception as e:
                if attempt < self.maxRetries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"\n  [{idx+1}/{total}] {cid}: failed ({e}), retry in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"\n  [{idx+1}/{total}] {cid}: FAILED after {self.maxRetries} attempts")
                    return cid, {"status": "error", "text": f"[error: {e}]"}

    def run_requests(self, requests):
        """Run requests sequentially via chat completions.

        requests: list of dicts with 'custom_id' and 'params' keys.
        Returns: dict {custom_id: {"status": str, "text": str}}
        """
        results = {}
        total = len(requests)
        for idx, req in enumerate(requests):
            cid, result = self._call_one(req, idx, total)
            results[cid] = result
        print()
        succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
        print(f"  Complete: {succeeded}/{total} succeeded")
        return results


class WrapperBackend(OpenAIBackend):
    """claude-code-openai-wrapper — uses CC Max subscription, parallel workers."""

    def __init__(self, model="claude-sonnet-4-20250514", judge_model=None,
                 base_url="http://localhost:8082/v1", api_key="none",
                 max_retries=3, timeout=None, workers=4):
        super().__init__(model=model, judge_model=judge_model, base_url=base_url,
                         api_key=api_key, max_retries=max_retries, timeout=timeout,
                         num_ctx=0)
        self.workers = workers
        self.name = "wrapper"

    def run_requests(self, requests):
        """Run requests in parallel via ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        total = len(requests)
        done = 0

        print(f"  Launching {total} requests with {self.workers} workers...")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._call_one, req, idx, total): req["custom_id"]
                for idx, req in enumerate(requests)
            }
            for future in as_completed(futures):
                cid, result = future.result()
                results[cid] = result
                done += 1
                if result["status"] == "succeeded":
                    print(f"\r  [{done}/{total}] completed    ", end="")
                else:
                    print(f"\n  [{done}/{total}] {cid}: {result['status']}")

        print()
        succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
        print(f"  Complete: {succeeded}/{total} succeeded")
        return results


def LLM_CreateBackend(backend_type, model, judge_model=None, **kwargs):
    """Factory function to create the appropriate backend.

    backend_type: "anthropic_batch", "openai", "wrapper"
    model: model name (e.g. "claude-haiku-4-5-20251001", "qwen2.5:32b")
    kwargs: backend-specific options (base_url, api_key, chunk_size, workers, etc.)
    """
    if backend_type == "anthropic_batch":
        return AnthropicBatchBackend(
            model=model,
            judge_model=judge_model,
            chunk_size=kwargs.get("chunk_size", 5),
            poll_interval=kwargs.get("poll_interval", 30),
            max_retries=kwargs.get("max_retries", 5),
        )
    elif backend_type == "openai":
        return OpenAIBackend(
            model=model,
            judge_model=judge_model,
            base_url=kwargs.get("base_url") or "http://localhost:11434/v1",
            api_key=kwargs.get("api_key") or "ollama",
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 300),
        )
    elif backend_type == "wrapper":
        return WrapperBackend(
            model=model or "claude-sonnet-4-20250514",
            judge_model=judge_model,
            base_url=kwargs.get("base_url") or "http://localhost:8082/v1",
            api_key=kwargs.get("api_key") or "none",
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", None),
            workers=kwargs.get("workers", 4),
        )
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Use 'anthropic_batch', 'openai', or 'wrapper'")

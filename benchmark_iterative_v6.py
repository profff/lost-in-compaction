#!python3
"""
Iterative compaction benchmark v6 — Compare compaction strategies on long conversations.

Simulates a 5M-token conversation being managed in a 190K context window through
repeated compaction cycles. Measures what each strategy preserves.

Strategies:
    S1 = Brutal       (compact everything except last turn, re-summarize each cycle)
    S2 = Incremental  (dual watermark, compact just enough)
    S3 = Frozen        (freeze summaries, merge oldest when budget exceeded)
    S4 = FrozenRanked  (freeze + rank-aware hierarchical merging)

Prerequisites:
    - Pre-built conversation: python build_conversation_v6.py --density 80
    - ANTHROPIC_API_KEY in .env

Usage:
    ./benchmark_iterative_v6.py --density 80 --dry-run
    ./benchmark_iterative_v6.py --density 80
    ./benchmark_iterative_v6.py --density 80 --strategies S1,S3
    ./benchmark_iterative_v6.py --density 80 --skip-feed
"""

import json
import os
import re
import time
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# IMPORTS
# ============================================================================

from compaction import (
    estimate_tokens, messages_to_text,
    BrutalCompactor, ContextCompactor, FrozenCompactor, FrozenRankedCompactor,
)
from benchmark_compaction_v2 import RateLimitedLLM, WrapperLLM


# ============================================================================
# CONSTANTS
# ============================================================================

CONTEXT_WINDOW = 190_000
CHARS_PER_TOKEN = 4.3

SYSTEM_PROMPT = ("You are a helpful assistant working on a complex software project. "
                 "Answer questions precisely from memory.")

STRATEGIES = {
    "S1": ("brutal", BrutalCompactor),
    "S2": ("incremental", ContextCompactor),
    "S3": ("frozen", FrozenCompactor),
    "S4": ("frozen_ranked", FrozenRankedCompactor),
}

# Batch API prompts (from benchmark_compaction_v5.py)
BATCH_QUESTION_PROMPT = """Answer each of the following questions based ONLY on what you know from our conversation.
Be specific: include exact numbers, names, paths, versions.
If you don't remember or aren't sure, say "I don't recall".

{questions}

Reply with a JSON array of objects, one per question:
[{{"id": "LM_0001", "answer": "your answer"}}, ...]

IMPORTANT: Return ONLY the JSON array, no other text."""

BATCH_JUDGE_PROMPT = """Evaluate whether each answer is correct.

{entries}

For each entry, check:
1. "recalled": Does the answer demonstrate knowledge of the expected information? "I don't recall" = false.
2. "accurate": Does the answer match the expected answer? Partial matches or wrong values = false.
   Example: expected "17 days", answer "10 days" → recalled=true (knows about days), accurate=false (wrong number).

Reply with ONLY a JSON array:
[{{"id": "LM_0001", "recalled": true/false, "accurate": true/false, "notes": "brief reason"}}, ...]"""

JUDGE_SYSTEM = "You are an objective evaluator. Answer ONLY with valid JSON."

QA_CHUNK_SIZE = 20
JUDGE_CHUNK_SIZE = 50


# ============================================================================
# UTILITIES
# ============================================================================

def save_json(data, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def parse_llm_json(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(clean)


def estimate_tokens_chars(chars):
    return int(chars / CHARS_PER_TOKEN)


# ============================================================================
# CONVERSATION LOADING
# ============================================================================

def load_conversation(density, targetTokens, seed=42):
    """Load pre-built conversation from build_conversation_v6.py."""
    convDir = Path("data/conversations/v6_R4")
    tokLabel = f"{targetTokens // 1_000_000}M" if targetTokens >= 1_000_000 else f"{targetTokens // 1_000}K"
    convFile = convDir / f"d{density}_{tokLabel}_seed{seed}.json"
    metaFile = convDir / f"d{density}_{tokLabel}_seed{seed}_meta.json"

    if not convFile.exists():
        raise FileNotFoundError(
            f"Conversation not found: {convFile}\n"
            f"Run: python build_conversation_v6.py --density {density} --target-tokens {targetTokens}")

    print(f"  Loading conversation: {convFile.name}...")
    with open(convFile, encoding="utf-8") as f:
        messages = json.load(f)
    with open(metaFile, encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"    {len(messages)} messages, ~{metadata['est_tokens']:,} tokens, "
          f"{len(metadata['facts'])} facts")
    return messages, metadata


def extract_facts(metadata):
    """Extract fact tuples from metadata."""
    facts = []
    for fm in metadata["facts"]:
        facts.append((
            fm["fact_id"],
            fm["question"],
            fm["answer"],
            fm["keywords"],
            fm.get("question_type", "unknown"),
        ))
    return facts


# ============================================================================
# TRACKED COMPACTOR WRAPPER
# ============================================================================

class TrackedCompactor:
    """Wraps any compactor with fact tracking through compaction cycles."""

    def __init__(self, compactor, factMeta, totalConversationMsgs):
        self.inner = compactor
        self.totalConversationMsgs = totalConversationMsgs

        # Per-fact tracking
        self.factStatus = {}
        for fm in factMeta:
            self.factStatus[fm["fact_id"]] = {
                "original_msg_start": fm["message_start"],
                "original_msg_end": fm["message_end"],
                "status": "not_yet_fed",
                "compacted_at_cycle": None,
                "n_compressions": 0,
                "keyword_survived_last": None,
            }

        self.factMeta = {fm["fact_id"]: fm for fm in factMeta}
        self.cycleLog = []
        self.fedUpTo = 0  # how many messages from original conversation have been fed

    def update_fed(self, newFedUpTo):
        """Update which facts are now live in context."""
        self.fedUpTo = newFedUpTo
        for fid, info in self.factStatus.items():
            if info["status"] == "not_yet_fed":
                if newFedUpTo >= info["original_msg_end"]:
                    info["status"] = "raw"

    def should_compact(self, messages, system=""):
        return self.inner.should_compact(messages, system)

    def compact(self, messages, llm, system=""):
        """Compact with fact tracking."""
        msgCountBefore = len(messages)

        result = self.inner.compact(messages, llm, system)

        if result.get("compacted"):
            msgCountAfter = len(messages)
            msgsRemoved = msgCountBefore - msgCountAfter

            # The compactor removed the oldest msgsRemoved messages
            # and replaced them with 2 summary messages (user + assistant)
            # So messages that were at index 0..msgsRemoved-1+2 are gone
            # The new messages[0:2] are the summary pair

            cycleNum = result.get("compactionNumber", len(self.cycleLog) + 1)

            # Check which raw facts got compacted
            affectedFacts = []
            for fid, info in self.factStatus.items():
                if info["status"] == "raw":
                    # This fact was live but may have been in the compacted range
                    info["n_compressions"] += 1
                    if info["n_compressions"] == 1:
                        info["status"] = f"compacted_cycle_{cycleNum}"
                        info["compacted_at_cycle"] = cycleNum
                        affectedFacts.append(fid)

            # For facts already compacted (in a previous summary), they got
            # re-summarized if their summary was in the compacted range
            # This is tracked by incrementing n_compressions
            for fid, info in self.factStatus.items():
                if info["status"].startswith("compacted_cycle_") and fid not in affectedFacts:
                    # Check if this fact was re-compressed
                    # For BrutalCompactor: everything gets re-compressed each time
                    # For Incremental: depends on watermark
                    # We can't know exactly, but n_compressions tracks it via
                    # the compactor's internal behavior
                    pass

            # Grep keywords in the current context
            fullText = ""
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    fullText += content.lower() + "\n"

            for fid in affectedFacts:
                kws = self.factMeta[fid]["keywords"]
                found = sum(1 for kw in kws if kw.lower() in fullText)
                self.factStatus[fid]["keyword_survived_last"] = found == len(kws)

            self.cycleLog.append({
                "cycle": cycleNum,
                "msgs_before": msgCountBefore,
                "msgs_after": msgCountAfter,
                "tokens_before": result.get("tokensBefore", 0),
                "tokens_after": result.get("tokensAfterEst", 0),
                "tokens_freed": result.get("tokensFreed", 0),
                "facts_affected": affectedFacts,
                "type": result.get("type", "compact"),
                "merge": result.get("merge"),
                "fed_progress": self.fedUpTo,
            })

        return result

    def get_tracking_summary(self):
        """Return final tracking state."""
        return {
            "fact_status": self.factStatus,
            "cycle_log": self.cycleLog,
            "total_cycles": len(self.cycleLog),
        }


# ============================================================================
# FEED + COMPACT LOOP
# ============================================================================

def feed_strategy(allMessages, compactor, llm, label, factMeta,
                  batchSize=10, system="", checkpoints=None):
    """Feed messages in batches, compact at watermark, track facts.

    Args:
        checkpoints: list of token thresholds (e.g. [500000, 1000000, 2000000]).
            At each threshold, yields (messages_copy, tracking_copy, fed_tokens).
            If None, no checkpoints — just runs to completion.

    Returns (finalMessages, trackingData, checkpointSnapshots).
        checkpointSnapshots: list of (messages, tracking, fed_tokens) at each checkpoint.
    """
    tracker = TrackedCompactor(compactor, factMeta, len(allMessages))
    messages = []
    consecutiveFailures = 0
    maxFailures = 3
    checkpointSnapshots = []
    pendingCheckpoints = list(checkpoints) if checkpoints else []
    fedChars = 0

    total = len(allMessages)
    for i in range(0, total, batchSize):
        batch = allMessages[i:i + batchSize]
        messages.extend(batch)
        tracker.update_fed(i + len(batch))
        fedChars += sum(len(m.get("content", "")) for m in batch)
        fedTokens = estimate_tokens_chars(fedChars)

        if hasattr(compactor, 'lastInputTokens'):
            compactor.lastInputTokens = estimate_tokens(messages, system)

        if consecutiveFailures >= maxFailures:
            # Check checkpoints even when compaction failed
            if pendingCheckpoints and fedTokens >= pendingCheckpoints[0]:
                cpTok = pendingCheckpoints.pop(0)
                print(f"  [{label}] CHECKPOINT @~{cpTok//1000}K "
                      f"(fed ~{fedTokens:,} tok, {len(messages)} msgs, "
                      f"{len(tracker.cycleLog)} cycles)")
                checkpointSnapshots.append((
                    deepcopy(messages),
                    deepcopy(tracker.get_tracking_summary()),
                    fedTokens,
                ))
            continue

        if tracker.should_compact(messages, system):
            result = tracker.compact(messages, llm, system)
            if result.get("compacted"):
                extra = ""
                if result.get("type") == "freeze":
                    extra = f", frozen#{result.get('frozenSummaries', '?')}"
                if result.get("merge"):
                    m = result["merge"]
                    extra += f" +merge#{m.get('mergeNumber', '?')}"
                    if "newRank" in m:
                        extra += f" R{m.get('mergedRanks','?')}->R{m['newRank']}"
                print(f"  [{label}] Cycle {result.get('compactionNumber', '?')}: "
                      f"{result.get('messagesCompacted', '?')} msgs -> "
                      f"{result.get('messagesRemaining', '?')} remaining, "
                      f"~{result.get('tokensFreed', 0):,} freed{extra} "
                      f"(fed {min(i + batchSize, total)}/{total})")
                consecutiveFailures = 0
            else:
                consecutiveFailures += 1
                reason = result.get("reason", "unknown")
                print(f"  [{label}] FAILED ({consecutiveFailures}/{maxFailures}): {reason}")
                if consecutiveFailures >= maxFailures:
                    print(f"  [{label}] Giving up after {maxFailures} failures")

        # Check if we crossed a checkpoint
        if pendingCheckpoints and fedTokens >= pendingCheckpoints[0]:
            cpTok = pendingCheckpoints.pop(0)
            print(f"  [{label}] CHECKPOINT @~{cpTok//1000}K "
                  f"(fed ~{fedTokens:,} tok, {len(messages)} msgs, "
                  f"{len(tracker.cycleLog)} cycles)")
            checkpointSnapshots.append((
                deepcopy(messages),
                deepcopy(tracker.get_tracking_summary()),
                fedTokens,
            ))

    # Final token count
    finalTokens = estimate_tokens(messages, system)
    print(f"  [{label}] Done: {len(messages)} msgs, ~{finalTokens:,} tokens, "
          f"{len(tracker.cycleLog)} compaction cycles")

    return messages, tracker.get_tracking_summary(), checkpointSnapshots


# ============================================================================
# GREP SCAN
# ============================================================================

def grep_keywords(messages, facts):
    fullText = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            fullText += content.lower() + "\n"

    results = []
    for factId, _, _, keywords, qType in facts:
        found = [kw for kw in keywords if kw.lower() in fullText]
        results.append({
            "fact_id": factId,
            "question_type": qType,
            "keywords": keywords,
            "keywords_found": found,
            "all_present": len(found) == len(keywords) and len(keywords) > 0,
            "any_present": len(found) > 0,
        })
    return results


def summarize_grep(grepResults):
    total = len(grepResults)
    allPresent = sum(1 for r in grepResults if r["all_present"])
    anyPresent = sum(1 for r in grepResults if r["any_present"])
    return {
        "facts_total": total,
        "all_present": allPresent,
        "any_present": anyPresent,
        "recall_upper_bound": allPresent / total if total > 0 else 0,
    }


# ============================================================================
# BATCH API
# ============================================================================

def submit_chunked(client, requests, chunkSize, description=""):
    batches = []
    for i in range(0, len(requests), chunkSize):
        chunk = requests[i:i + chunkSize]
        batch = client.messages.batches.create(requests=chunk)
        print(f"  Batch submitted: {batch.id} ({len(chunk)} requests) "
              f"{description} [{i+1}-{i+len(chunk)}/{len(requests)}]")
        batches.append(batch)
    return batches


def wait_for_batch(client, batchId, pollInterval=30):
    startTime = time.time()
    while True:
        status = client.messages.batches.retrieve(batchId)
        counts = status.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        elapsed = time.time() - startTime
        print(f"\r  Batch {batchId[:20]}...: {done}/{total} "
              f"(ok={counts.succeeded} err={counts.errored}) [{elapsed:.0f}s]    ", end="")
        if status.processing_status == "ended":
            print()
            break
        time.sleep(pollInterval)

    results = {}
    for result in client.messages.batches.results(batchId):
        cid = result.custom_id
        if result.result.type == "succeeded":
            text = ""
            for block in result.result.message.content:
                if block.type == "text":
                    text = block.text
                    break
            results[cid] = {"status": "succeeded", "text": text}
        else:
            errMsg = getattr(result.result, 'error', None)
            errDetail = str(errMsg) if errMsg else result.result.type
            results[cid] = {"status": result.result.type, "text": f"[{result.result.type}]"}

    succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
    failed = len(results) - succeeded
    print(f"  Batch complete: {succeeded}/{len(results)} succeeded")
    if failed > 0:
        firstErr = next((r for r in results.values() if r["status"] != "succeeded"), None)
        print(f"  WARNING: {failed} failed — first error: {firstErr['status'] if firstErr else '?'}")
    return results


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(verdicts, factMeta, tracking=None):
    """Compute recall metrics with zone + category breakdown."""
    total = len(verdicts)
    recalled = sum(1 for v in verdicts if v.get("recalled"))
    accurate = sum(1 for v in verdicts if v.get("recalled") and v.get("accurate"))

    # Position zone breakdown (based on original conversation position)
    factPositions = {fm["fact_id"]: fm["position_pct"] for fm in factMeta}
    early = [v for v in verdicts if factPositions.get(v["fact_id"], 0) < 20]
    midEarly = [v for v in verdicts if 20 <= factPositions.get(v["fact_id"], 0) < 40]
    mid = [v for v in verdicts if 40 <= factPositions.get(v["fact_id"], 0) < 60]
    midLate = [v for v in verdicts if 60 <= factPositions.get(v["fact_id"], 0) < 80]
    late = [v for v in verdicts if factPositions.get(v["fact_id"], 0) >= 80]

    def zone_recall(zone):
        if not zone:
            return 0.0
        return sum(1 for v in zone if v.get("recalled")) / len(zone)

    # Per-category breakdown
    factTypes = {fm["fact_id"]: fm.get("question_type", "unknown") for fm in factMeta}
    byCategory = defaultdict(lambda: {"total": 0, "recalled": 0, "accurate": 0})
    for v in verdicts:
        cat = factTypes.get(v["fact_id"], "unknown")
        byCategory[cat]["total"] += 1
        if v.get("recalled"):
            byCategory[cat]["recalled"] += 1
        if v.get("recalled") and v.get("accurate"):
            byCategory[cat]["accurate"] += 1

    categoryMetrics = {}
    for cat, stats in sorted(byCategory.items()):
        categoryMetrics[cat] = {
            "total": stats["total"],
            "recalled": stats["recalled"],
            "accurate": stats["accurate"],
            "recall": stats["recalled"] / stats["total"] if stats["total"] > 0 else 0,
            "accuracy": stats["accurate"] / stats["total"] if stats["total"] > 0 else 0,
        }

    result = {
        "recall": recalled / total if total > 0 else 0,
        "accuracy": accurate / total if total > 0 else 0,
        "facts_recalled": recalled,
        "facts_total": total,
        "recall_q1": zone_recall(early),
        "recall_q2": zone_recall(midEarly),
        "recall_q3": zone_recall(mid),
        "recall_q4": zone_recall(midLate),
        "recall_q5": zone_recall(late),
        "n_q1": len(early),
        "n_q2": len(midEarly),
        "n_q3": len(mid),
        "n_q4": len(midLate),
        "n_q5": len(late),
        "by_category": categoryMetrics,
    }

    # Compaction depth breakdown (if tracking available)
    if tracking:
        factStatus = tracking.get("fact_status", {})
        compressions = {}
        for fid, info in factStatus.items():
            compressions[fid] = info.get("n_compressions", 0)

        # Recall by compression count
        byCompression = defaultdict(lambda: {"total": 0, "recalled": 0})
        for v in verdicts:
            nComp = compressions.get(v["fact_id"], 0)
            byCompression[nComp]["total"] += 1
            if v.get("recalled"):
                byCompression[nComp]["recalled"] += 1

        result["by_compression_depth"] = {
            str(k): {
                "total": v["total"],
                "recalled": v["recalled"],
                "recall": v["recalled"] / v["total"] if v["total"] > 0 else 0,
            }
            for k, v in sorted(byCompression.items())
        }

        result["compaction_cycles"] = tracking.get("total_cycles", 0)

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Iterative compaction benchmark v6 — strategy comparison")
    parser.add_argument("--density", type=int, default=80)
    parser.add_argument("--target-tokens", type=int, default=5_000_000,
                        help="Conversation size in tokens (must match build_conversation_v6.py)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-window", type=int, default=CONTEXT_WINDOW)
    parser.add_argument("--high-watermark", type=float, default=0.90)
    parser.add_argument("--low-watermark", type=float, default=0.60)
    parser.add_argument("--feed-batch-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Q&A batch size (questions per request)")
    parser.add_argument("--strategies", type=str, default="S1,S2,S3,S4",
                        help="Comma-separated strategies to test")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model for judge (default: same as --model)")
    parser.add_argument("--compact-delay", type=float, default=0.5,
                        help="Seconds between compaction API calls")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-feed", action="store_true",
                        help="Skip feed+compact, load saved contexts")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated token thresholds for mid-feed evaluation "
                             "(e.g. 500000,1000000,2000000). Enables Phase D mode.")
    parser.add_argument("--grep-only", action="store_true")
    parser.add_argument("--conversation", type=str, default=None,
                        help="Path to conversation JSON (overrides --density/--target-tokens lookup)")
    parser.add_argument("--backend", type=str, default="wrapper",
                        choices=["anthropic_batch", "wrapper", "openai"],
                        help="Backend for Q&A (default: wrapper)")
    parser.add_argument("--judge-backend", type=str, default=None,
                        help="Backend for judge (default: same as --backend)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for wrapper/openai backend")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for wrapper backend (default: 4)")
    args = parser.parse_args()
    if args.judge_backend is None:
        args.judge_backend = args.backend

    strategyKeys = [s.strip() for s in args.strategies.split(",")]
    for sk in strategyKeys:
        if sk not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {sk}. Available: {list(STRATEGIES.keys())}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outputDir = Path(f"iterative_v6_R4_{timestamp}")
    outputDir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"  ITERATIVE COMPACTION BENCHMARK v6")
    print(f"  Strategies: {strategyKeys}")
    print(f"  Density: d{args.density}, bs={args.batch_size}")
    print(f"  Context window: {args.context_window:,}")
    print(f"  Watermarks: {args.high_watermark:.0%} / {args.low_watermark:.0%}")
    print(f"  Output: {outputDir}")
    print("=" * 70)

    # Load conversation
    if args.conversation:
        convFile = Path(args.conversation)
        metaFile = convFile.parent / (convFile.stem + "_meta.json")
        print(f"  Loading conversation: {convFile.name}...")
        with open(convFile, encoding="utf-8") as f:
            allMessages = json.load(f)
        with open(metaFile, encoding="utf-8") as f:
            convMeta = json.load(f)
        print(f"    {len(allMessages)} messages, ~{convMeta['est_tokens']:,} tokens, "
              f"{len(convMeta['facts'])} facts")
    else:
        allMessages, convMeta = load_conversation(args.density, args.target_tokens, args.seed)
    facts = extract_facts(convMeta)
    factMeta = convMeta["facts"]

    # Sanitize empty messages
    allMessages = [m for m in allMessages if m.get("content", "").strip()]

    totalTokens = convMeta["est_tokens"]
    freePerCycle = int(args.context_window * (args.high_watermark - args.low_watermark))
    estCycles = max(1, (totalTokens - args.context_window) // freePerCycle)

    if args.dry_run:
        nStrats = len(strategyKeys)
        # S1 (brutal) has compaction calls too
        compactCalls = estCycles * nStrats
        qaBatches = len(facts) // args.batch_size + (1 if len(facts) % args.batch_size else 0)
        qaRequests = qaBatches * nStrats
        judgePerStrat = len(facts) // 15 + (1 if len(facts) % 15 else 0)
        judgeRequests = judgePerStrat * nStrats

        # Cost estimate (Haiku batch pricing)
        compactInputTok = freePerCycle  # avg input per compact call
        compactCostPerCall = (compactInputTok * 0.80 / 1_000_000) + (compactInputTok * 0.2 * 4.0 / 1_000_000)
        compactTotal = compactCalls * compactCostPerCall
        qaCostPerReq = (args.context_window * 0.40 / 1_000_000) + (500 * 2.0 / 1_000_000)
        qaTotal = qaRequests * qaCostPerReq
        judgeCostPerReq = (2000 * 0.40 / 1_000_000) + (500 * 2.0 / 1_000_000)
        judgeTotal = judgeRequests * judgeCostPerReq

        print(f"\n  --- Dry-run estimates ---")
        print(f"  Conversation: {len(allMessages)} msgs, ~{totalTokens:,} tokens")
        print(f"  Facts: {len(facts)}")
        print(f"  Est. compaction cycles per strategy: ~{estCycles}")
        print(f"  Est. time per strategy (compact): ~{estCycles * args.compact_delay:.0f}s")
        print(f"")
        print(f"  Compaction calls (sync): {compactCalls:>6}  ~${compactTotal:.2f}")
        print(f"  Q&A requests (batch):    {qaRequests:>6}  ~${qaTotal:.2f}")
        print(f"  Judge requests (batch):  {judgeRequests:>6}  ~${judgeTotal:.2f}")
        print(f"  Total:                   {compactCalls + qaRequests + judgeRequests:>6} calls  "
              f"~${compactTotal + qaTotal + judgeTotal:.2f}")
        return

    # ================================================================
    # Setup backends
    # ================================================================

    from llm_backend import LLM_CreateBackend

    judgeModel = args.judge_model or args.model

    # LLM for compaction (sync, single calls)
    if args.backend == "wrapper":
        llm = WrapperLLM(args.model, minDelay=args.compact_delay)
    else:
        llm = RateLimitedLLM(args.model, minDelay=args.compact_delay)

    # Backend for Q&A phase (parallel)
    qaBackend = LLM_CreateBackend(
        args.backend, model=args.model,
        base_url=getattr(args, 'base_url', None),
        workers=args.workers,
    )

    # Backend for judge phase
    judgeBackend = LLM_CreateBackend(
        args.judge_backend, model=judgeModel,
        base_url=getattr(args, 'base_url', None),
        workers=args.workers,
    )

    # Save config
    config = {
        "experiment": "iterative_v6",
        "run_mode": "R4",
        "density": args.density,
        "seed": args.seed,
        "context_window": args.context_window,
        "high_watermark": args.high_watermark,
        "low_watermark": args.low_watermark,
        "feed_batch_size": args.feed_batch_size,
        "qa_batch_size": args.batch_size,
        "strategies": strategyKeys,
        "model": args.model,
        "judge_model": judgeModel,
        "backend": args.backend,
        "judge_backend": args.judge_backend,
        "workers": args.workers,
        "conversation_tokens": totalTokens,
        "conversation_messages": len(allMessages),
        "n_facts": len(facts),
        "timestamp": timestamp,
    }
    save_json(config, outputDir / "config.json")

    # Parse checkpoints early (needed for feed phase)
    checkpointTokens = []
    if args.checkpoints:
        checkpointTokens = [int(c.strip()) for c in args.checkpoints.split(",")]

    strategyContexts = {}  # stratKey -> final messages
    strategyTracking = {}  # stratKey -> tracking data
    strategyCheckpoints = {}  # stratKey -> [(messages, tracking, fedTokens), ...]

    if args.skip_feed:
        print(f"\n  --- Skipping feed+compact (loading saved contexts) ---")
        for sk in strategyKeys:
            sName = STRATEGIES[sk][0]
            contextFile = outputDir / "strategies" / f"{sk}_{sName}" / "final_context.json"
            trackFile = outputDir / "strategies" / f"{sk}_{sName}" / "fact_tracking.json"
            if not contextFile.exists():
                raise FileNotFoundError(f"Saved context not found: {contextFile}")
            with open(contextFile, encoding="utf-8") as f:
                strategyContexts[sk] = json.load(f)
            if trackFile.exists():
                with open(trackFile, encoding="utf-8") as f:
                    strategyTracking[sk] = json.load(f)
            else:
                strategyTracking[sk] = None
            print(f"  [{sk}] Loaded: {len(strategyContexts[sk])} msgs")
    else:
        print(f"\n  === PHASE 1: Feed + Compact ===")
        for sk in strategyKeys:
            sName, sClass = STRATEGIES[sk]
            print(f"\n  --- {sk} ({sName}) ---")

            # Create compactor
            if sk == "S1":
                compactor = sClass(
                    maxContextTokens=args.context_window,
                    highWatermark=args.high_watermark,
                    minKeepRecent=2,
                )
            else:
                compactor = sClass(
                    maxContextTokens=args.context_window,
                    highWatermark=args.high_watermark,
                    lowWatermark=args.low_watermark,
                )

            finalMsgs, tracking, cpSnapshots = feed_strategy(
                allMessages, compactor, llm, sk, factMeta,
                batchSize=args.feed_batch_size, system=SYSTEM_PROMPT,
                checkpoints=checkpointTokens,
            )

            strategyContexts[sk] = finalMsgs
            strategyTracking[sk] = tracking
            strategyCheckpoints[sk] = cpSnapshots

            # Save
            stratDir = outputDir / "strategies" / f"{sk}_{sName}"
            save_json(finalMsgs, stratDir / "final_context.json")
            save_json(tracking, stratDir / "fact_tracking.json")

            # Save compaction log separately
            if tracking:
                save_json(tracking.get("cycle_log", []), stratDir / "compaction_log.json")

    # ================================================================
    # EVALUATION FUNCTION (reusable for checkpoints)
    # ================================================================

    def evaluate_snapshot(snapshotContexts, snapshotTracking, evalFacts, evalFactMeta,
                          evalLabel, evalDir):
        """Run grep + QA + judge + metrics on a set of strategy contexts.

        Args:
            snapshotContexts: dict {stratKey: messages}
            snapshotTracking: dict {stratKey: tracking_data}
            evalFacts: list of (fid, question, answer, keywords, qType)
            evalFactMeta: list of fact metadata dicts
            evalLabel: label for print output
            evalDir: directory to save results
        Returns: dict {stratKey: metrics}
        """
        evalKeys = [sk for sk in strategyKeys if sk in snapshotContexts]
        bs = args.batch_size

        # -- Grep --
        grepResults = {}
        for sk in evalKeys:
            gResult = grep_keywords(snapshotContexts[sk], evalFacts)
            gSummary = summarize_grep(gResult)
            grepResults[sk] = gSummary
            save_json(gResult, evalDir / "grep" / f"{sk}.json")
            print(f"  [{sk}] grep: {gSummary['recall_upper_bound']:.1%}")

        # -- QA --
        qaRequests = []
        for sk in evalKeys:
            contextMsgs = snapshotContexts[sk]
            for bIdx in range(0, len(evalFacts), bs):
                batch = evalFacts[bIdx:bIdx + bs]
                questionsText = "\n".join(
                    f"- [{fid}] {question}" for fid, question, _, _, _ in batch
                )
                prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
                reqMessages = contextMsgs + [{"role": "user", "content": prompt}]
                qaRequests.append({
                    "custom_id": f"qa_{sk}_bs{bs}_b{bIdx // bs}",
                    "params": {
                        "model": args.model,
                        "max_tokens": 4096,
                        "system": SYSTEM_PROMPT,
                        "messages": reqMessages,
                    },
                })

        print(f"  QA: {len(qaRequests)} requests")
        allQaResults = qaBackend.run_requests(qaRequests)

        answersByStrategy = {sk: {} for sk in evalKeys}
        for cid, result in allQaResults.items():
            parts = cid.split("_")
            sk = parts[1]
            batchIdx = int(parts[3][1:])
            if result["status"] != "succeeded":
                continue
            try:
                answers = parse_llm_json(result["text"])
                answerMap = {a["id"]: a.get("answer", "[parse error]") for a in answers}
            except Exception:
                answerMap = {}
            batchStart = batchIdx * bs
            batchFacts = evalFacts[batchStart:batchStart + bs]
            for fid, _, _, _, _ in batchFacts:
                answersByStrategy[sk][fid] = answerMap.get(fid, "[no answer]")

        for sk in evalKeys:
            save_json(answersByStrategy[sk], evalDir / "answers" / f"{sk}_bs{bs}.json")

        # -- Judge --
        judgeBatchSize = 15
        judgeRequests = []
        for sk in evalKeys:
            entries = []
            for fid, question, answer, keywords, qType in evalFacts:
                llmAnswer = answersByStrategy[sk].get(fid, "[no answer]")
                entries.append({
                    "id": fid, "question": question,
                    "expected_answer": answer, "expected_keywords": keywords,
                    "llm_answer": llmAnswer,
                })
            for jIdx in range(0, len(entries), judgeBatchSize):
                batch = entries[jIdx:jIdx + judgeBatchSize]
                judgeRequests.append({
                    "custom_id": f"judge_{sk}_b{jIdx // judgeBatchSize}",
                    "params": {
                        "model": judgeModel,
                        "max_tokens": 4096,
                        "system": JUDGE_SYSTEM,
                        "messages": [{"role": "user", "content": json.dumps(batch, indent=2)}],
                    },
                })
                # Fix: use proper prompt format
                judgeRequests[-1]["params"]["messages"] = [
                    {"role": "user", "content": BATCH_JUDGE_PROMPT.format(
                        entries=json.dumps(batch, indent=2))}
                ]

        print(f"  Judge: {len(judgeRequests)} requests")
        allJudgeResults = judgeBackend.run_requests(judgeRequests)

        verdictsByStrategy = {sk: [] for sk in evalKeys}
        for cid, result in allJudgeResults.items():
            parts = cid.split("_")
            sk = parts[1]
            if result["status"] != "succeeded":
                continue
            try:
                verdicts = parse_llm_json(result["text"])
                for v in verdicts:
                    v["fact_id"] = v.pop("id", v.get("fact_id", "?"))
                verdictsByStrategy[sk].extend(verdicts)
            except Exception:
                pass

        for sk in evalKeys:
            save_json({"verdicts": verdictsByStrategy[sk]},
                      evalDir / "judgments" / f"{sk}_bs{bs}.json")

        # -- Metrics --
        evalResults = {}
        for sk in evalKeys:
            sName = STRATEGIES[sk][0]
            metrics = compute_metrics(
                verdictsByStrategy[sk], evalFactMeta,
                tracking=snapshotTracking.get(sk),
            )
            metrics["grep"] = grepResults.get(sk, {})
            evalResults[sk] = metrics
            print(f"  [{sk}] Recall: {metrics['recall']:.1%} "
                  f"({metrics['facts_recalled']}/{metrics['facts_total']}) "
                  f"| Grep: {metrics['grep'].get('recall_upper_bound', 0):.1%}")

        save_json({"results": evalResults, "label": evalLabel}, evalDir / "summary.json")
        return evalResults

    # ================================================================
    # PHASE 2-5 — Evaluate (final + checkpoints)
    # ================================================================

    if args.grep_only:
        # Quick grep scan only
        for sk in strategyKeys:
            gResult = grep_keywords(strategyContexts[sk], facts)
            gSummary = summarize_grep(gResult)
            save_json(gResult, outputDir / "grep" / f"{sk}.json")
            print(f"  [{sk}] grep: {gSummary['recall_upper_bound']:.1%}")
        return

    # Evaluate checkpoints (if any)
    checkpointResults = {}
    if checkpointTokens:
        for cpIdx, (sk, snapshots) in enumerate(
            [(sk, strategyCheckpoints.get(sk, [])) for sk in strategyKeys]
        ):
            pass  # handled below

        # Collect all checkpoint snapshots across strategies
        # strategyCheckpoints[sk] = [(messages, tracking, fedTokens), ...]
        nCheckpoints = max(len(strategyCheckpoints.get(sk, []))
                          for sk in strategyKeys) if strategyCheckpoints else 0

        for cpIdx in range(nCheckpoints):
            cpContexts = {}
            cpTracking = {}
            cpTok = 0
            for sk in strategyKeys:
                snaps = strategyCheckpoints.get(sk, [])
                if cpIdx < len(snaps):
                    msgs, tracking, fedTok = snaps[cpIdx]
                    cpContexts[sk] = msgs
                    cpTracking[sk] = tracking
                    cpTok = fedTok

            if not cpContexts:
                continue

            cpLabel = f"checkpoint_{cpTok // 1000}K"
            cpDir = outputDir / f"checkpoint_{cpTok // 1000}K"
            print(f"\n  === EVAL CHECKPOINT @~{cpTok // 1000}K ===")

            # Only evaluate facts that have been fed at this checkpoint
            fedFactIds = set()
            for sk in cpContexts:
                tracking = cpTracking[sk]
                for fid, info in tracking.get("fact_status", {}).items():
                    if info["status"] != "not_yet_fed":
                        fedFactIds.add(fid)

            cpFacts = [(fid, q, a, kw, qt) for fid, q, a, kw, qt in facts
                       if fid in fedFactIds]
            cpFactMeta = [fm for fm in factMeta if fm["fact_id"] in fedFactIds]

            print(f"  Facts fed: {len(cpFacts)}/{len(facts)}")
            cpResults = evaluate_snapshot(
                cpContexts, cpTracking, cpFacts, cpFactMeta, cpLabel, cpDir)
            checkpointResults[cpLabel] = cpResults

    # Evaluate final state
    print(f"\n  === EVAL FINAL ===")
    summaryResults = evaluate_snapshot(
        strategyContexts, strategyTracking, facts, factMeta, "final", outputDir)

    # Save summary
    summary = {
        "experiment": "iterative_v6",
        "run_mode": "R4",
        "density": args.density,
        "conversation_tokens": totalTokens,
        "context_window": args.context_window,
        "qa_batch_size": args.batch_size,
        "results": summaryResults,
        "checkpoints": checkpointResults,
        "config": config,
    }
    save_json(summary, outputDir / "summary.json")

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")

    # Print checkpoint results
    if checkpointResults:
        for cpLabel, cpRes in sorted(checkpointResults.items()):
            print(f"\n  {cpLabel}:")
            print(f"  {'Strategy':<20s} {'Recall':>8s} {'Grep':>8s}")
            print(f"  {'-'*20} {'-'*8} {'-'*8}")
            for sk in strategyKeys:
                if sk in cpRes:
                    m = cpRes[sk]
                    print(f"  {sk} {STRATEGIES[sk][0]:<15s} "
                          f"{m['recall']:>7.1%} "
                          f"{m['grep'].get('recall_upper_bound', 0):>7.1%}")

    # Print final results
    print(f"\n  FINAL ({totalTokens:,} tok):")
    print(f"  {'Strategy':<20s} {'Recall':>8s} {'Grep':>8s} {'Cycles':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for sk in strategyKeys:
        sName = STRATEGIES[sk][0]
        m = summaryResults[sk]
        cycles = m.get("compaction_cycles", "?")
        print(f"  {sk} {sName:<15s} {m['recall']:>7.1%} "
              f"{m['grep'].get('recall_upper_bound', 0):>7.1%} {cycles:>8}")
    print(f"\n  Output: {outputDir}/")
    print(f"  Done!")


if __name__ == "__main__":
    main()

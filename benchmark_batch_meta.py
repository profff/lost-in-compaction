#!python3
"""
Meta-test: batch size × density × compaction — methodology validation.

Before re-running the full compaction benchmark, validate the measurement
methodology itself. Three questions:
1. Does Q&A batch size (questions per LLM call) affect recall?
2. Does fact density change the behavior?
3. Does compaction level interact with both?

Bonus: compare grep (keyword scan) vs LLM judge.

Stage 1: Baseline (no compaction) — 3 densities × 4 batch sizes = 12 runs
Stage 2: With compaction — if stage 1 shows batch size effect

Usage:
    ./benchmark_batch_meta.py --dry-run
    ./benchmark_batch_meta.py --stage 1 --min-delay 2.0
    ./benchmark_batch_meta.py --stage 2 --skip-compact --densities 50 --batch-sizes 1,10
    ./benchmark_batch_meta.py --grep-only    # just keyword scan, no API calls
"""

import sys
import os
import json
import time
import math
import random
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from itertools import combinations
from dotenv import load_dotenv

# Load .env from benchmark root
load_dotenv(Path(__file__).parent / ".env")

from compaction import estimate_tokens, messages_to_text, COMPACT_SYSTEM, COMPACT_PROMPT

# Reuse from benchmark v2
from benchmark_compaction_v2 import (
    generate_facts,
    make_padding_exchange,
    RateLimitedLLM, OllamaLLM,
    BATCH_QUESTION_PROMPT, BATCH_JUDGE_PROMPT, JUDGE_SYSTEM,
)

# Force unbuffered output
_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# CONSTANTS
# ============================================================================

# Real target context size (Anthropic API token count).
# Leaves ~10K headroom for question prompt + system (Haiku max: 200K input).
REAL_TARGET_TOKENS = 190_000

# estimate_tokens (chars/4) underestimates by ~22% vs Anthropic tokenizer.
# Measured: chars/token ≈ 3.1, not 4.0. Ratio estimated/real ≈ 0.78.
# We build contexts to an approximate target, then calibrate precisely
# with count_tokens API (free, exact).
ESTIMATE_RATIO = 0.78
COMPACTION_LEVELS = {
    "C0": 0.0,    # No compaction (baseline)
    "C1": 0.10,   # Compact oldest 10%
    "C2": 0.25,   # Compact oldest 25%
    "C3": 0.50,   # Compact oldest 50%
    "C4": 0.98,   # Compact ~everything except last 2 messages
}


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def build_context(facts: list[tuple], targetTokens: int, seed: int = 42) -> list[dict]:
    """
    Build a conversation of ~targetTokens with facts uniformly distributed.

    Unlike benchmark_compaction_v2.build_conversation(), this builds a context
    that is EXACTLY at targetTokens (padded) and is meant to be used as-is
    for Q&A, not fed through a compaction loop.
    """
    r = random.Random(seed)
    messages = []

    # Opening
    messages.extend([
        {"role": "user", "content": "Hey, I need help with a complex project. Let me give you context as we go."},
        {"role": "assistant", "content": "Sure! I'm ready. Share the details and I'll help wherever I can."},
    ])

    nFacts = len(facts)
    currentTokens = estimate_tokens(messages)
    paddingKinds = ["file_read", "discussion", "tool_chain"]

    # Token budget per fact slot (padding before + fact itself)
    tokensPerSlot = max(300, (targetTokens - currentTokens) // max(1, nFacts))

    for i, (factId, factText, _, _) in enumerate(facts):
        # Add padding before the fact — cap chunk size to slot budget
        tokenTarget = currentTokens + tokensPerSlot - 50  # reserve ~50 for fact
        while estimate_tokens(messages) < tokenTarget:
            remaining = tokenTarget - estimate_tokens(messages)
            if remaining < 200:
                break
            kind = r.choice(paddingKinds)
            # Scale chunk to remaining budget (avoid overshooting)
            maxChars = min(5000, max(600, remaining * 3))
            chunkChars = r.randint(min(600, maxChars), maxChars)
            padding = make_padding_exchange(kind, r, targetChars=chunkChars)
            messages.extend(padding)

        # Add the fact
        messages.extend([
            {"role": "user", "content": factText},
            {"role": "assistant", "content": "Got it, noted. I'll keep that in mind for the implementation."},
        ])

        currentTokens = estimate_tokens(messages)

        if (i + 1) % 25 == 0:
            print(f"    Built {i+1}/{nFacts} facts, ~{currentTokens:,} tokens...")

    # If already over target (high density), trim trailing padding
    # If under target, pad up
    currentTokens = estimate_tokens(messages)
    if currentTokens > targetTokens:
        # Trim non-fact messages from the end until we're at target
        messages = _trim_to_target(messages, facts, targetTokens)
    else:
        messages = pad_to_target(messages, targetTokens, seed + 9999)

    return messages


def _trim_to_target(messages: list[dict], facts: list[tuple], targetTokens: int) -> list[dict]:
    """
    Remove trailing non-fact messages to get closer to targetTokens.

    Preserves all fact messages. Removes padding from the end.
    """
    factTexts = {ftext for _, ftext, _, _ in facts}

    # Work backwards, removing pairs of non-fact messages
    while estimate_tokens(messages) > targetTokens and len(messages) > 4:
        # Check last 2 messages — if neither contains a fact, remove them
        lastContent = messages[-2].get("content", "")
        if lastContent in factTexts:
            break  # Don't remove facts
        # Remove last 2 messages (user + assistant pair)
        messages.pop()
        messages.pop()

    return messages


def pad_to_target(messages: list[dict], targetTokens: int, seed: int) -> list[dict]:
    """
    Add neutral padding at the end until we reach targetTokens.

    Padding contains NO facts — just technical discussions and code.
    """
    r = random.Random(seed)
    paddingKinds = ["file_read", "discussion", "tool_chain"]

    current = estimate_tokens(messages)
    while current < targetTokens:
        remaining = targetTokens - current
        # Adjust chunk size to not overshoot too much
        if remaining < 500:
            break
        chunkChars = min(r.randint(1500, 4000), remaining * 3)
        kind = r.choice(paddingKinds)
        padding = make_padding_exchange(kind, r, targetChars=chunkChars)
        messages.extend(padding)
        current = estimate_tokens(messages)

    return messages


# ============================================================================
# TOKEN CALIBRATION (count_tokens API — free, exact)
# ============================================================================

def count_real_tokens(client, messages, model, system=""):
    """Get exact token count via Anthropic count_tokens API (free, no model invocation)."""
    params = {"model": model, "messages": messages}
    if system:
        params["system"] = system
    result = client.messages.count_tokens(**params)
    return result.input_tokens


def _total_chars(messages):
    """Sum of all content character lengths in messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("content", "") or block.get("text", "") or "")
    return total


def calibrate_context(messages, facts, client, model, targetRealTokens=REAL_TARGET_TOKENS,
                      seed=42, tolerance=3000):
    """
    Adjust context size to hit targetRealTokens precisely using count_tokens API.

    Strategy:
    1. Measure real token count (1 free API call)
    2. Compute actual chars/token ratio for this content
    3. Trim or pad using measured ratio
    4. Verify with 1 more API call

    Returns (messages, realTokenCount).
    """
    realTokens = count_real_tokens(client, messages, model)
    estimated = estimate_tokens(messages)
    totalChars = _total_chars(messages)
    charsPerToken = totalChars / realTokens if realTokens > 0 else 3.1

    print(f"    Calibration: {estimated:,} est, {realTokens:,} real "
          f"({charsPerToken:.2f} chars/tok, est/real={estimated/realTokens:.3f})")

    if abs(realTokens - targetRealTokens) <= tolerance:
        print(f"    Within tolerance ({tolerance:,} tok), no adjustment needed")
        return messages, realTokens

    factTexts = {ftext for _, ftext, _, _ in facts}

    if realTokens > targetRealTokens:
        # Trim non-fact messages from the end
        overshoot = realTokens - targetRealTokens
        targetCharsToRemove = overshoot * charsPerToken
        removedChars = 0
        removedMsgs = 0
        while removedChars < targetCharsToRemove and len(messages) > 4:
            if messages[-2].get("content", "") in factTexts:
                break
            removedChars += len(str(messages[-1].get("content", "")))
            removedChars += len(str(messages[-2].get("content", "")))
            messages.pop()
            messages.pop()
            removedMsgs += 2
        print(f"    Trimmed {removedMsgs} msgs (~{removedChars:,} chars)")

    else:
        # Pad with neutral content
        deficit = targetRealTokens - realTokens
        # Scale deficit from real token space to estimated token space
        estToRealRatio = estimated / realTokens if realTokens > 0 else ESTIMATE_RATIO
        currentEstimated = estimate_tokens(messages)
        paddingTarget = currentEstimated + int(deficit * estToRealRatio)
        messages = pad_to_target(messages, paddingTarget, seed)
        print(f"    Padded: +{deficit:,} real tok target (to ~{paddingTarget:,} estimated)")

    # Verify
    finalReal = count_real_tokens(client, messages, model)
    finalEstimated = estimate_tokens(messages)
    delta = finalReal - targetRealTokens
    print(f"    Final: {finalEstimated:,} est, {finalReal:,} real "
          f"(target: {targetRealTokens:,}, delta: {delta:+,})")

    # If still over by a lot, do one more pass
    if finalReal > targetRealTokens + tolerance:
        print(f"    Still over tolerance, trimming more...")
        charsPerToken2 = _total_chars(messages) / finalReal if finalReal > 0 else charsPerToken
        overshoot2 = finalReal - targetRealTokens
        targetChars2 = overshoot2 * charsPerToken2
        removedChars2 = 0
        while removedChars2 < targetChars2 and len(messages) > 4:
            if messages[-2].get("content", "") in factTexts:
                break
            removedChars2 += len(str(messages[-1].get("content", "")))
            removedChars2 += len(str(messages[-2].get("content", "")))
            messages.pop()
            messages.pop()
        finalReal = count_real_tokens(client, messages, model)
        print(f"    After 2nd pass: {finalReal:,} real tokens")

    return messages, finalReal


# ============================================================================
# CONTROLLED COMPACTION
# ============================================================================

def compact_portion(messages: list[dict], fraction: float, llm,
                    system: str = "") -> tuple[list[dict], dict]:
    """
    Compact the oldest `fraction` of messages into a single summary pair.

    This is a controlled, single-pass compaction (not multi-cycle).
    Returns (new_messages, compaction_info).
    """
    if fraction <= 0:
        return messages, {"compacted": False, "fraction": 0, "reason": "no compaction requested"}

    nToCompact = max(2, int(len(messages) * fraction))
    # Ensure even number (user/assistant pairs)
    nToCompact = nToCompact - (nToCompact % 2)

    if nToCompact >= len(messages) - 2:
        nToCompact = len(messages) - 2  # keep at least last 2

    toCompact = messages[:nToCompact]
    remaining = messages[nToCompact:]

    conversationText = messages_to_text(toCompact)

    # Truncate if too long for the model
    maxChars = 500_000  # ~125K tokens
    if len(conversationText) > maxChars:
        conversationText = conversationText[:maxChars] + "\n\n[...truncated...]"

    inputChars = len(COMPACT_PROMPT) + len(conversationText)
    print(f"    Compacting {nToCompact} msgs ({fraction:.0%}), "
          f"~{inputChars:,} chars to summarize...")

    try:
        response = llm.chat_raw(
            [{"role": "user", "content": f"{COMPACT_PROMPT}{conversationText}"}],
            tools=None, system=COMPACT_SYSTEM,
        )
        summary = None
        for block in response.content:
            if block.type == "text":
                summary = block.text
                break

        if not summary:
            return messages, {
                "compacted": False, "fraction": fraction,
                "reason": "empty summary response"
            }
    except Exception as e:
        return messages, {
            "compacted": False, "fraction": fraction,
            "reason": f"{type(e).__name__}: {e}"
        }

    summaryPair = [
        {"role": "user", "content": f"[Summary of {nToCompact} earlier messages]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from our earlier conversation."},
    ]

    newMessages = summaryPair + remaining

    tokensBefore = estimate_tokens(messages, system)
    tokensAfter = estimate_tokens(newMessages, system)

    info = {
        "compacted": True,
        "fraction": fraction,
        "messagesCompacted": nToCompact,
        "messagesRemaining": len(newMessages),
        "tokensBefore": tokensBefore,
        "tokensAfter": tokensAfter,
        "tokensFreed": tokensBefore - tokensAfter,
        "summaryLength": len(summary),
    }

    print(f"    Compacted: {nToCompact} msgs -> summary ({len(summary)} chars), "
          f"freed ~{info['tokensFreed']:,} tokens")

    return newMessages, info


# ============================================================================
# KEYWORD GREP SCAN (FREE — no API calls)
# ============================================================================

def grep_keywords(messages: list[dict], facts: list[tuple]) -> list[dict]:
    """
    For each fact, check if its keywords are present in the context text.

    Returns: [{fact_id, keywords, keywords_found, all_present, any_present}, ...]
    Cost: 0 (pure string matching).
    """
    # Build full text from all messages (lowercase for matching)
    fullText = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            fullText += content.lower() + "\n"
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    fullText += (block.get("text", "") or block.get("content", "")).lower() + "\n"

    results = []
    for factId, _, _, keywords in facts:
        found = [kw for kw in keywords if kw.lower() in fullText]
        results.append({
            "fact_id": factId,
            "keywords": keywords,
            "keywords_found": found,
            "all_present": len(found) == len(keywords) and len(keywords) > 0,
            "any_present": len(found) > 0,
        })

    return results


# ============================================================================
# Q&A WITH FULL ARCHIVAL
# ============================================================================

def ask_questions_archived(messages: list[dict], facts: list[tuple], llm,
                           system: str = "", batchSize: int = 10) -> dict:
    """
    Ask questions in batches, archiving prompts and raw responses.

    Returns a dict with the archival structure:
    {
        "context_tokens": int,
        "batch_size": int,
        "batches": [
            {
                "batch_id": int,
                "prompt_sent": str,
                "raw_llm_response": str,
                "answers": [{fact_id, question, raw_answer, batch_position}, ...]
            }, ...
        ]
    }
    """
    contextTokens = estimate_tokens(messages, system)
    result = {
        "context_tokens": contextTokens,
        "batch_size": batchSize,
        "batches": [],
    }

    for batchStart in range(0, len(facts), batchSize):
        batch = facts[batchStart:batchStart + batchSize]
        batchId = batchStart // batchSize

        questionsText = "\n".join(
            f"- [{fid}] {question}"
            for fid, _, question, _ in batch
        )
        prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
        questionMessages = messages + [{"role": "user", "content": prompt}]

        batchResult = {
            "batch_id": batchId,
            "prompt_sent": prompt,
            "raw_llm_response": "",
            "answers": [],
        }

        try:
            response = llm.chat_raw(questionMessages, tools=None, system=system)
            responseText = ""
            for block in response.content:
                if block.type == "text":
                    responseText = block.text
                    break

            batchResult["raw_llm_response"] = responseText

            # Parse JSON
            cleanText = responseText.strip()
            if cleanText.startswith("```"):
                cleanText = cleanText.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            answers = json.loads(cleanText)
            answerMap = {a["id"]: a.get("answer", "") for a in answers}

            for pos, (fid, _, question, _) in enumerate(batch):
                batchResult["answers"].append({
                    "fact_id": fid,
                    "question": question,
                    "raw_answer": answerMap.get(fid, "[no answer in batch response]"),
                    "batch_position": pos,
                })

        except (json.JSONDecodeError, Exception) as e:
            batchResult["raw_llm_response"] = f"[ERROR] {e}"
            for pos, (fid, _, question, _) in enumerate(batch):
                batchResult["answers"].append({
                    "fact_id": fid,
                    "question": question,
                    "raw_answer": f"[batch error: {e}]",
                    "batch_position": pos,
                })

        result["batches"].append(batchResult)

        answered = batchStart + len(batch)
        print(f"    Questions: {answered}/{len(facts)} answered (bs={batchSize})")
        time.sleep(0.5)

    return result


def judge_answers_archived(answersArchive: dict, facts: list[tuple], llm,
                           judgeBatchSize: int = 15) -> dict:
    """
    Judge all answers, archiving prompts and raw judge responses.

    Returns a dict with archival structure:
    {
        "judge_batch_size": int,
        "batches": [
            {
                "batch_id": int,
                "judge_prompt_sent": str,
                "raw_judge_response": str,
                "verdicts": [{fact_id, recalled, accurate, notes}, ...]
            }, ...
        ]
    }
    """
    # Flatten all answers from the Q&A archive
    allAnswers = []
    for batch in answersArchive["batches"]:
        for ans in batch["answers"]:
            allAnswers.append(ans)

    # Build keyword map
    keywordMap = {fid: kw for fid, _, _, kw in facts}

    result = {
        "judge_batch_size": judgeBatchSize,
        "batches": [],
    }

    for batchStart in range(0, len(allAnswers), judgeBatchSize):
        batch = allAnswers[batchStart:batchStart + judgeBatchSize]
        batchId = batchStart // judgeBatchSize

        entriesText = "\n\n".join(
            f"[{a['fact_id']}]\n"
            f"  Expected keywords: {', '.join(keywordMap.get(a['fact_id'], []))}\n"
            f"  Answer: {a['raw_answer']}"
            for a in batch
        )
        prompt = BATCH_JUDGE_PROMPT.format(entries=entriesText)

        batchResult = {
            "batch_id": batchId,
            "judge_prompt_sent": prompt,
            "raw_judge_response": "",
            "verdicts": [],
        }

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": prompt}],
                tools=None, system=JUDGE_SYSTEM,
            )
            responseText = ""
            for block in response.content:
                if block.type == "text":
                    responseText = block.text
                    break

            batchResult["raw_judge_response"] = responseText

            cleanText = responseText.strip()
            if cleanText.startswith("```"):
                cleanText = cleanText.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            evals = json.loads(cleanText)
            evalMap = {e["id"]: e for e in evals}

            for a in batch:
                fid = a["fact_id"]
                ev = evalMap.get(fid, {
                    "recalled": False, "accurate": False,
                    "notes": "missing from judge response"
                })
                batchResult["verdicts"].append({
                    "fact_id": fid,
                    "recalled": ev.get("recalled", False),
                    "accurate": ev.get("accurate", False),
                    "notes": ev.get("notes", ""),
                })

        except (json.JSONDecodeError, Exception) as e:
            batchResult["raw_judge_response"] = f"[ERROR] {e}"
            for a in batch:
                batchResult["verdicts"].append({
                    "fact_id": a["fact_id"],
                    "recalled": False,
                    "accurate": False,
                    "notes": f"judge error: {e}",
                })

        result["batches"].append(batchResult)

        judged = batchStart + len(batch)
        print(f"    Judged: {judged}/{len(allAnswers)}")
        time.sleep(0.5)

    return result


# ============================================================================
# AGREEMENT ANALYSIS
# ============================================================================

def compute_agreement(resultsPerBatchSize: dict[int, list[dict]]) -> dict:
    """
    Compute agreement between batch sizes on a per-fact basis.

    resultsPerBatchSize: {batch_size: [verdicts]}

    Returns:
    {
        "jaccard": {(bs1, bs2): float, ...},
        "stable_facts": [fact_ids recalled by ALL batch sizes],
        "unstable_facts": [fact_ids recalled by SOME but not all],
        "never_recalled": [fact_ids recalled by NONE],
    }
    """
    # Build recalled sets per batch size
    recalledSets = {}
    for bs, verdicts in resultsPerBatchSize.items():
        recalledSets[bs] = set(
            v["fact_id"] for v in verdicts if v.get("recalled")
        )

    # Pairwise Jaccard
    batchSizes = sorted(recalledSets.keys())
    jaccard = {}
    for bs1, bs2 in combinations(batchSizes, 2):
        s1, s2 = recalledSets[bs1], recalledSets[bs2]
        union = s1 | s2
        inter = s1 & s2
        jaccard[f"{bs1}_vs_{bs2}"] = len(inter) / len(union) if union else 1.0

    # Stability analysis
    allFactIds = set()
    for verdicts in resultsPerBatchSize.values():
        allFactIds.update(v["fact_id"] for v in verdicts)

    stableFacts = []
    unstableFacts = []
    neverRecalled = []
    for fid in sorted(allFactIds):
        recalledBy = [bs for bs, s in recalledSets.items() if fid in s]
        if len(recalledBy) == len(batchSizes):
            stableFacts.append(fid)
        elif len(recalledBy) > 0:
            unstableFacts.append(fid)
        else:
            neverRecalled.append(fid)

    return {
        "jaccard": jaccard,
        "stable_facts": stableFacts,
        "unstable_facts": unstableFacts,
        "never_recalled": neverRecalled,
        "stable_count": len(stableFacts),
        "unstable_count": len(unstableFacts),
        "never_count": len(neverRecalled),
    }


# ============================================================================
# METRICS
# ============================================================================

def compute_run_metrics(verdicts: list[dict], facts: list[tuple]) -> dict:
    """Compute recall metrics from judge verdicts."""
    total = len(verdicts)
    recalled = sum(1 for v in verdicts if v.get("recalled"))
    accurate = sum(1 for v in verdicts if v.get("recalled") and v.get("accurate"))

    # Zone breakdown (thirds)
    zoneSize = total // 3
    zones = {
        "early": verdicts[:zoneSize],
        "mid": verdicts[zoneSize:2 * zoneSize],
        "late": verdicts[2 * zoneSize:],
    }
    zoneRecall = {}
    for zone, evals in zones.items():
        if evals:
            zoneRecall[zone] = sum(1 for e in evals if e.get("recalled")) / len(evals)
        else:
            zoneRecall[zone] = 0.0

    return {
        "recall": recalled / total if total > 0 else 0,
        "accuracy": accurate / total if total > 0 else 0,
        "facts_recalled": recalled,
        "facts_total": total,
        "recall_early": zoneRecall["early"],
        "recall_mid": zoneRecall["mid"],
        "recall_late": zoneRecall["late"],
    }


# ============================================================================
# FILE I/O
# ============================================================================

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, filepath: str):
    """Save data to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: str):
    """Load data from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# ANTHROPIC BATCH API (50% discount, no rate limits)
# ============================================================================

def submit_batch(client, requests, description=""):
    """Submit a message batch and return the batch object."""
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch submitted: {batch.id} ({len(requests)} requests) {description}")
    return batch


def wait_for_batch(client, batchId, pollInterval=30):
    """Poll until batch completes. Returns results dict keyed by custom_id."""
    startTime = time.time()
    while True:
        status = client.messages.batches.retrieve(batchId)
        counts = status.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        elapsed = time.time() - startTime
        print(f"\r  Batch {batchId[:20]}...: {done}/{total} done "
              f"(ok={counts.succeeded} err={counts.errored}) [{elapsed:.0f}s]    ", end="")
        if status.processing_status == "ended":
            print()
            break
        time.sleep(pollInterval)

    # Retrieve results
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
            results[cid] = {
                "status": result.result.type,
                "text": f"[{result.result.type}]"
            }

    succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
    print(f"  Batch complete: {succeeded}/{len(results)} succeeded")
    return results


def _parse_llm_json(text: str) -> list[dict]:
    """Parse JSON from LLM response (handles ```json blocks)."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(clean)


def run_stage1_batch(args, outputDir: str):
    """
    Stage 1 using Anthropic Batch API — 50% cost, no rate limits.

    Three-phase approach:
    1. Build all contexts + grep scan (local, free)
    2. Submit all Q&A requests as one batch, wait
    3. Submit all judge requests as one batch, wait
    """
    import anthropic

    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]
    model = args.model or "claude-haiku-4-5-20251001"
    system = "You are a helpful assistant working on a complex software project. Answer questions precisely from memory."
    judgeBatchSize = 15  # questions per judge call

    client = anthropic.Anthropic()

    # Create output dirs
    for sub in ["facts", "contexts", "grep", "answers", "judgments"]:
        ensure_dir(os.path.join(outputDir, sub))

    # Save config
    config = {
        "experiment": "batch_size_meta",
        "stage": 1,
        "timestamp": datetime.now().isoformat(),
        "densities": densities,
        "batch_sizes": batchSizes,
        "real_target_tokens": REAL_TARGET_TOKENS,
        "estimate_ratio": ESTIMATE_RATIO,
        "compaction": "C0 (none)",
        "seed": args.seed,
        "model": model,
        "backend": "anthropic_batch",
    }
    save_json(config, os.path.join(outputDir, "config.json"))

    # ===== PHASE 1: Build contexts + grep (local, free) =====
    print(f"\n  Phase 1: Building contexts and grep scan...")

    allContexts = {}   # dKey -> messages
    allFacts = {}      # density -> [(fid, text, question, keywords), ...]
    allGrep = {}       # dKey -> [{fact_id, all_present, ...}, ...]
    grepSummary = {}   # dKey -> {total, present, ...}

    for density in densities:
        dKey = f"d{density}"
        print(f"\n  --- {dKey}: {density} facts (~1/{REAL_TARGET_TOKENS // density:,} tok) ---")

        # Generate facts
        facts = generate_facts(density, seed=args.seed)
        allFacts[density] = facts
        factsData = [
            {"id": fid, "text": text, "question": q, "keywords": kw}
            for fid, text, q, kw in facts
        ]
        save_json(factsData, os.path.join(outputDir, "facts", f"density_{density}.json"))

        # Build or load context
        contextFile = os.path.join(outputDir, "contexts", f"{dKey}_C0.json")
        if args.skip_compact and os.path.exists(contextFile):
            print(f"    Loading pre-built context...")
            messages = load_json(contextFile)
            print(f"    Context: {len(messages)} msgs, ~{estimate_tokens(messages):,} est. tokens")
        else:
            buildTarget = int(REAL_TARGET_TOKENS * ESTIMATE_RATIO)
            print(f"    Building context (~{buildTarget:,} est -> ~{REAL_TARGET_TOKENS:,} real)...")
            messages = build_context(facts, buildTarget, seed=args.seed)

            # Calibrate with count_tokens API (exact, free)
            messages, realTokens = calibrate_context(
                messages, facts, client, model, REAL_TARGET_TOKENS, seed=args.seed
            )
            save_json(messages, contextFile)
            print(f"    Context: {len(messages)} msgs, {realTokens:,} real tokens")

        allContexts[dKey] = messages

        # Grep scan
        grepResults = grep_keywords(messages, facts)
        save_json(grepResults, os.path.join(outputDir, "grep", f"{dKey}_C0.json"))
        allGrep[dKey] = grepResults

        grepPresent = sum(1 for g in grepResults if g["all_present"])
        grepAny = sum(1 for g in grepResults if g["any_present"])
        print(f"    Grep: {grepPresent}/{density} all present, {grepAny}/{density} any present")
        grepSummary[dKey] = {
            "facts_total": density,
            "all_present": grepPresent,
            "any_present": grepAny,
            "recall_upper_bound": grepPresent / density if density > 0 else 0,
        }

    if args.grep_only:
        summary = {"stage1": {dKey: {"grep": gs} for dKey, gs in grepSummary.items()}}
        save_json(summary, os.path.join(outputDir, "summary.json"))
        print(f"\n  Grep-only done. Results: {outputDir}/summary.json")
        return summary

    # ===== PHASE 2: Submit all Q&A requests as one batch =====
    print(f"\n  Phase 2: Building Q&A requests...")

    qaRequests = []
    # Index: custom_id -> metadata for dispatching results
    qaIndex = {}

    for density in densities:
        dKey = f"d{density}"
        facts = allFacts[density]
        messages = allContexts[dKey]

        for bs in batchSizes:
            for batchIdx, batchStart in enumerate(range(0, len(facts), bs)):
                batchFacts = facts[batchStart:batchStart + bs]

                questionsText = "\n".join(
                    f"- [{fid}] {question}"
                    for fid, _, question, _ in batchFacts
                )
                prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
                questionMessages = messages + [{"role": "user", "content": prompt}]

                customId = f"qa_{dKey}_bs{bs}_b{batchIdx}"
                qaRequests.append({
                    "custom_id": customId,
                    "params": {
                        "model": model,
                        "max_tokens": 4096,
                        "system": system,
                        "messages": questionMessages,
                    }
                })
                qaIndex[customId] = {
                    "density": density,
                    "dKey": dKey,
                    "bs": bs,
                    "batchIdx": batchIdx,
                    "facts": batchFacts,
                    "prompt": prompt,
                }

    print(f"  Total Q&A requests: {len(qaRequests)}")

    # Submit Q&A batch
    qaBatch = submit_batch(client, qaRequests, description="[Q&A]")
    qaResults = wait_for_batch(client, qaBatch.id)

    # ===== Parse Q&A results + save archives =====
    print(f"\n  Parsing Q&A results...")

    # Group results by (density, batch_size) to build answer archives
    answersByConfig = {}  # (density, bs) -> {archive dict}

    for customId, meta in qaIndex.items():
        density = meta["density"]
        dKey = meta["dKey"]
        bs = meta["bs"]
        batchIdx = meta["batchIdx"]
        batchFacts = meta["facts"]
        prompt = meta["prompt"]

        configKey = (density, bs)
        if configKey not in answersByConfig:
            answersByConfig[configKey] = {
                "context_ref": f"{dKey}_C0",
                "context_tokens": estimate_tokens(allContexts[dKey], system),
                "batch_size": bs,
                "batches": [],
            }

        result = qaResults.get(customId, {"status": "missing", "text": ""})
        responseText = result["text"]

        batchResult = {
            "batch_id": batchIdx,
            "prompt_sent": prompt,
            "raw_llm_response": responseText,
            "answers": [],
        }

        try:
            if result["status"] == "succeeded":
                parsed = _parse_llm_json(responseText)
                answerMap = {a["id"]: a.get("answer", "") for a in parsed}
            else:
                answerMap = {}
        except (json.JSONDecodeError, KeyError):
            answerMap = {}

        for pos, (fid, _, question, _) in enumerate(batchFacts):
            batchResult["answers"].append({
                "fact_id": fid,
                "question": question,
                "raw_answer": answerMap.get(fid, f"[{result['status']}]"),
                "batch_position": pos,
            })

        answersByConfig[configKey]["batches"].append(batchResult)

    # Sort batches by batch_id and save
    for (density, bs), archive in answersByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey = f"d{density}"
        runLabel = f"{dKey}_C0_bs{bs}_r0"
        save_json(archive, os.path.join(outputDir, "answers", f"{runLabel}.json"))

    print(f"  Saved {len(answersByConfig)} answer archives")

    # ===== PHASE 3: Submit all judge requests as one batch =====
    print(f"\n  Phase 3: Building judge requests...")

    # Build keyword map for all densities
    allKeywordMaps = {}
    for density, facts in allFacts.items():
        allKeywordMaps[density] = {fid: kw for fid, _, _, kw in facts}

    judgeRequests = []
    judgeIndex = {}

    for (density, bs), archive in answersByConfig.items():
        dKey = f"d{density}"
        keywordMap = allKeywordMaps[density]

        # Flatten all answers
        allAnswers = []
        for batch in archive["batches"]:
            allAnswers.extend(batch["answers"])

        # Build judge batches
        for jBatchIdx, jBatchStart in enumerate(range(0, len(allAnswers), judgeBatchSize)):
            jBatch = allAnswers[jBatchStart:jBatchStart + judgeBatchSize]

            entriesText = "\n\n".join(
                f"[{a['fact_id']}]\n"
                f"  Expected keywords: {', '.join(keywordMap.get(a['fact_id'], []))}\n"
                f"  Answer: {a['raw_answer']}"
                for a in jBatch
            )
            judgePrompt = BATCH_JUDGE_PROMPT.format(entries=entriesText)

            customId = f"judge_{dKey}_bs{bs}_jb{jBatchIdx}"
            judgeRequests.append({
                "custom_id": customId,
                "params": {
                    "model": model,
                    "max_tokens": 4096,
                    "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": judgePrompt}],
                }
            })
            judgeIndex[customId] = {
                "density": density,
                "dKey": dKey,
                "bs": bs,
                "jBatchIdx": jBatchIdx,
                "answers": jBatch,
                "prompt": judgePrompt,
            }

    print(f"  Total judge requests: {len(judgeRequests)}")

    # Submit judge batch
    judgeBatch = submit_batch(client, judgeRequests, description="[Judge]")
    judgeResults = wait_for_batch(client, judgeBatch.id)

    # ===== Parse judge results + save archives =====
    print(f"\n  Parsing judge results...")

    judgeByConfig = {}  # (density, bs) -> {archive dict}

    for customId, meta in judgeIndex.items():
        density = meta["density"]
        dKey = meta["dKey"]
        bs = meta["bs"]
        jBatchIdx = meta["jBatchIdx"]
        batchAnswers = meta["answers"]
        judgePrompt = meta["prompt"]

        configKey = (density, bs)
        if configKey not in judgeByConfig:
            judgeByConfig[configKey] = {
                "answers_ref": f"{dKey}_C0_bs{bs}_r0",
                "judge_batch_size": judgeBatchSize,
                "batches": [],
            }

        result = judgeResults.get(customId, {"status": "missing", "text": ""})
        responseText = result["text"]

        jBatchResult = {
            "batch_id": jBatchIdx,
            "judge_prompt_sent": judgePrompt,
            "raw_judge_response": responseText,
            "verdicts": [],
        }

        try:
            if result["status"] == "succeeded":
                parsed = _parse_llm_json(responseText)
                evalMap = {e["id"]: e for e in parsed}
            else:
                evalMap = {}
        except (json.JSONDecodeError, KeyError):
            evalMap = {}

        for a in batchAnswers:
            fid = a["fact_id"]
            ev = evalMap.get(fid, {
                "recalled": False, "accurate": False,
                "notes": f"[{result.get('status', 'error')}]"
            })
            jBatchResult["verdicts"].append({
                "fact_id": fid,
                "recalled": ev.get("recalled", False),
                "accurate": ev.get("accurate", False),
                "notes": ev.get("notes", ""),
            })

        judgeByConfig[configKey]["batches"].append(jBatchResult)

    # Sort and save
    for (density, bs), archive in judgeByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey = f"d{density}"
        runLabel = f"{dKey}_C0_bs{bs}_r0"
        save_json(archive, os.path.join(outputDir, "judgments", f"{runLabel}.json"))

    print(f"  Saved {len(judgeByConfig)} judgment archives")

    # ===== PHASE 4: Compute metrics + summary =====
    print(f"\n  Phase 4: Computing metrics...")

    summary = {"stage1": {}}

    for density in densities:
        dKey = f"d{density}"
        densitySummary = {"grep": grepSummary[dKey]}
        verdictsPerBs = {}

        for bs in batchSizes:
            configKey = (density, bs)
            if configKey not in judgeByConfig:
                continue

            judgeArchive = judgeByConfig[configKey]
            allVerdicts = []
            for jBatch in judgeArchive["batches"]:
                allVerdicts.extend(jBatch["verdicts"])

            metrics = compute_run_metrics(allVerdicts, allFacts[density])
            densitySummary[f"batch_{bs}"] = metrics
            verdictsPerBs[bs] = allVerdicts

            print(f"    {dKey} bs={bs}: recall={metrics['recall']:.1%} "
                  f"({metrics['facts_recalled']}/{metrics['facts_total']}) "
                  f"early={metrics['recall_early']:.1%} "
                  f"mid={metrics['recall_mid']:.1%} "
                  f"late={metrics['recall_late']:.1%}")

        # Agreement analysis
        if len(verdictsPerBs) > 1:
            agreement = compute_agreement(verdictsPerBs)
            densitySummary["agreement"] = agreement
            print(f"    Agreement: stable={agreement['stable_count']}, "
                  f"unstable={agreement['unstable_count']}, "
                  f"never={agreement['never_count']}")

        summary["stage1"][dKey] = densitySummary

    # Grep vs Judge comparison
    grepVsJudge = compute_grep_vs_judge(outputDir, densities)
    if grepVsJudge:
        summary["grep_vs_judge"] = grepVsJudge

    # Decision
    summary["conclusion"] = analyze_batch_effect(summary["stage1"], batchSizes)

    # Batch API stats
    summary["batch_api"] = {
        "qa_batch_id": qaBatch.id,
        "judge_batch_id": judgeBatch.id,
        "qa_requests": len(qaRequests),
        "judge_requests": len(judgeRequests),
        "total_requests": len(qaRequests) + len(judgeRequests),
    }

    save_json(summary, os.path.join(outputDir, "summary.json"))
    print_stage1_summary(summary, batchSizes)

    return summary


# ============================================================================
# MAIN EXPERIMENT (synchronous fallback — for Ollama or explicit --no-batch)
# ============================================================================

def run_stage1(args, llm, outputDir: str):
    """
    Stage 1: Baseline — no compaction, test batch sizes × densities.

    For each density:
      1. Generate facts
      2. Build context at 190K tokens (C0 = no compaction)
      3. Run grep keyword scan
      4. For each batch size: ask questions, judge, archive
    """
    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]
    system = "You are a helpful assistant working on a complex software project. Answer questions precisely from memory."

    # Create output dirs
    for sub in ["facts", "contexts", "grep", "answers", "judgments"]:
        ensure_dir(os.path.join(outputDir, sub))

    # Save config
    config = {
        "experiment": "batch_size_meta",
        "stage": 1,
        "timestamp": datetime.now().isoformat(),
        "densities": densities,
        "batch_sizes": batchSizes,
        "real_target_tokens": REAL_TARGET_TOKENS,
        "estimate_ratio": ESTIMATE_RATIO,
        "compaction": "C0 (none)",
        "seed": args.seed,
        "model": getattr(llm, "model", "unknown"),
        "backend": args.backend,
    }
    save_json(config, os.path.join(outputDir, "config.json"))

    # Create anthropic client for token calibration (if backend is anthropic)
    calibClient = None
    calibModel = None
    if args.backend == "anthropic":
        import anthropic
        calibClient = anthropic.Anthropic()
        calibModel = getattr(llm, "model", "claude-haiku-4-5-20251001")

    summary = {"stage1": {}}

    for density in densities:
        dKey = f"d{density}"
        print(f"\n{'=' * 60}")
        print(f"  DENSITY: {density} facts in ~{REAL_TARGET_TOKENS:,} real tokens")
        print(f"  (~1 fact / {REAL_TARGET_TOKENS // density:,} tokens)")
        print(f"{'=' * 60}")

        # 1. Generate facts
        print(f"\n  Generating {density} facts (seed={args.seed})...")
        facts = generate_facts(density, seed=args.seed)
        factsData = [
            {"id": fid, "text": text, "question": q, "keywords": kw}
            for fid, text, q, kw in facts
        ]
        save_json(factsData, os.path.join(outputDir, "facts", f"density_{density}.json"))
        print(f"    {len(facts)} facts saved")

        # 2. Build context (or load if --skip-compact)
        contextFile = os.path.join(outputDir, "contexts", f"{dKey}_C0.json")
        if args.skip_compact and os.path.exists(contextFile):
            print(f"\n  Loading pre-built context from {contextFile}...")
            messages = load_json(contextFile)
            print(f"    Loaded: {len(messages)} messages, ~{estimate_tokens(messages):,} est. tokens")
        else:
            buildTarget = int(REAL_TARGET_TOKENS * ESTIMATE_RATIO)
            print(f"\n  Building context (~{buildTarget:,} est -> ~{REAL_TARGET_TOKENS:,} real)...")
            messages = build_context(facts, buildTarget, seed=args.seed)

            # Calibrate with count_tokens API if available (exact, free)
            if calibClient:
                messages, realTokens = calibrate_context(
                    messages, facts, calibClient, calibModel,
                    REAL_TARGET_TOKENS, seed=args.seed
                )
                print(f"    Built: {len(messages)} messages, {realTokens:,} real tokens")
            else:
                print(f"    Built: {len(messages)} messages, ~{estimate_tokens(messages):,} est. tokens "
                      f"(no calibration for {args.backend})")

            save_json(messages, contextFile)

        # 3. Grep keyword scan
        print(f"\n  Running keyword scan (grep)...")
        grepResults = grep_keywords(messages, facts)
        save_json(grepResults, os.path.join(outputDir, "grep", f"{dKey}_C0.json"))

        grepPresent = sum(1 for g in grepResults if g["all_present"])
        grepAny = sum(1 for g in grepResults if g["any_present"])
        print(f"    Keywords: {grepPresent}/{density} all present, "
              f"{grepAny}/{density} any present")

        if args.grep_only:
            summary["stage1"][dKey] = {
                "grep": {
                    "facts_total": density,
                    "all_present": grepPresent,
                    "any_present": grepAny,
                    "recall_upper_bound": grepPresent / density if density > 0 else 0,
                }
            }
            continue

        # 4. For each batch size: ask + judge
        densitySummary = {
            "grep": {
                "facts_total": density,
                "all_present": grepPresent,
                "any_present": grepAny,
                "recall_upper_bound": grepPresent / density if density > 0 else 0,
            }
        }
        verdictsPerBs = {}

        for bs in batchSizes:
            runLabel = f"{dKey}_C0_bs{bs}_r0"
            print(f"\n  --- Batch size {bs} ---")

            # Ask questions
            print(f"  Asking questions (batch_size={bs})...")
            answersArchive = ask_questions_archived(
                messages, facts, llm, system=system, batchSize=bs
            )
            answersArchive["context_ref"] = f"{dKey}_C0"
            save_json(answersArchive, os.path.join(outputDir, "answers", f"{runLabel}.json"))

            # Judge answers
            print(f"  Judging answers...")
            judgeArchive = judge_answers_archived(answersArchive, facts, llm)
            judgeArchive["answers_ref"] = runLabel
            save_json(judgeArchive, os.path.join(outputDir, "judgments", f"{runLabel}.json"))

            # Flatten verdicts for metrics
            allVerdicts = []
            for jBatch in judgeArchive["batches"]:
                allVerdicts.extend(jBatch["verdicts"])

            metrics = compute_run_metrics(allVerdicts, facts)
            densitySummary[f"batch_{bs}"] = metrics
            verdictsPerBs[bs] = allVerdicts

            print(f"  Recall: {metrics['recall']:.1%} "
                  f"({metrics['facts_recalled']}/{metrics['facts_total']})")
            print(f"  Zones: early={metrics['recall_early']:.1%} "
                  f"mid={metrics['recall_mid']:.1%} "
                  f"late={metrics['recall_late']:.1%}")

        # Agreement analysis
        if len(verdictsPerBs) > 1:
            agreement = compute_agreement(verdictsPerBs)
            densitySummary["agreement"] = agreement
            print(f"\n  Agreement:")
            for pair, jac in agreement["jaccard"].items():
                print(f"    Jaccard {pair}: {jac:.3f}")
            print(f"    Stable: {agreement['stable_count']}, "
                  f"Unstable: {agreement['unstable_count']}, "
                  f"Never: {agreement['never_count']}")

        summary["stage1"][dKey] = densitySummary

    # Grep vs Judge comparison
    grepVsJudge = compute_grep_vs_judge(outputDir, densities)
    if grepVsJudge:
        summary["grep_vs_judge"] = grepVsJudge

    # Decision
    summary["conclusion"] = analyze_batch_effect(summary["stage1"], batchSizes)

    save_json(summary, os.path.join(outputDir, "summary.json"))
    print_stage1_summary(summary, batchSizes)

    return summary


def run_stage2(args, llm, outputDir: str):
    """
    Stage 2: With compaction levels — test batch size × compaction interaction.

    Only runs if stage 1 showed a significant batch size effect.
    """
    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]
    reps = args.reps
    system = "You are a helpful assistant working on a complex software project. Answer questions precisely from memory."

    # Create output dirs
    for sub in ["contexts", "grep", "answers", "judgments"]:
        ensure_dir(os.path.join(outputDir, sub))

    # Update config
    configFile = os.path.join(outputDir, "config.json")
    if os.path.exists(configFile):
        config = load_json(configFile)
    else:
        config = {}
    config["stage2"] = {
        "timestamp": datetime.now().isoformat(),
        "densities": densities,
        "batch_sizes": batchSizes,
        "compaction_levels": list(COMPACTION_LEVELS.keys()),
        "reps": reps,
    }
    save_json(config, configFile)

    # Load or build summary
    summaryFile = os.path.join(outputDir, "summary.json")
    if os.path.exists(summaryFile):
        summary = load_json(summaryFile)
    else:
        summary = {}
    summary["stage2"] = {}

    for density in densities:
        dKey = f"d{density}"
        print(f"\n{'=' * 60}")
        print(f"  STAGE 2 — DENSITY: {density} facts")
        print(f"{'=' * 60}")

        # Load facts (generated in stage 1)
        factsFile = os.path.join(outputDir, "facts", f"density_{density}.json")
        if os.path.exists(factsFile):
            factsData = load_json(factsFile)
            facts = [(f["id"], f["text"], f["question"], f["keywords"]) for f in factsData]
        else:
            print(f"  Generating facts (not found from stage 1)...")
            facts = generate_facts(density, seed=args.seed)
            factsData = [
                {"id": fid, "text": text, "question": q, "keywords": kw}
                for fid, text, q, kw in facts
            ]
            ensure_dir(os.path.join(outputDir, "facts"))
            save_json(factsData, factsFile)

        densitySummary = {}

        for cLevel, fraction in COMPACTION_LEVELS.items():
            if cLevel == "C0":
                continue  # Already done in stage 1

            cKey = f"{dKey}_{cLevel}"
            print(f"\n  --- Compaction level: {cLevel} ({fraction:.0%}) ---")

            # Build or load context
            contextFile = os.path.join(outputDir, "contexts", f"{cKey}.json")
            if args.skip_compact and os.path.exists(contextFile):
                print(f"    Loading pre-compacted context...")
                messages = load_json(contextFile)
            else:
                # Start from raw context (C0)
                rawContextFile = os.path.join(outputDir, "contexts", f"{dKey}_C0.json")
                if os.path.exists(rawContextFile):
                    rawMessages = load_json(rawContextFile)
                else:
                    print(f"    Building raw context first...")
                    buildTarget = int(REAL_TARGET_TOKENS * ESTIMATE_RATIO)
                    rawMessages = build_context(facts, buildTarget, seed=args.seed)
                    save_json(rawMessages, rawContextFile)

                # Compact
                messages, compactInfo = compact_portion(
                    deepcopy(rawMessages), fraction, llm, system
                )
                if compactInfo.get("compacted"):
                    # Re-pad to target (using estimated tokens — stage 2 uses sync)
                    padTarget = int(REAL_TARGET_TOKENS * ESTIMATE_RATIO)
                    print(f"    Re-padding to ~{padTarget:,} est. tokens...")
                    messages = pad_to_target(messages, padTarget, args.seed + hash(cKey))
                else:
                    print(f"    Compaction failed: {compactInfo.get('reason')}")
                    messages = rawMessages  # fallback

                save_json(messages, contextFile)

            print(f"    Context: {len(messages)} msgs, ~{estimate_tokens(messages):,} tokens")

            # Grep scan
            grepResults = grep_keywords(messages, facts)
            save_json(grepResults, os.path.join(outputDir, "grep", f"{cKey}.json"))
            grepPresent = sum(1 for g in grepResults if g["all_present"])
            print(f"    Grep: {grepPresent}/{density} keywords present")

            if args.grep_only:
                densitySummary[cLevel] = {
                    "grep": {"all_present": grepPresent, "recall_upper_bound": grepPresent / density}
                }
                continue

            # For each batch size × reps
            levelSummary = {
                "grep": {"all_present": grepPresent, "recall_upper_bound": grepPresent / density}
            }

            for bs in batchSizes:
                repMetrics = []
                for rep in range(reps):
                    runLabel = f"{cKey}_bs{bs}_r{rep}"
                    print(f"\n    bs={bs}, rep={rep}")

                    answersArchive = ask_questions_archived(
                        messages, facts, llm, system=system, batchSize=bs
                    )
                    answersArchive["context_ref"] = cKey
                    save_json(answersArchive, os.path.join(outputDir, "answers", f"{runLabel}.json"))

                    judgeArchive = judge_answers_archived(answersArchive, facts, llm)
                    judgeArchive["answers_ref"] = runLabel
                    save_json(judgeArchive, os.path.join(outputDir, "judgments", f"{runLabel}.json"))

                    allVerdicts = []
                    for jBatch in judgeArchive["batches"]:
                        allVerdicts.extend(jBatch["verdicts"])

                    metrics = compute_run_metrics(allVerdicts, facts)
                    repMetrics.append(metrics)

                    print(f"    Recall: {metrics['recall']:.1%}")

                # Average across reps
                avgRecall = sum(m["recall"] for m in repMetrics) / len(repMetrics)
                levelSummary[f"batch_{bs}"] = {
                    "avg_recall": avgRecall,
                    "reps": [m["recall"] for m in repMetrics],
                }

            densitySummary[cLevel] = levelSummary

        summary["stage2"][dKey] = densitySummary

    save_json(summary, os.path.join(outputDir, "summary.json"))
    return summary


# ============================================================================
# ANALYSIS
# ============================================================================

def compute_grep_vs_judge(outputDir: str, densities: list[int]) -> dict:
    """Compare grep keyword presence vs LLM judge verdicts."""
    totalFacts = 0
    grepPresent = 0
    llmRecalled = 0
    grepAbsentButRecalled = 0
    grepPresentNotRecalled = 0

    for density in densities:
        dKey = f"d{density}"

        # Load grep results
        grepFile = os.path.join(outputDir, "grep", f"{dKey}_C0.json")
        if not os.path.exists(grepFile):
            continue
        grepData = load_json(grepFile)
        grepMap = {g["fact_id"]: g["all_present"] for g in grepData}

        # Load judgments (use batch_size=1 if available, else smallest)
        judgmentFiles = sorted(Path(outputDir, "judgments").glob(f"{dKey}_C0_bs*_r0.json"))
        if not judgmentFiles:
            continue
        judgeData = load_json(str(judgmentFiles[0]))  # use first (smallest bs)
        judgeMap = {}
        for batch in judgeData["batches"]:
            for v in batch["verdicts"]:
                judgeMap[v["fact_id"]] = v.get("recalled", False)

        for fid in grepMap:
            totalFacts += 1
            gPresent = grepMap.get(fid, False)
            jRecalled = judgeMap.get(fid, False)

            if gPresent:
                grepPresent += 1
            if jRecalled:
                llmRecalled += 1
            if not gPresent and jRecalled:
                grepAbsentButRecalled += 1
            if gPresent and not jRecalled:
                grepPresentNotRecalled += 1

    if totalFacts == 0:
        return {}

    return {
        "total_facts": totalFacts,
        "grep_present": grepPresent,
        "llm_recalled": llmRecalled,
        "grep_absent_but_recalled": grepAbsentButRecalled,
        "grep_present_not_recalled": grepPresentNotRecalled,
        "grep_recall_upper_bound": grepPresent / totalFacts,
        "llm_recall": llmRecalled / totalFacts,
        "lost_in_middle_gap": (grepPresent - llmRecalled) / totalFacts if grepPresent > 0 else 0,
    }


def analyze_batch_effect(stage1: dict, batchSizes: list[int]) -> dict:
    """
    Analyze whether batch size has a significant effect on recall.

    Decision criteria:
    - If max recall difference > 5pp across batch sizes → significant effect
    - If < 3pp → no effect → keep batch_size=10
    """
    maxDiff = 0
    diffs = {}

    for dKey, data in stage1.items():
        recalls = {}
        for bs in batchSizes:
            bsKey = f"batch_{bs}"
            if bsKey in data:
                recalls[bs] = data[bsKey]["recall"]

        if len(recalls) >= 2:
            diff = max(recalls.values()) - min(recalls.values())
            diffs[dKey] = {
                "recalls": recalls,
                "max_diff_pp": diff * 100,
            }
            maxDiff = max(maxDiff, diff)

    if maxDiff > 0.05:
        conclusion = "significant_effect"
        recommendation = "proceed_to_stage2"
    elif maxDiff > 0.03:
        conclusion = "marginal_effect"
        recommendation = "consider_stage2"
    else:
        conclusion = "no_significant_effect"
        recommendation = "keep_batch_size_10"

    return {
        "max_diff_pp": maxDiff * 100,
        "conclusion": conclusion,
        "recommendation": recommendation,
        "per_density": diffs,
    }


def print_stage1_summary(summary: dict, batchSizes: list[int]):
    """Pretty-print stage 1 results."""
    print(f"\n{'=' * 70}")
    print(f"  META-TEST STAGE 1 RESULTS")
    print(f"{'=' * 70}")

    stage1 = summary.get("stage1", {})
    colWidth = 12

    # Header
    header = f"  {'Density':<15} {'Grep':>{colWidth}}"
    for bs in batchSizes:
        header += f" {'bs=' + str(bs):>{colWidth}}"
    print(header)
    print(f"  {'-' * (15 + colWidth * (len(batchSizes) + 1))}")

    for dKey in sorted(stage1.keys()):
        data = stage1[dKey]
        grepRecall = data.get("grep", {}).get("recall_upper_bound", 0)
        row = f"  {dKey:<15} {grepRecall:>{colWidth}.1%}"

        for bs in batchSizes:
            bsKey = f"batch_{bs}"
            if bsKey in data:
                recall = data[bsKey]["recall"]
                row += f" {recall:>{colWidth}.1%}"
            else:
                row += f" {'—':>{colWidth}}"
        print(row)

    # Conclusion
    conclusion = summary.get("conclusion", {})
    if conclusion:
        print(f"\n  Max diff: {conclusion.get('max_diff_pp', 0):.1f}pp")
        print(f"  Conclusion: {conclusion.get('conclusion', '?')}")
        print(f"  Recommendation: {conclusion.get('recommendation', '?')}")

    # Grep vs Judge
    gvj = summary.get("grep_vs_judge", {})
    if gvj:
        print(f"\n  Grep vs Judge:")
        print(f"    Grep present: {gvj.get('grep_present', 0)}/{gvj.get('total_facts', 0)}")
        print(f"    LLM recalled: {gvj.get('llm_recalled', 0)}/{gvj.get('total_facts', 0)}")
        print(f"    Lost in Middle gap: {gvj.get('lost_in_middle_gap', 0):.1%}")
        print(f"    Grep absent but recalled: {gvj.get('grep_absent_but_recalled', 0)} "
              f"(should be 0)")

    print()


# ============================================================================
# DRY RUN
# ============================================================================

def dry_run(args):
    """Show what would happen without making any API calls."""
    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"\n{'=' * 60}")
    print(f"  DRY RUN — Meta-test batch size")
    print(f"{'=' * 60}")
    buildTarget = int(REAL_TARGET_TOKENS * ESTIMATE_RATIO)
    print(f"  Target: {REAL_TARGET_TOKENS:,} real tokens per context")
    print(f"  Build target: ~{buildTarget:,} est. tokens (ratio={ESTIMATE_RATIO})")
    print(f"  Densities: {densities}")
    print(f"  Batch sizes: {batchSizes}")
    print(f"  Stage: {args.stage}")
    print(f"  Seed: {args.seed}")

    totalApiCalls = 0

    for density in densities:
        print(f"\n  --- Density: {density} facts ---")
        facts = generate_facts(density, seed=args.seed)
        print(f"    Facts: {len(facts)}")
        print(f"    Fact density: ~1 fact / {REAL_TARGET_TOKENS // density:,} real tokens")

        # Sample facts
        for fid, text, q, kw in facts[:3]:
            print(f"    {fid}: {text[:60]}...")
            print(f"           Keywords: {kw}")

        # Build context (actually build it in dry run to verify)
        print(f"\n    Building context...")
        messages = build_context(facts, buildTarget, seed=args.seed)
        estTokens = estimate_tokens(messages)
        expectedReal = int(estTokens / ESTIMATE_RATIO)
        print(f"    Context: {len(messages)} msgs, ~{estTokens:,} est. "
              f"(~{expectedReal:,} expected real, calibrated at runtime)")

        # Grep scan
        grepResults = grep_keywords(messages, facts)
        grepPresent = sum(1 for g in grepResults if g["all_present"])
        grepAny = sum(1 for g in grepResults if g["any_present"])
        print(f"    Grep: {grepPresent}/{density} all present, {grepAny}/{density} any present")

        # Fact distribution
        factPositions = []
        for i, msg in enumerate(messages):
            for fid, ftext, _, _ in facts:
                if ftext == msg.get("content"):
                    factPositions.append((i, fid))
                    break
        nMsgs = len(messages)
        earlyCount = sum(1 for p, _ in factPositions if p < nMsgs // 3)
        midCount = sum(1 for p, _ in factPositions if nMsgs // 3 <= p < 2 * nMsgs // 3)
        lateCount = sum(1 for p, _ in factPositions if p >= 2 * nMsgs // 3)
        print(f"    Zone distribution: early={earlyCount}, mid={midCount}, late={lateCount}")

        # API call estimate (stage 1)
        for bs in batchSizes:
            nBatches = math.ceil(density / bs)
            totalApiCalls += nBatches * 2  # ask + judge

    if args.stage in ("2", "both"):
        # Stage 2 estimate
        nCompactionLevels = len(COMPACTION_LEVELS) - 1  # exclude C0
        reps = args.reps
        for density in densities:
            for bs in batchSizes:
                nBatches = math.ceil(density / bs)
                totalApiCalls += nBatches * 2 * nCompactionLevels * reps
            totalApiCalls += nCompactionLevels  # compaction calls

    # Cost estimate
    # Q&A calls have 190K context, judge calls are small (~1-2K)
    qaCallCount = sum(
        math.ceil(d / bs)
        for d in densities for bs in batchSizes
    )
    judgeCallCount = sum(
        math.ceil(d / 15)  # judge batch size = 15
        for d in densities for _ in batchSizes
    )
    totalApiCalls = qaCallCount + judgeCallCount

    # Q&A cost dominates (190K input each), judge is negligible
    qaCost = qaCallCount * REAL_TARGET_TOKENS * 0.80 / 1_000_000  # Haiku $0.80/MTok
    judgeCost = judgeCallCount * 2000 * 0.80 / 1_000_000      # ~2K tokens each
    estimatedCost = qaCost + judgeCost
    batchCost = estimatedCost * 0.5

    useBatch = not args.no_batch_api if hasattr(args, 'no_batch_api') else True
    print(f"\n  --- Cost Estimate ---")
    print(f"  Q&A requests: {qaCallCount} (190K tok each)")
    print(f"  Judge requests: {judgeCallCount} (~2K tok each)")
    print(f"  Total requests: {totalApiCalls}")
    print(f"  Haiku sync ($0.80/MTok in): ~${estimatedCost:.0f}")
    print(f"  Haiku batch ($0.40/MTok in): ~${batchCost:.0f}  <-- default")
    if useBatch:
        print(f"  Mode: BATCH API (50% off, no rate limits)")

    # Output directory structure
    outputDir = args.output_dir
    print(f"\n  --- Output Structure ---")
    print(f"  {outputDir}/")
    print(f"  +-- config.json")
    print(f"  +-- facts/")
    for d in densities:
        print(f"  |   +-- density_{d}.json")
    print(f"  +-- contexts/")
    for d in densities:
        print(f"  |   +-- d{d}_C0.json")
        if args.stage in ("2", "both"):
            for cLevel in COMPACTION_LEVELS:
                if cLevel != "C0":
                    print(f"  |   +-- d{d}_{cLevel}.json")
    print(f"  +-- grep/")
    print(f"  +-- answers/")
    for d in densities:
        for bs in batchSizes:
            print(f"  |   +-- d{d}_C0_bs{bs}_r0.json")
    print(f"  +-- judgments/")
    print(f"  +-- summary.json")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Meta-test: batch size × density × compaction")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats and structure, no API calls")
    parser.add_argument("--grep-only", action="store_true",
                        help="Only run keyword grep scan, no API calls")
    parser.add_argument("--skip-compact", action="store_true",
                        help="Load pre-built contexts instead of rebuilding")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="1",
                        help="Which stage to run (default: 1)")
    parser.add_argument("--densities", type=str, default="4,8,19,50,100",
                        help="Comma-separated fact counts (default: 4,8,19,50,100)")
    parser.add_argument("--batch-sizes", type=str, default="1,5,10,15",
                        help="Comma-separated Q&A batch sizes (default: 1,5,10,15)")
    parser.add_argument("--reps", type=int, default=3,
                        help="Repetitions per config in stage 2 (default: 3)")
    defaultOutputDir = str(Path(__file__).resolve().parent / "results_batch_meta")
    parser.add_argument("--output-dir", type=str, default=defaultOutputDir,
                        help="Output directory (default: results_batch_meta/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic",
                        help="LLM backend (default: anthropic)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (default: haiku for anthropic)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--min-delay", type=float, default=2.0,
                        help="Min seconds between API calls (default: 2.0)")
    parser.add_argument("--no-batch-api", action="store_true",
                        help="Use synchronous API instead of batch API (default: batch for anthropic)")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                        help="Seconds between batch API polls (default: 30)")

    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    # Determine if we use batch API
    useBatchApi = (args.backend == "anthropic" and not args.no_batch_api and not args.grep_only)

    outputDir = args.output_dir
    ensure_dir(outputDir)

    startTime = time.time()

    if useBatchApi:
        print(f"Using Anthropic Batch API (50% discount, no rate limits)")
        print(f"  Model: {args.model or 'claude-haiku-4-5-20251001'}")
        print(f"  Poll interval: {args.poll_interval}s")

        if args.stage in ("1", "both"):
            print(f"\n{'#' * 60}")
            print(f"  STAGE 1: Baseline (Batch API)")
            print(f"{'#' * 60}")
            summary = run_stage1_batch(args, outputDir)

        if args.stage in ("2", "both"):
            print(f"\n{'#' * 60}")
            print(f"  STAGE 2: With compaction levels")
            print(f"{'#' * 60}")
            # Stage 2 needs sync LLM for compaction calls, batch for Q&A
            llm = RateLimitedLLM(
                model=args.model or "claude-haiku-4-5-20251001",
                minDelay=args.min_delay
            )
            run_stage2(args, llm, outputDir)

    else:
        # Synchronous mode (Ollama or explicit --no-batch-api)
        llm = None
        if not args.grep_only:
            if args.backend == "ollama":
                modelName = args.model or "qwen2.5:3b"
                print(f"Initializing Ollama (model: {modelName})...")
                llm = OllamaLLM(model=modelName, baseUrl=args.ollama_url)
            else:
                modelName = args.model or "claude-haiku-4-5-20251001"
                print(f"Initializing Anthropic sync (model: {modelName})...")
                llm = RateLimitedLLM(model=modelName, minDelay=args.min_delay)
            print(f"  Model: {llm.model}")

        if args.stage in ("1", "both"):
            print(f"\n{'#' * 60}")
            print(f"  STAGE 1: Baseline (no compaction)")
            print(f"{'#' * 60}")
            summary = run_stage1(args, llm, outputDir)

        if args.stage in ("2", "both"):
            print(f"\n{'#' * 60}")
            print(f"  STAGE 2: With compaction levels")
            print(f"{'#' * 60}")
            run_stage2(args, llm, outputDir)

    elapsed = time.time() - startTime
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results: {outputDir}/summary.json")


if __name__ == "__main__":
    main()

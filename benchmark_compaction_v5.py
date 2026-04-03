#!python3
"""
Compaction benchmark v5 — Measure information loss from compaction at constant size.

Design: take a v5 context (190K tokens), compact the oldest X%, re-pad back to 190K,
then run recall (grep + Q&A + judge). Compare to C0 baseline (v5 recall results).

Compaction levels:
    C0 = 0%   (baseline, from v5 recall runs — not re-run)
    C1 = 5%   (compact oldest 5%)
    C2 = 25%  (compact oldest 25%)
    C3 = 50%  (compact oldest 50%)
    C4 = 98%  (compact nearly everything)

Prerequisites:
    - Pre-built v5 contexts: python build_contexts_v5.py --run R4
    - v5 recall baseline: recall_v5_R4_*/summary.json

Usage:
    ./benchmark_compaction_v5.py --run R4 --densities 40,60,80 --dry-run
    ./benchmark_compaction_v5.py --run R4 --densities 40,60,80 --grep-only
    ./benchmark_compaction_v5.py --run R4 --densities 40,60,80
    ./benchmark_compaction_v5.py --run R4 --densities 40,60,80 --skip-compact
    ./benchmark_compaction_v5.py --run R4 --densities 40 --levels C2,C3
"""

import json
import os
import time
import math
import random
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Force unbuffered output
_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# IMPORTS FROM PROJECT MODULES
# ============================================================================

from compaction import estimate_tokens, messages_to_text, COMPACT_SYSTEM, COMPACT_PROMPT
from benchmark_compaction_v2 import RateLimitedLLM, OllamaLLM


# ============================================================================
# CONSTANTS
# ============================================================================

REAL_TARGET_TOKENS = 190_000  # overridden by --target-tokens
CHARS_PER_TOKEN = 4.3

COMPACTION_LEVELS = {
    "C0": 0.00,
    "C1": 0.05,
    "C2": 0.25,
    "C3": 0.50,
    "C4": 0.98,
}

LEVEL_SEED_OFFSET = {"C1": 5001, "C2": 5002, "C3": 5003, "C4": 5004}

PADDING_POOL_PATH = Path("data/padding_pool.jsonl")


# ============================================================================
# PROMPTS (from benchmark_recall_v5.py)
# ============================================================================

SYSTEM_PROMPT = ("You are a helpful assistant working on a complex software project. "
                 "Answer questions precisely from memory.")

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


# ============================================================================
# UTILITY FUNCTIONS
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
# CONTEXT LOADING (from benchmark_recall_v5.py)
# ============================================================================

_CONTEXTS_DIR_OVERRIDE = None

def contexts_dir(runMode):
    if _CONTEXTS_DIR_OVERRIDE:
        return Path(_CONTEXTS_DIR_OVERRIDE) / f"v5_{runMode}"
    return Path("data/contexts") / f"v5_{runMode}"


def load_context(runMode, density, seed=42):
    cdir = contexts_dir(runMode)
    contextFile = cdir / f"d{density}_seed{seed}.json"
    metaFile = cdir / f"d{density}_seed{seed}_meta.json"

    if not contextFile.exists():
        raise FileNotFoundError(
            f"Context not found: {contextFile}\n"
            f"Run: python build_contexts_v5.py --run {runMode} --densities {density}")

    with open(contextFile, encoding="utf-8") as f:
        messages = json.load(f)
    with open(metaFile, encoding="utf-8") as f:
        metadata = json.load(f)

    return messages, metadata


def extract_facts(metadata):
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
# GREP SCAN (from benchmark_recall_v5.py)
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


# ============================================================================
# PADDING POOL
# ============================================================================

def load_padding_pool():
    sessions = []
    with open(PADDING_POOL_PATH, encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line))
    return sessions


def select_repadding(paddingPool, targetTokens, currentTokens, seed):
    """Select padding sessions to fill deficit after compaction."""
    budget = targetTokens - currentTokens
    if budget <= 0:
        return []

    rng = random.Random(seed)
    pool = list(paddingPool)
    rng.shuffle(pool)

    selected = []
    usedTokens = 0
    for sess in pool:
        sessTok = estimate_tokens_chars(sess["chars"])
        if usedTokens + sessTok > budget:
            if usedTokens >= budget * 0.95:
                break
            continue
        selected.append(sess)
        usedTokens += sessTok
        if usedTokens >= budget * 0.98:
            break

    return selected


# ============================================================================
# BATCH API (from benchmark_recall_v5.py)
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
        # Log first error for diagnosis
        firstErr = next((r for r in results.values() if r["status"] != "succeeded"), None)
        print(f"  WARNING: {failed} failed — first error type: {firstErr['status'] if firstErr else '?'}")
    return results


# ============================================================================
# COMPACTION (adapted from benchmark_batch_meta.py:299-377)
# ============================================================================

def compact_portion(messages, fraction, llm, system=""):
    """
    Compact the oldest `fraction` of messages into a single summary pair.
    Single-pass, controlled compaction. Returns (new_messages, compaction_info).
    """
    if fraction <= 0:
        return messages, {"compacted": False, "fraction": 0, "reason": "no compaction requested"}

    nToCompact = max(2, int(len(messages) * fraction))
    nToCompact = nToCompact - (nToCompact % 2)

    if nToCompact >= len(messages) - 2:
        nToCompact = len(messages) - 2

    toCompact = messages[:nToCompact]
    remaining = messages[nToCompact:]

    conversationText = messages_to_text(toCompact)

    maxChars = 500_000
    truncated = False
    if len(conversationText) > maxChars:
        conversationText = conversationText[:maxChars] + "\n\n[...truncated...]"
        truncated = True

    inputChars = len(COMPACT_PROMPT) + len(conversationText)
    print(f"    Compacting {nToCompact} msgs ({fraction:.0%}), "
          f"~{inputChars:,} chars to summarize"
          f"{' [TRUNCATED]' if truncated else ''}...")

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
        "truncated": truncated,
    }

    print(f"    Compacted: {nToCompact} msgs -> summary ({len(summary):,} chars), "
          f"freed ~{info['tokensFreed']:,} tokens")

    return newMessages, info


# ============================================================================
# METRICS (extended from benchmark_recall_v5.py)
# ============================================================================

def compute_metrics(verdicts, metadata, compactedFactIds=None):
    """
    Compute recall metrics with zone + category breakdown.
    If compactedFactIds is provided, also compute recall for compacted vs remaining zones.
    """
    total = len(verdicts)
    recalled = sum(1 for v in verdicts if v.get("recalled"))
    accurate = sum(1 for v in verdicts if v.get("recalled") and v.get("accurate"))

    # Position zone breakdown
    factPositions = {fm["fact_id"]: fm["position_pct"] for fm in metadata["facts"]}
    early = [v for v in verdicts if factPositions.get(v["fact_id"], 0) < 33]
    mid = [v for v in verdicts if 33 <= factPositions.get(v["fact_id"], 0) < 67]
    late = [v for v in verdicts if factPositions.get(v["fact_id"], 0) >= 67]

    def zone_recall(zone):
        if not zone:
            return 0.0
        return sum(1 for v in zone if v.get("recalled")) / len(zone)

    # Per-category breakdown
    factTypes = {fm["fact_id"]: fm.get("question_type", "unknown") for fm in metadata["facts"]}
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
        "recall_early": zone_recall(early),
        "recall_mid": zone_recall(mid),
        "recall_late": zone_recall(late),
        "n_early": len(early),
        "n_mid": len(mid),
        "n_late": len(late),
        "by_category": categoryMetrics,
    }

    # Compaction zone breakdown
    if compactedFactIds is not None:
        compactedVerdicts = [v for v in verdicts if v["fact_id"] in compactedFactIds]
        remainingVerdicts = [v for v in verdicts if v["fact_id"] not in compactedFactIds]
        result["recall_compacted_zone"] = zone_recall(compactedVerdicts)
        result["recall_remaining_zone"] = zone_recall(remainingVerdicts)
        result["n_compacted"] = len(compactedVerdicts)
        result["n_remaining"] = len(remainingVerdicts)

    return result


# ============================================================================
# C0 BASELINE LOADING
# ============================================================================

def load_c0_baseline(runMode):
    """Load C0 baseline results from the most recent v5 recall run."""
    candidates = sorted(Path(".").glob(f"recall_v5_{runMode}_*"))
    candidates = [c for c in candidates if (c / "summary.json").exists()]

    if not candidates:
        print(f"  WARNING: No v5 baseline found for {runMode}")
        return None, None

    baselineDir = candidates[-1]
    with open(baselineDir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    print(f"  C0 baseline: {baselineDir.name}")
    return summary, baselineDir.name


# ============================================================================
# PHASE 0 — BUILD COMPACTED CONTEXTS
# ============================================================================

def build_compacted_context(messages, metadata, cLevel, fraction, llm, paddingPool, seed):
    """
    Compact + re-pad a single context. Returns (newMessages, updatedMetadata, compactInfo).
    """
    # 1. Compact
    compactedMessages, compactInfo = compact_portion(
        deepcopy(messages), fraction, llm, SYSTEM_PROMPT
    )

    if not compactInfo.get("compacted"):
        print(f"    FAILED: {compactInfo.get('reason')}")
        return messages, metadata, compactInfo

    nCompacted = compactInfo["messagesCompacted"]

    # 2. Identify which facts were in the compacted zone
    compactedFactIds = set()
    for fm in metadata["facts"]:
        if fm["message_start"] < nCompacted:
            compactedFactIds.add(fm["fact_id"])

    # 3. Re-pad to target
    currentChars = sum(len(m.get("content", "")) for m in compactedMessages)
    currentTokens = estimate_tokens_chars(currentChars)
    paddingSeed = seed + LEVEL_SEED_OFFSET[cLevel]

    paddingSessions = select_repadding(
        paddingPool, REAL_TARGET_TOKENS, currentTokens, paddingSeed
    )

    repadTokens = 0
    for sess in paddingSessions:
        compactedMessages.extend(sess["turns"])
        repadTokens += estimate_tokens_chars(sess["chars"])

    finalChars = sum(len(m.get("content", "")) for m in compactedMessages)
    finalTokens = estimate_tokens_chars(finalChars)

    print(f"    Re-padded: +{len(paddingSessions)} sessions (+{repadTokens:,} tok), "
          f"final ~{finalTokens:,} tok")

    # 4. Update metadata
    updatedMeta = deepcopy(metadata)
    updatedMeta["compaction"] = {
        "level": cLevel,
        "fraction": fraction,
        "messages_compacted": nCompacted,
        "summary_length": compactInfo.get("summaryLength", 0),
        "tokens_freed": compactInfo.get("tokensFreed", 0),
        "repad_sessions": len(paddingSessions),
        "repad_tokens": repadTokens,
        "truncated": compactInfo.get("truncated", False),
    }
    updatedMeta["n_messages"] = len(compactedMessages)
    updatedMeta["total_chars"] = finalChars
    updatedMeta["est_tokens"] = finalTokens

    compactedFactIdsList = []
    for fm in updatedMeta["facts"]:
        if fm["fact_id"] in compactedFactIds:
            fm["compacted"] = True
            fm["original_position_pct"] = fm["position_pct"]
            fm["message_start"] = 0
            fm["message_end"] = 2
            fm["position_pct"] = 0.0
            compactedFactIdsList.append(fm["fact_id"])
        else:
            fm["compacted"] = False
            shift = nCompacted - 2
            fm["message_start"] = max(0, fm["message_start"] - shift)
            fm["message_end"] = max(2, fm["message_end"] - shift)
            fm["position_pct"] = fm["message_start"] / len(compactedMessages) * 100

    updatedMeta["compaction"]["facts_compacted"] = compactedFactIdsList
    updatedMeta["compaction"]["n_facts_compacted"] = len(compactedFactIdsList)
    updatedMeta["compaction"]["n_facts_remaining"] = len(updatedMeta["facts"]) - len(compactedFactIdsList)

    return compactedMessages, updatedMeta, compactInfo


def build_all_contexts(args, llm, paddingPool, outputDir):
    """Phase 0: Build or load all compacted contexts."""
    runMode = args.run
    densities = [int(d) for d in args.densities.split(",")]
    levels = [l.strip() for l in args.levels.split(",")]
    seed = args.seed

    allContexts = {}
    allMetadata = {}
    allFacts = {}
    allCompactedFacts = {}

    for density in densities:
        dKey = f"d{density}"

        try:
            rawMessages, rawMetadata = load_context(runMode, density, seed)
        except FileNotFoundError as e:
            print(f"\n  {dKey}: SKIP — {e}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {dKey}: {rawMetadata['n_messages']} msgs, "
              f"~{rawMetadata['est_tokens']:,} tok, "
              f"{rawMetadata['n_evidence']} facts")
        print(f"{'=' * 60}")

        for cLevel in levels:
            if cLevel == "C0":
                continue

            fraction = COMPACTION_LEVELS[cLevel]
            configKey = (dKey, cLevel)
            contextFile = outputDir / "contexts" / f"{dKey}_{cLevel}.json"
            metaFile = outputDir / "contexts" / f"{dKey}_{cLevel}_meta.json"

            print(f"\n  --- {dKey} {cLevel} ({fraction:.0%}) ---")

            if args.skip_compact and contextFile.exists():
                print(f"    Loading pre-built context...")
                with open(contextFile, encoding="utf-8") as f:
                    messages = json.load(f)
                with open(metaFile, encoding="utf-8") as f:
                    metadata = json.load(f)
                print(f"    Loaded: {len(messages)} msgs, ~{metadata['est_tokens']:,} tok")
            else:
                messages, metadata, compactInfo = build_compacted_context(
                    rawMessages, rawMetadata, cLevel, fraction, llm, paddingPool, seed
                )

                if not compactInfo.get("compacted"):
                    print(f"    SKIPPING {dKey}_{cLevel}: compaction failed")
                    continue

                save_json(messages, str(contextFile))
                save_json(metadata, str(metaFile))

            facts = extract_facts(metadata)

            compactedFactIds = set(
                fm["fact_id"] for fm in metadata["facts"] if fm.get("compacted")
            )

            # Sanitize: remove empty messages (some padding sessions have empty user turns)
            messages = [m for m in messages if m.get("content", "").strip()]
            allContexts[configKey] = messages
            allMetadata[configKey] = metadata
            allFacts[configKey] = facts
            allCompactedFacts[configKey] = compactedFactIds

    return allContexts, allMetadata, allFacts, allCompactedFacts


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(args):
    from llm_backend import LLM_CreateBackend

    runMode = args.run
    densities = [int(d) for d in args.densities.split(",")]
    levels = [l.strip() for l in args.levels.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]
    model = args.model
    judgeModel = args.judge_model or model
    judgeBatchSize = 15

    backend = LLM_CreateBackend(
        args.backend, model=model, judge_model=judgeModel,
        base_url=getattr(args, 'base_url', None),
        api_key=getattr(args, 'api_key', None),
        poll_interval=args.poll_interval,
        workers=getattr(args, 'workers', 4),
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outputDir = Path(args.output_dir or f"compaction_v5_{runMode}_{timestamp}")

    for sub in ["contexts", "grep", "answers", "judgments"]:
        (outputDir / sub).mkdir(parents=True, exist_ok=True)

    # ===== PHASE 0: Build compacted contexts =====
    print(f"\n{'#' * 60}")
    print(f"  PHASE 0: Building compacted contexts")
    print(f"{'#' * 60}")

    # LLM for sync compaction calls
    compactModel = args.compact_model or model
    if args.backend == "openai":
        llm = OllamaLLM(model=compactModel, baseUrl=args.base_url.replace("/v1", ""))
    else:
        llm = RateLimitedLLM(model=compactModel, minDelay=2.0)

    print(f"\n  Loading padding pool...")
    paddingPool = load_padding_pool()
    print(f"  Loaded {len(paddingPool)} padding sessions")

    allContexts, allMetadata, allFacts, allCompactedFacts = build_all_contexts(
        args, llm, paddingPool, outputDir
    )

    if not allContexts:
        print("\n  No contexts built. Check v5 context availability.")
        return

    # ===== PHASE 1: Grep scan =====
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: Grep keyword scan")
    print(f"{'#' * 60}")

    grepSummary = {}

    for configKey in sorted(allContexts.keys()):
        dKey, cLevel = configKey
        messages = allContexts[configKey]
        facts = allFacts[configKey]
        compactedFactIds = allCompactedFacts[configKey]

        grepResults = grep_keywords(messages, facts)
        save_json(grepResults, str(outputDir / "grep" / f"{dKey}_{cLevel}.json"))

        grepPresent = sum(1 for g in grepResults if g["all_present"])
        grepAny = sum(1 for g in grepResults if g["any_present"])

        # Split grep by compaction zone
        grepCompacted = [g for g in grepResults if g["fact_id"] in compactedFactIds]
        grepRemaining = [g for g in grepResults if g["fact_id"] not in compactedFactIds]
        grepCompPresent = sum(1 for g in grepCompacted if g["all_present"])
        grepRemPresent = sum(1 for g in grepRemaining if g["all_present"])

        print(f"  {dKey}_{cLevel}: grep {grepPresent}/{len(facts)} "
              f"(compacted: {grepCompPresent}/{len(grepCompacted)}, "
              f"remaining: {grepRemPresent}/{len(grepRemaining)})")

        # Per-category
        grepByType = defaultdict(lambda: {"total": 0, "present": 0})
        for g in grepResults:
            cat = g["question_type"]
            grepByType[cat]["total"] += 1
            if g["all_present"]:
                grepByType[cat]["present"] += 1

        grepSummary[f"{dKey}_{cLevel}"] = {
            "facts_total": len(facts),
            "all_present": grepPresent,
            "any_present": grepAny,
            "recall_upper_bound": grepPresent / len(facts) if facts else 0,
            "compacted_present": grepCompPresent,
            "compacted_total": len(grepCompacted),
            "remaining_present": grepRemPresent,
            "remaining_total": len(grepRemaining),
            "by_category": {cat: {"total": s["total"], "present": s["present"],
                                   "recall": s["present"] / s["total"] if s["total"] else 0}
                            for cat, s in sorted(grepByType.items())},
        }

    if args.grep_only:
        summary = {"run_mode": runMode, "experiment": "compaction_v5", "grep": grepSummary}
        save_json(summary, str(outputDir / "summary.json"))
        print(f"\n  Grep-only done. Results: {outputDir}/summary.json")
        return

    # Save config
    config = {
        "experiment": "compaction_v5",
        "run_mode": runMode,
        "timestamp": datetime.now().isoformat(),
        "densities": densities,
        "levels": levels,
        "batch_sizes": batchSizes,
        "seed": args.seed,
        "model": model,
        "judge_model": judgeModel,
        "compact_model": compactModel,
        "backend": backend.name,
        "contexts_dir": str(contexts_dir(runMode)),
    }
    save_json(config, str(outputDir / "config.json"))

    # ===== PHASE 2: Q&A batch =====
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: Q&A via Batch API")
    print(f"{'#' * 60}")

    qaRequests = []
    qaIndex = {}

    for configKey in sorted(allContexts.keys()):
        dKey, cLevel = configKey
        facts = allFacts[configKey]
        messages = allContexts[configKey]

        for bs in batchSizes:
            for batchIdx, batchStart in enumerate(range(0, len(facts), bs)):
                batchFacts = facts[batchStart:batchStart + bs]

                questionsText = "\n".join(
                    f"- [{fid}] {question}"
                    for fid, question, _, _, _ in batchFacts
                )
                prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
                questionMessages = messages + [{"role": "user", "content": prompt}]

                customId = f"qa_{dKey}_{cLevel}_bs{bs}_b{batchIdx}"
                qaRequests.append({
                    "custom_id": customId,
                    "params": {
                        "model": model,
                        "max_tokens": 4096,
                        "system": SYSTEM_PROMPT,
                        "messages": questionMessages,
                    }
                })
                qaIndex[customId] = {
                    "dKey": dKey,
                    "cLevel": cLevel,
                    "bs": bs,
                    "batchIdx": batchIdx,
                    "facts": batchFacts,
                    "prompt": prompt,
                }

    print(f"  Total Q&A requests: {len(qaRequests)}")

    qaResults = backend.run_requests(qaRequests)

    # Parse Q&A
    print(f"\n  Parsing Q&A results...")

    answersByConfig = {}

    for customId, meta in qaIndex.items():
        dKey = meta["dKey"]
        cLevel = meta["cLevel"]
        bs = meta["bs"]
        batchIdx = meta["batchIdx"]
        batchFacts = meta["facts"]
        prompt = meta["prompt"]

        configKey = (dKey, cLevel, bs)
        if configKey not in answersByConfig:
            answersByConfig[configKey] = {
                "context_ref": f"{dKey}_{cLevel}",
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
                parsed = parse_llm_json(responseText)
                answerMap = {a["id"]: a.get("answer", "") for a in parsed}
            else:
                answerMap = {}
        except (json.JSONDecodeError, KeyError):
            answerMap = {}

        for pos, (fid, question, _, _, qType) in enumerate(batchFacts):
            batchResult["answers"].append({
                "fact_id": fid,
                "question": question,
                "question_type": qType,
                "raw_answer": answerMap.get(fid, f"[{result.get('status', 'error')}]"),
                "batch_position": pos,
            })

        answersByConfig[configKey]["batches"].append(batchResult)

    for configKey, archive in answersByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey, cLevel, bs = configKey
        save_json(archive, str(outputDir / "answers" / f"{dKey}_{cLevel}_bs{bs}.json"))

    print(f"  Saved {len(answersByConfig)} answer archives")

    # ===== PHASE 3: Judge batch =====
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: Judge via Batch API")
    print(f"{'#' * 60}")

    allKeywordMaps = {}
    allAnswerMaps = {}
    for configKey, facts in allFacts.items():
        allKeywordMaps[configKey] = {fid: kw for fid, _, _, kw, _ in facts}
        allAnswerMaps[configKey] = {fid: ans for fid, _, ans, _, _ in facts}

    judgeRequests = []
    judgeIndex = {}

    for ansConfigKey, archive in answersByConfig.items():
        dKey, cLevel, bs = ansConfigKey
        factsKey = (dKey, cLevel)
        keywordMap = allKeywordMaps[factsKey]
        answerMap = allAnswerMaps[factsKey]

        allAnswers = []
        for batch in archive["batches"]:
            allAnswers.extend(batch["answers"])

        for jBatchIdx, jBatchStart in enumerate(range(0, len(allAnswers), judgeBatchSize)):
            jBatch = allAnswers[jBatchStart:jBatchStart + judgeBatchSize]

            entriesText = "\n\n".join(
                f"[{a['fact_id']}]\n"
                f"  Expected answer: {answerMap.get(a['fact_id'], 'N/A')}\n"
                f"  Expected keywords: {', '.join(keywordMap.get(a['fact_id'], []))}\n"
                f"  LLM answer: {a['raw_answer']}"
                for a in jBatch
            )
            judgePrompt = BATCH_JUDGE_PROMPT.format(entries=entriesText)

            customId = f"judge_{dKey}_{cLevel}_bs{bs}_jb{jBatchIdx}"
            judgeRequests.append({
                "custom_id": customId,
                "params": {
                    "model": judgeModel,
                    "max_tokens": 4096,
                    "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": judgePrompt}],
                }
            })
            judgeIndex[customId] = {
                "dKey": dKey,
                "cLevel": cLevel,
                "bs": bs,
                "jBatchIdx": jBatchIdx,
                "answers": jBatch,
                "judgePrompt": judgePrompt,
            }

    print(f"  Total judge requests: {len(judgeRequests)}")

    for jr in judgeRequests:
        jr["params"]["model"] = judgeModel
    judgeResults = backend.run_requests(judgeRequests)

    # Parse judge
    print(f"\n  Parsing judge results...")

    judgeByConfig = {}

    for customId, meta in judgeIndex.items():
        dKey = meta["dKey"]
        cLevel = meta["cLevel"]
        bs = meta["bs"]
        jBatchIdx = meta["jBatchIdx"]
        batchAnswers = meta["answers"]
        judgePrompt = meta["judgePrompt"]

        configKey = (dKey, cLevel, bs)
        if configKey not in judgeByConfig:
            judgeByConfig[configKey] = {
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
                parsed = parse_llm_json(responseText)
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
                "question_type": a.get("question_type", "unknown"),
                "recalled": ev.get("recalled", False),
                "accurate": ev.get("accurate", False),
                "notes": ev.get("notes", ""),
            })

        judgeByConfig[configKey]["batches"].append(jBatchResult)

    for configKey, archive in judgeByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey, cLevel, bs = configKey
        save_json(archive, str(outputDir / "judgments" / f"{dKey}_{cLevel}_bs{bs}.json"))

    print(f"  Saved {len(judgeByConfig)} judgment archives")

    # ===== PHASE 4: Metrics =====
    print(f"\n{'#' * 60}")
    print(f"  PHASE 4: Computing metrics")
    print(f"{'#' * 60}")

    # Load C0 baseline
    c0Summary, c0Dir = load_c0_baseline(runMode)

    summary = {
        "experiment": "compaction_v5",
        "run_mode": runMode,
        "c0_reference": c0Dir,
        "results": {},
        "grep": grepSummary,
    }

    for configKey in sorted(allContexts.keys()):
        dKey, cLevel = configKey
        metadata = allMetadata[configKey]
        compactedFactIds = allCompactedFacts[configKey]

        resultKey = f"{dKey}_{cLevel}"

        # Get C0 baseline recall for this density
        c0Recall = {}
        if c0Summary:
            c0Data = c0Summary.get("results", {}).get(dKey, {})
            for bs in batchSizes:
                bsKey = f"bs{bs}"
                if bsKey in c0Data:
                    c0Recall[bs] = c0Data[bsKey].get("recall", 0)

        levelResults = {"grep": grepSummary.get(resultKey, {})}

        for bs in batchSizes:
            judgeKey = (dKey, cLevel, bs)
            if judgeKey not in judgeByConfig:
                continue

            allVerdicts = []
            for jBatch in judgeByConfig[judgeKey]["batches"]:
                allVerdicts.extend(jBatch["verdicts"])

            metrics = compute_metrics(allVerdicts, metadata, compactedFactIds)

            # Add delta vs C0
            if bs in c0Recall:
                metrics["c0_recall"] = c0Recall[bs]
                metrics["delta_vs_c0"] = metrics["recall"] - c0Recall[bs]

            levelResults[f"bs{bs}"] = metrics

            # Print
            deltaStr = ""
            if bs in c0Recall:
                delta = metrics["recall"] - c0Recall[bs]
                deltaStr = f" (delta={delta:+.1%})"

            compStr = ""
            if "recall_compacted_zone" in metrics:
                compStr = (f" | compacted={metrics['recall_compacted_zone']:.0%}"
                           f"({metrics['n_compacted']})"
                           f" remaining={metrics['recall_remaining_zone']:.0%}"
                           f"({metrics['n_remaining']})")

            print(f"    {resultKey} bs={bs}: recall={metrics['recall']:.1%}"
                  f"{deltaStr}{compStr}")

        summary["results"][resultKey] = levelResults

    # Batch API info
    summary["batch_api"] = {
        "qa_batch_ids": [],
        "judge_batch_ids": [],
        "qa_requests": len(qaRequests),
        "judge_requests": len(judgeRequests),
        "compact_calls": llm.totalCalls,
    }

    save_json(summary, str(outputDir / "summary.json"))
    print_summary(summary, batchSizes, levels, c0Summary)
    print(f"\n  Results saved to: {outputDir}/")

    return summary


# ============================================================================
# DRY RUN
# ============================================================================

def dry_run(args):
    runMode = args.run
    densities = [int(d) for d in args.densities.split(",")]
    levels = [l.strip() for l in args.levels.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"\n{'=' * 70}")
    print(f"  DRY RUN — Compaction v5 ({runMode})")
    print(f"{'=' * 70}")
    print(f"  Model (Q&A):      {args.model}")
    print(f"  Model (judge):    {args.judge_model or args.model}")
    print(f"  Model (compact):  {args.compact_model or args.model}")
    print(f"  Densities:        {densities}")
    print(f"  Levels:           {levels}")
    print(f"  Batch sizes:      {batchSizes}")

    totalCompactCalls = 0
    totalQa = 0
    totalJudge = 0

    for density in densities:
        dKey = f"d{density}"
        try:
            messages, metadata = load_context(runMode, density, args.seed)
        except FileNotFoundError as e:
            print(f"\n  {dKey}: MISSING — {e}")
            continue

        facts = extract_facts(metadata)
        nMsgs = metadata["n_messages"]

        print(f"\n  --- {dKey} ---")
        print(f"    Context: {nMsgs} msgs, ~{metadata['est_tokens']:,} tok, "
              f"{metadata['n_evidence']} facts")

        # Grep on raw context
        grepResults = grep_keywords(messages, facts)
        grepPresent = sum(1 for g in grepResults if g["all_present"])
        print(f"    Grep (raw): {grepPresent}/{len(facts)} all present")

        for cLevel in levels:
            if cLevel == "C0":
                continue
            fraction = COMPACTION_LEVELS[cLevel]
            nToCompact = max(2, int(nMsgs * fraction))
            nToCompact = nToCompact - (nToCompact % 2)
            if nToCompact >= nMsgs - 2:
                nToCompact = nMsgs - 2

            factsInZone = sum(1 for fm in metadata["facts"]
                             if fm["message_start"] < nToCompact)
            compactChars = sum(
                len(messages[i].get("content", ""))
                for i in range(min(nToCompact, len(messages)))
            )
            compactTokens = estimate_tokens_chars(compactChars)

            print(f"    {cLevel} ({fraction:.0%}): compact {nToCompact} msgs "
                  f"(~{compactTokens:,} tok), {factsInZone}/{len(facts)} facts in zone")

            totalCompactCalls += 1

            for bs in batchSizes:
                nBatches = math.ceil(len(facts) / bs)
                totalQa += nBatches
                nJudge = math.ceil(len(facts) / 15)
                totalJudge += nJudge

    totalRequests = totalQa + totalJudge
    compactCost = totalCompactCalls * REAL_TARGET_TOKENS * 0.80 / 1_000_000
    qaCost = totalQa * REAL_TARGET_TOKENS * 0.40 / 1_000_000
    judgeCost = totalJudge * 2_000 * 0.40 / 1_000_000
    totalCost = compactCost + qaCost + judgeCost

    print(f"\n  --- Cost Estimate ---")
    print(f"  Compaction calls (sync):  {totalCompactCalls}  ~${compactCost:.2f}")
    print(f"  Q&A requests (batch):     {totalQa}  ~${qaCost:.2f}")
    print(f"  Judge requests (batch):   {totalJudge}  ~${judgeCost:.2f}")
    print(f"  Total:                    {totalCompactCalls + totalRequests} calls  ~${totalCost:.2f}")


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_summary(summary, batchSizes, levels, c0Summary=None):
    runMode = summary.get("run_mode", "?")

    print(f"\n{'=' * 70}")
    print(f"  COMPACTION v5 — {runMode} — RESULTS")
    print(f"{'=' * 70}")

    results = summary.get("results", {})

    # Group by density
    byDensity = defaultdict(dict)
    for key, data in results.items():
        parts = key.split("_")
        dKey = parts[0]
        cLevel = parts[1]
        byDensity[dKey][cLevel] = data

    for dKey in sorted(byDensity.keys(), key=lambda k: int(k[1:])):
        levelsData = byDensity[dKey]

        print(f"\n  --- {dKey} ---")

        # Header
        colW = 12
        header = f"  {'bs':<6}"
        header += f" {'C0(ref)':>{colW}}"
        for cLevel in sorted(levelsData.keys(), key=lambda l: COMPACTION_LEVELS.get(l, 0)):
            header += f" {cLevel:>{colW}}"
        print(header)
        print(f"  {'-' * (6 + colW * (len(levelsData) + 1))}")

        # Grep row
        grepRow = f"  {'grep':<6}"
        c0Grep = ""
        if c0Summary:
            c0Data = c0Summary.get("results", {}).get(dKey, {})
            c0Grep = c0Data.get("grep", {}).get("recall_upper_bound", 0)
            grepRow += f" {c0Grep:>{colW}.0%}"
        else:
            grepRow += f" {'—':>{colW}}"
        for cLevel in sorted(levelsData.keys(), key=lambda l: COMPACTION_LEVELS.get(l, 0)):
            gData = summary.get("grep", {}).get(f"{dKey}_{cLevel}", {})
            grepRow += f" {gData.get('recall_upper_bound', 0):>{colW}.0%}"
        print(grepRow)

        # Recall rows per batch size
        for bs in batchSizes:
            bsKey = f"bs{bs}"
            row = f"  {bsKey:<6}"

            # C0 baseline
            if c0Summary:
                c0Data = c0Summary.get("results", {}).get(dKey, {})
                if bsKey in c0Data:
                    row += f" {c0Data[bsKey]['recall']:>{colW}.1%}"
                else:
                    row += f" {'—':>{colW}}"
            else:
                row += f" {'—':>{colW}}"

            for cLevel in sorted(levelsData.keys(), key=lambda l: COMPACTION_LEVELS.get(l, 0)):
                data = levelsData[cLevel]
                if bsKey in data:
                    recall = data[bsKey]["recall"]
                    delta = data[bsKey].get("delta_vs_c0")
                    if delta is not None:
                        row += f" {recall:>5.1%}({delta:+.0%})"
                    else:
                        row += f" {recall:>{colW}.1%}"
                else:
                    row += f" {'—':>{colW}}"
            print(row)

        # Compaction zone breakdown for bs=1
        bs1Key = "bs1"
        for cLevel in sorted(levelsData.keys(), key=lambda l: COMPACTION_LEVELS.get(l, 0)):
            data = levelsData[cLevel]
            if bs1Key in data and "recall_compacted_zone" in data[bs1Key]:
                d = data[bs1Key]
                print(f"    {cLevel}: compacted zone {d['recall_compacted_zone']:.0%} "
                      f"({d['n_compacted']} facts) | "
                      f"remaining zone {d['recall_remaining_zone']:.0%} "
                      f"({d['n_remaining']} facts)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compaction benchmark v5 — measure info loss from compaction at constant 190K")
    parser.add_argument("--run", type=str, required=True,
                        choices=["R1", "R2", "R3", "R4"],
                        help="v5 run mode (source contexts)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--grep-only", action="store_true",
                        help="Run Phase 0 (compaction) + Phase 1 (grep) only, no Batch API")
    parser.add_argument("--skip-compact", action="store_true",
                        help="Load pre-built compacted contexts (skip Phase 0 LLM calls)")
    parser.add_argument("--densities", type=str, default="40,60,80",
                        help="Comma-separated densities (default: 40,60,80)")
    parser.add_argument("--levels", type=str, default="C1,C2,C3,C4",
                        help="Compaction levels (default: C1,C2,C3,C4)")
    parser.add_argument("--batch-sizes", type=str, default="1,5,10",
                        help="Q&A batch sizes (default: 1,5,10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="Model for Q&A (default: haiku)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model for judge (default: same as --model)")
    parser.add_argument("--compact-model", type=str, default=None,
                        help="Model for compaction LLM calls (default: same as --model)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--backend", type=str, default="anthropic_batch",
                        choices=["anthropic_batch", "openai", "wrapper"],
                        help="LLM backend (default: anthropic_batch)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for OpenAI/wrapper backend")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for OpenAI backend")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for wrapper backend (default: 4)")
    parser.add_argument("--contexts-dir", type=str, default=None,
                        help="Override contexts directory")
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="Target context size for re-padding (default: 190000)")
    args = parser.parse_args()

    global _CONTEXTS_DIR_OVERRIDE, REAL_TARGET_TOKENS
    if args.contexts_dir:
        _CONTEXTS_DIR_OVERRIDE = args.contexts_dir
    if args.target_tokens:
        REAL_TARGET_TOKENS = args.target_tokens

    if args.dry_run:
        dry_run(args)
        return

    startTime = time.time()
    run_benchmark(args)
    elapsed = time.time() - startTime
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()

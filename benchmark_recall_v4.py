#!python3
"""
Recall benchmark v4 — Realistic conversations (LongMemEval data).

Loads pre-assembled contexts from data/contexts/recall_190K/ and runs
Q&A + judge via Anthropic Batch API (50% discount).

Prerequisites:
    1. Run extract_padding.py   → data/padding_pool.jsonl
    2. Run extract_evidence.py  → data/evidence_longmemeval.json
    3. Run build_contexts.py    → data/contexts/recall_190K/d*_seed*.json

Usage:
    ./benchmark_recall_v4.py --dry-run
    ./benchmark_recall_v4.py --densities 4,8,19 --batch-sizes 1,5,10
    ./benchmark_recall_v4.py --grep-only
    ./benchmark_recall_v4.py --densities 4 --batch-sizes 5 --calibrate
"""

import json
import os
import time
import math
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Force unbuffered output
_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# PROMPTS (same as v3 for comparability)
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
# CONTEXT LOADING
# ============================================================================

CONTEXTS_DIR = Path("data/contexts/recall_190K")


def load_context(density, seed=42, contextsDir=CONTEXTS_DIR):
    """Load pre-assembled context and its metadata."""
    contextFile = contextsDir / f"d{density}_seed{seed}.json"
    metaFile = contextsDir / f"d{density}_seed{seed}_meta.json"

    if not contextFile.exists():
        raise FileNotFoundError(f"Context not found: {contextFile}\n"
                                f"Run build_contexts.py --density {density} first.")

    with open(contextFile, encoding="utf-8") as f:
        messages = json.load(f)
    with open(metaFile, encoding="utf-8") as f:
        metadata = json.load(f)

    return messages, metadata


def extract_facts_from_meta(metadata):
    """
    Extract facts from context metadata.

    Returns: [(fact_id, question, answer, keywords), ...]
    Note: order matches the fact positions in the context.
    """
    facts = []
    for fm in metadata["facts"]:
        facts.append((
            fm["fact_id"],
            fm["question"],
            fm["answer"],
            fm["keywords"],
        ))
    return facts


# ============================================================================
# KEYWORD GREP SCAN (free)
# ============================================================================

def grep_keywords(messages, facts):
    """Check if fact keywords are present in the context text."""
    fullText = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            fullText += content.lower() + "\n"

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
# TOKEN CALIBRATION (optional, for verification)
# ============================================================================

def calibrate_context_tokens(client, messages, model):
    """Get exact token count via Anthropic count_tokens API (free)."""
    result = client.messages.count_tokens(
        model=model, messages=messages, system=SYSTEM_PROMPT
    )
    return result.input_tokens


# ============================================================================
# BATCH API
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
            results[cid] = {"status": result.result.type, "text": f"[{result.result.type}]"}

    succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
    print(f"  Batch complete: {succeeded}/{len(results)} succeeded")
    return results


def parse_llm_json(text):
    """Parse JSON from LLM response (handles ```json blocks)."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(clean)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(verdicts, metadata):
    """Compute recall metrics with position-aware zone breakdown."""
    total = len(verdicts)
    recalled = sum(1 for v in verdicts if v.get("recalled"))
    accurate = sum(1 for v in verdicts if v.get("recalled") and v.get("accurate"))

    # Zone breakdown based on actual position in context
    factPositions = {fm["fact_id"]: fm["position_pct"] for fm in metadata["facts"]}
    early = [v for v in verdicts if factPositions.get(v["fact_id"], 0) < 33]
    mid = [v for v in verdicts if 33 <= factPositions.get(v["fact_id"], 0) < 67]
    late = [v for v in verdicts if factPositions.get(v["fact_id"], 0) >= 67]

    def zone_recall(zone):
        if not zone:
            return 0.0
        return sum(1 for v in zone if v.get("recalled")) / len(zone)

    return {
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
    }


def compute_agreement(verdictsPerBs):
    """Compute pairwise Jaccard agreement between batch sizes."""
    recalledSets = {}
    for bs, verdicts in verdictsPerBs.items():
        recalledSets[bs] = set(v["fact_id"] for v in verdicts if v.get("recalled"))

    batchSizes = sorted(recalledSets.keys())
    jaccard = {}
    for bs1, bs2 in combinations(batchSizes, 2):
        s1, s2 = recalledSets[bs1], recalledSets[bs2]
        union = s1 | s2
        inter = s1 & s2
        jaccard[f"{bs1}_vs_{bs2}"] = len(inter) / len(union) if union else 1.0

    allFactIds = set()
    for verdicts in verdictsPerBs.values():
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
# I/O
# ============================================================================

def save_json(data, filepath):
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(args):
    """Run the recall benchmark with Anthropic Batch API."""
    import anthropic

    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]
    model = args.model
    seed = args.seed
    judgeBatchSize = 15

    client = anthropic.Anthropic()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outputDir = Path(args.output_dir or f"recall_v4_{timestamp}")

    for sub in ["grep", "answers", "judgments"]:
        (outputDir / sub).mkdir(parents=True, exist_ok=True)

    # ===== PHASE 1: Load contexts + grep (free) =====
    print(f"\n  Phase 1: Loading contexts and grep scan...")

    allContexts = {}   # dKey -> messages
    allMetadata = {}   # dKey -> metadata
    allFacts = {}      # dKey -> [(fid, question, question, keywords)]
    grepSummary = {}

    for density in densities:
        dKey = f"d{density}"
        print(f"\n  --- {dKey} ---")

        messages, metadata = load_context(density, seed)
        facts = extract_facts_from_meta(metadata)

        print(f"    Context: {metadata['n_messages']} msgs, "
              f"~{metadata['est_tokens']:,} est tokens, "
              f"{metadata['n_evidence']} facts")

        # Optional: calibrate with count_tokens
        if args.calibrate:
            realTokens = calibrate_context_tokens(client, messages, model)
            print(f"    Calibrated: {realTokens:,} real tokens "
                  f"(est/real = {metadata['est_tokens']/realTokens:.3f})")
            metadata["real_tokens"] = realTokens

        allContexts[dKey] = messages
        allMetadata[dKey] = metadata
        allFacts[dKey] = facts

        # Grep scan
        grepResults = grep_keywords(messages, facts)
        save_json(grepResults, str(outputDir / "grep" / f"{dKey}.json"))

        grepPresent = sum(1 for g in grepResults if g["all_present"])
        grepAny = sum(1 for g in grepResults if g["any_present"])
        print(f"    Grep: {grepPresent}/{len(facts)} all present, "
              f"{grepAny}/{len(facts)} any present")

        grepSummary[dKey] = {
            "facts_total": len(facts),
            "all_present": grepPresent,
            "any_present": grepAny,
            "recall_upper_bound": grepPresent / len(facts) if facts else 0,
        }

    if args.grep_only:
        summary = {"grep": grepSummary}
        save_json(summary, str(outputDir / "summary.json"))
        print(f"\n  Grep-only done. Results: {outputDir}/summary.json")
        return

    # Save config
    config = {
        "experiment": "recall_v4_longmemeval",
        "timestamp": datetime.now().isoformat(),
        "densities": densities,
        "batch_sizes": batchSizes,
        "seed": seed,
        "model": model,
        "data_source": "longmemeval",
        "contexts_dir": str(CONTEXTS_DIR),
        "backend": "anthropic_batch",
    }
    save_json(config, str(outputDir / "config.json"))

    # ===== PHASE 2: Build + submit Q&A batch =====
    print(f"\n  Phase 2: Building Q&A requests...")

    qaRequests = []
    qaIndex = {}

    for dKey in sorted(allContexts.keys()):
        facts = allFacts[dKey]
        messages = allContexts[dKey]

        for bs in batchSizes:
            for batchIdx, batchStart in enumerate(range(0, len(facts), bs)):
                batchFacts = facts[batchStart:batchStart + bs]

                questionsText = "\n".join(
                    f"- [{fid}] {question}"
                    for fid, question, _, _ in batchFacts
                )
                prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
                questionMessages = messages + [{"role": "user", "content": prompt}]

                customId = f"qa_{dKey}_bs{bs}_b{batchIdx}"
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
                    "bs": bs,
                    "batchIdx": batchIdx,
                    "facts": batchFacts,
                    "prompt": prompt,
                }

    print(f"  Total Q&A requests: {len(qaRequests)}")

    qaBatch = submit_batch(client, qaRequests, description="[Q&A]")
    qaResults = wait_for_batch(client, qaBatch.id, pollInterval=args.poll_interval)

    # ===== Parse Q&A results =====
    print(f"\n  Parsing Q&A results...")

    answersByConfig = {}  # (dKey, bs) -> archive

    for customId, meta in qaIndex.items():
        dKey = meta["dKey"]
        bs = meta["bs"]
        batchIdx = meta["batchIdx"]
        batchFacts = meta["facts"]
        prompt = meta["prompt"]

        configKey = (dKey, bs)
        if configKey not in answersByConfig:
            answersByConfig[configKey] = {
                "context_ref": dKey,
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

        for pos, (fid, question, _, _) in enumerate(batchFacts):
            batchResult["answers"].append({
                "fact_id": fid,
                "question": question,
                "raw_answer": answerMap.get(fid, f"[{result.get('status', 'error')}]"),
                "batch_position": pos,
            })

        answersByConfig[configKey]["batches"].append(batchResult)

    # Sort batches and save
    for configKey, archive in answersByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey, bs = configKey
        save_json(archive, str(outputDir / "answers" / f"{dKey}_bs{bs}.json"))

    print(f"  Saved {len(answersByConfig)} answer archives")

    # ===== PHASE 3: Build + submit judge batch =====
    print(f"\n  Phase 3: Building judge requests...")

    # Build keyword + answer maps from all facts
    allKeywordMaps = {}
    allAnswerMaps = {}
    for dKey, facts in allFacts.items():
        allKeywordMaps[dKey] = {fid: kw for fid, _, _, kw in facts}
        allAnswerMaps[dKey] = {fid: ans for fid, _, ans, _ in facts}

    judgeRequests = []
    judgeIndex = {}

    for configKey, archive in answersByConfig.items():
        dKey, bs = configKey
        keywordMap = allKeywordMaps[dKey]
        answerMap = allAnswerMaps[dKey]

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
                "dKey": dKey,
                "bs": bs,
                "jBatchIdx": jBatchIdx,
                "answers": jBatch,
            }

    print(f"  Total judge requests: {len(judgeRequests)}")

    judgeBatch = submit_batch(client, judgeRequests, description="[Judge]")
    judgeResults = wait_for_batch(client, judgeBatch.id, pollInterval=args.poll_interval)

    # ===== Parse judge results =====
    print(f"\n  Parsing judge results...")

    judgeByConfig = {}

    for customId, meta in judgeIndex.items():
        dKey = meta["dKey"]
        bs = meta["bs"]
        jBatchIdx = meta["jBatchIdx"]
        batchAnswers = meta["answers"]

        configKey = (dKey, bs)
        if configKey not in judgeByConfig:
            judgeByConfig[configKey] = {
                "judge_batch_size": judgeBatchSize,
                "batches": [],
            }

        result = judgeResults.get(customId, {"status": "missing", "text": ""})
        responseText = result["text"]

        jBatchResult = {
            "batch_id": jBatchIdx,
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
                "recalled": ev.get("recalled", False),
                "accurate": ev.get("accurate", False),
                "notes": ev.get("notes", ""),
            })

        judgeByConfig[configKey]["batches"].append(jBatchResult)

    # Sort and save
    for configKey, archive in judgeByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey, bs = configKey
        save_json(archive, str(outputDir / "judgments" / f"{dKey}_bs{bs}.json"))

    print(f"  Saved {len(judgeByConfig)} judgment archives")

    # ===== PHASE 4: Metrics + summary =====
    print(f"\n  Phase 4: Computing metrics...")

    summary = {"results": {}, "grep": grepSummary}

    for dKey in sorted(allContexts.keys()):
        metadata = allMetadata[dKey]
        densityResults = {"grep": grepSummary[dKey]}
        verdictsPerBs = {}

        for bs in batchSizes:
            configKey = (dKey, bs)
            if configKey not in judgeByConfig:
                continue

            allVerdicts = []
            for jBatch in judgeByConfig[configKey]["batches"]:
                allVerdicts.extend(jBatch["verdicts"])

            metrics = compute_metrics(allVerdicts, metadata)
            densityResults[f"bs{bs}"] = metrics
            verdictsPerBs[bs] = allVerdicts

            print(f"    {dKey} bs={bs}: recall={metrics['recall']:.1%} "
                  f"({metrics['facts_recalled']}/{metrics['facts_total']}) "
                  f"early={metrics['recall_early']:.1%} "
                  f"mid={metrics['recall_mid']:.1%} "
                  f"late={metrics['recall_late']:.1%}")

        if len(verdictsPerBs) > 1:
            agreement = compute_agreement(verdictsPerBs)
            densityResults["agreement"] = agreement
            print(f"    Agreement: stable={agreement['stable_count']}, "
                  f"unstable={agreement['unstable_count']}, "
                  f"never={agreement['never_count']}")

        summary["results"][dKey] = densityResults

    # Batch API info
    summary["batch_api"] = {
        "qa_batch_id": qaBatch.id,
        "judge_batch_id": judgeBatch.id,
        "qa_requests": len(qaRequests),
        "judge_requests": len(judgeRequests),
        "total_requests": len(qaRequests) + len(judgeRequests),
    }

    save_json(summary, str(outputDir / "summary.json"))
    print_summary(summary, batchSizes)

    print(f"\n  Results saved to: {outputDir}/")
    return summary


# ============================================================================
# DRY RUN
# ============================================================================

def dry_run(args):
    """Show what would happen without making any API calls."""
    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"\n{'=' * 60}")
    print(f"  DRY RUN — Recall Benchmark v4 (LongMemEval)")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Densities: {densities}")
    print(f"  Batch sizes: {batchSizes}")
    print(f"  Seed: {args.seed}")

    totalQaRequests = 0
    totalJudgeRequests = 0

    for density in densities:
        dKey = f"d{density}"
        try:
            messages, metadata = load_context(density, args.seed)
        except FileNotFoundError as e:
            print(f"\n  {dKey}: {e}")
            continue

        facts = extract_facts_from_meta(metadata)
        print(f"\n  --- {dKey} ---")
        print(f"    Context: {metadata['n_messages']} msgs, "
              f"~{metadata['est_tokens']:,} est tokens")
        print(f"    Evidence: {metadata['n_evidence']} facts, "
              f"sources: {metadata['evidence_sources']}")
        print(f"    Padding: {metadata['n_padding_sessions']} sessions")

        # Grep scan (free)
        grepResults = grep_keywords(messages, facts)
        grepPresent = sum(1 for g in grepResults if g["all_present"])
        print(f"    Grep: {grepPresent}/{len(facts)} all keywords present")

        # Fact positions
        for fm in metadata["facts"]:
            print(f"      {fm['fact_id']} @ {fm['position_pct']:.0f}% "
                  f"({fm['n_turns']} turns, ~{fm['est_tokens']:,} tok)")

        # Q&A requests per batch size
        for bs in batchSizes:
            nBatches = math.ceil(len(facts) / bs)
            totalQaRequests += nBatches
            print(f"    bs={bs}: {nBatches} Q&A requests")

        # Judge requests (all facts, regardless of batch size, grouped by 15)
        for bs in batchSizes:
            nJudge = math.ceil(len(facts) / 15)
            totalJudgeRequests += nJudge

    totalRequests = totalQaRequests + totalJudgeRequests

    # Cost estimate (Haiku batch: $0.40/MTok input, $0.10/MTok output)
    # Q&A: ~190K input tokens each
    # Judge: ~2K input tokens each
    qaCostIn = totalQaRequests * 190_000 * 0.40 / 1_000_000
    qaCostOut = totalQaRequests * 500 * 2.00 / 1_000_000  # ~500 tok output @ $2/MTok
    judgeCostIn = totalJudgeRequests * 2_000 * 0.40 / 1_000_000
    judgeCostOut = totalJudgeRequests * 500 * 2.00 / 1_000_000
    totalCost = qaCostIn + qaCostOut + judgeCostIn + judgeCostOut

    print(f"\n  --- Cost Estimate (Batch API, 50% off) ---")
    print(f"  Q&A requests:   {totalQaRequests} (~190K tok each)")
    print(f"  Judge requests:  {totalJudgeRequests} (~2K tok each)")
    print(f"  Total requests:  {totalRequests}")
    print(f"  Estimated cost:  ~${totalCost:.2f}")


def print_summary(summary, batchSizes):
    """Pretty-print results table."""
    print(f"\n{'=' * 70}")
    print(f"  RECALL BENCHMARK v4 — RESULTS")
    print(f"{'=' * 70}")

    results = summary.get("results", {})
    colWidth = 12

    header = f"  {'Density':<10} {'Grep':>{colWidth}}"
    for bs in batchSizes:
        header += f" {'bs=' + str(bs):>{colWidth}}"
    print(header)
    print(f"  {'-' * (10 + colWidth * (len(batchSizes) + 1))}")

    for dKey in sorted(results.keys()):
        data = results[dKey]
        grepRecall = data.get("grep", {}).get("recall_upper_bound", 0)
        row = f"  {dKey:<10} {grepRecall:>{colWidth}.1%}"

        for bs in batchSizes:
            bsKey = f"bs{bs}"
            if bsKey in data:
                recall = data[bsKey]["recall"]
                row += f" {recall:>{colWidth}.1%}"
            else:
                row += f" {'—':>{colWidth}}"
        print(row)

    # Zone detail
    print(f"\n  Zone breakdown (early <33% | mid 33-67% | late >67%):")
    for dKey in sorted(results.keys()):
        data = results[dKey]
        for bs in batchSizes:
            bsKey = f"bs{bs}"
            if bsKey in data:
                m = data[bsKey]
                print(f"    {dKey} bs={bs}: early={m['recall_early']:.0%} "
                      f"({m['n_early']}f) mid={m['recall_mid']:.0%} "
                      f"({m['n_mid']}f) late={m['recall_late']:.0%} ({m['n_late']}f)")

    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recall benchmark v4 — LongMemEval realistic conversations")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats and cost estimate, no API calls")
    parser.add_argument("--grep-only", action="store_true",
                        help="Only run keyword grep, no API calls")
    parser.add_argument("--calibrate", action="store_true",
                        help="Verify context sizes via count_tokens API")
    parser.add_argument("--densities", type=str, default="4,8,19",
                        help="Comma-separated densities (default: 4,8,19)")
    parser.add_argument("--batch-sizes", type=str, default="1,5,10",
                        help="Comma-separated Q&A batch sizes (default: 1,5,10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Context seed (default: 42)")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="Model ID (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: recall_v4_YYYYMMDD_HHMM)")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                        help="Seconds between batch API polls (default: 30)")

    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    startTime = time.time()
    run_benchmark(args)
    elapsed = time.time() - startTime
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()

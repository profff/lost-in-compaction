#!python3
"""
Recall benchmark v5 — Density sweep with 3 run modes (R1/R2/R3).

Loads pre-assembled contexts from data/contexts/v5_R{1,2,3}/ and runs
Q&A + judge via Anthropic Batch API.

Full archival: saves contexts refs, Q&A prompts, raw answers, judge prompts,
judge responses, and per-category breakdown.

Prerequisites:
    Run build_contexts_v5.py --run R1|R2|R3  (or --run all)

Usage:
    ./benchmark_recall_v5.py --run R1 --dry-run
    ./benchmark_recall_v5.py --run R1
    ./benchmark_recall_v5.py --run R2
    ./benchmark_recall_v5.py --run R3
    ./benchmark_recall_v5.py --run R4
    ./benchmark_recall_v5.py --run R1 --grep-only
"""

import json
import os
import time
import math
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# PROMPTS
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

def contexts_dir(runMode):
    return Path("data/contexts") / f"v5_{runMode}"


def load_context(runMode, density, seed=42):
    """Load pre-assembled context and metadata."""
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
    """
    Extract facts from context metadata.
    Returns: [(fact_id, question, answer, keywords, question_type), ...]
    """
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
# GREP SCAN (free)
# ============================================================================

def grep_keywords(messages, facts):
    """Check keyword presence in context."""
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
# BATCH API
# ============================================================================

def submit_batch(client, requests, description=""):
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch submitted: {batch.id} ({len(requests)} requests) {description}")
    return batch


def submit_chunked(client, requests, chunkSize, description=""):
    """Submit requests in chunks to avoid MemoryError on large payloads.
    Returns list of batch objects."""
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
            results[cid] = {"status": result.result.type, "text": f"[{result.result.type}]"}

    succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
    print(f"  Batch complete: {succeeded}/{len(results)} succeeded")
    return results


def parse_llm_json(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(clean)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(verdicts, metadata):
    """Compute recall metrics with position-aware zones and per-category breakdown."""
    total = len(verdicts)
    recalled = sum(1 for v in verdicts if v.get("recalled"))
    accurate = sum(1 for v in verdicts if v.get("recalled") and v.get("accurate"))

    # Zone breakdown
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
        "by_category": categoryMetrics,
    }


def compute_agreement(verdictsPerBs):
    """Pairwise Jaccard between batch sizes."""
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

    stableFacts, unstableFacts, neverRecalled = [], [], []
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
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(args):
    import anthropic

    runMode = args.run
    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]
    model = args.model
    seed = args.seed
    judgeBatchSize = 15

    client = anthropic.Anthropic()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outputDir = Path(args.output_dir or f"recall_v5_{runMode}_{timestamp}")

    for sub in ["grep", "answers", "judgments"]:
        (outputDir / sub).mkdir(parents=True, exist_ok=True)

    # ===== PHASE 1: Load contexts + grep =====
    print(f"\n  Phase 1: Loading contexts ({runMode}) + grep scan...")

    allContexts = {}
    allMetadata = {}
    allFacts = {}
    grepSummary = {}

    for density in densities:
        dKey = f"d{density}"
        print(f"\n  --- {dKey} ---")

        try:
            messages, metadata = load_context(runMode, density, seed)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        facts = extract_facts(metadata)

        modeInfo = metadata.get("run_mode", {})
        print(f"    Context: {metadata['n_messages']} msgs, "
              f"~{metadata['est_tokens']:,} est tok, "
              f"{metadata['n_evidence']} facts")
        print(f"    Categories: {metadata.get('question_types', {})}")

        allContexts[dKey] = messages
        allMetadata[dKey] = metadata
        allFacts[dKey] = facts

        # Grep scan
        grepResults = grep_keywords(messages, facts)
        save_json(grepResults, str(outputDir / "grep" / f"{dKey}.json"))

        grepPresent = sum(1 for g in grepResults if g["all_present"])
        grepAny = sum(1 for g in grepResults if g["any_present"])
        print(f"    Grep: {grepPresent}/{len(facts)} all, {grepAny}/{len(facts)} any")

        # Grep per category
        grepByType = defaultdict(lambda: {"total": 0, "present": 0})
        for g in grepResults:
            cat = g["question_type"]
            grepByType[cat]["total"] += 1
            if g["all_present"]:
                grepByType[cat]["present"] += 1
        grepCatStr = ", ".join(
            f"{cat}={s['present']}/{s['total']}"
            for cat, s in sorted(grepByType.items())
        )
        print(f"    Grep by category: {grepCatStr}")

        grepSummary[dKey] = {
            "facts_total": len(facts),
            "all_present": grepPresent,
            "any_present": grepAny,
            "recall_upper_bound": grepPresent / len(facts) if facts else 0,
            "by_category": {cat: {"total": s["total"], "present": s["present"],
                                   "recall": s["present"] / s["total"] if s["total"] else 0}
                            for cat, s in sorted(grepByType.items())},
        }

    if not allContexts:
        print("\n  No contexts loaded. Run build_contexts_v5.py first.")
        return

    if args.grep_only:
        summary = {"run_mode": runMode, "grep": grepSummary}
        save_json(summary, str(outputDir / "summary.json"))
        print(f"\n  Grep-only done. Results: {outputDir}/summary.json")
        return

    # Save config
    config = {
        "experiment": "recall_v5_density_sweep",
        "run_mode": runMode,
        "timestamp": datetime.now().isoformat(),
        "densities": densities,
        "batch_sizes": batchSizes,
        "seed": seed,
        "model": model,
        "backend": "anthropic_batch",
        "contexts_dir": str(contexts_dir(runMode)),
    }
    save_json(config, str(outputDir / "config.json"))

    # ===== PHASE 2: Q&A batch =====
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
                    for fid, question, _, _, _ in batchFacts
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

    # Submit in chunks to avoid MemoryError (each request carries ~140K tok context)
    QA_CHUNK_SIZE = 20
    qaBatches = submit_chunked(client, qaRequests, QA_CHUNK_SIZE, description=f"[Q&A {runMode}]")
    qaResults = {}
    for qab in qaBatches:
        qaResults.update(wait_for_batch(client, qab.id, pollInterval=args.poll_interval))

    # ===== Parse Q&A =====
    print(f"\n  Parsing Q&A results...")

    answersByConfig = {}

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
        dKey, bs = configKey
        save_json(archive, str(outputDir / "answers" / f"{dKey}_bs{bs}.json"))

    print(f"  Saved {len(answersByConfig)} answer archives")

    # ===== PHASE 3: Judge batch =====
    print(f"\n  Phase 3: Building judge requests...")

    allKeywordMaps = {}
    allAnswerMaps = {}
    for dKey, facts in allFacts.items():
        allKeywordMaps[dKey] = {fid: kw for fid, _, _, kw, _ in facts}
        allAnswerMaps[dKey] = {fid: ans for fid, _, ans, _, _ in facts}

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
                "judgePrompt": judgePrompt,
            }

    print(f"  Total judge requests: {len(judgeRequests)}")

    # Judge requests are lightweight, single batch should be fine, but chunk for safety
    JUDGE_CHUNK_SIZE = 50
    judgeBatches = submit_chunked(client, judgeRequests, JUDGE_CHUNK_SIZE, description=f"[Judge {runMode}]")
    judgeResults = {}
    for jb in judgeBatches:
        judgeResults.update(wait_for_batch(client, jb.id, pollInterval=args.poll_interval))

    # ===== Parse judge =====
    print(f"\n  Parsing judge results...")

    judgeByConfig = {}

    for customId, meta in judgeIndex.items():
        dKey = meta["dKey"]
        bs = meta["bs"]
        jBatchIdx = meta["jBatchIdx"]
        batchAnswers = meta["answers"]
        judgePrompt = meta["judgePrompt"]

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
        dKey, bs = configKey
        save_json(archive, str(outputDir / "judgments" / f"{dKey}_bs{bs}.json"))

    print(f"  Saved {len(judgeByConfig)} judgment archives")

    # ===== PHASE 4: Metrics =====
    print(f"\n  Phase 4: Computing metrics...")

    summary = {"run_mode": runMode, "results": {}, "grep": grepSummary}

    for dKey in sorted(allContexts.keys()):
        metadata = allMetadata[dKey]
        densityResults = {"grep": grepSummary.get(dKey, {})}
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

            # Print summary line
            catStr = ", ".join(
                f"{cat}={m['recall']:.0%}"
                for cat, m in sorted(metrics["by_category"].items())
            )
            print(f"    {dKey} bs={bs}: recall={metrics['recall']:.1%} "
                  f"({metrics['facts_recalled']}/{metrics['facts_total']}) "
                  f"| {catStr}")

        if len(verdictsPerBs) > 1:
            agreement = compute_agreement(verdictsPerBs)
            densityResults["agreement"] = agreement

        summary["results"][dKey] = densityResults

    # Batch API info
    summary["batch_api"] = {
        "qa_batch_ids": [b.id for b in qaBatches],
        "judge_batch_ids": [b.id for b in judgeBatches],
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
    runMode = args.run
    densities = [int(d) for d in args.densities.split(",")]
    batchSizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"\n{'=' * 70}")
    print(f"  DRY RUN — Recall v5 ({runMode})")
    print(f"{'=' * 70}")
    print(f"  Model: {args.model}")
    print(f"  Densities: {densities}")
    print(f"  Batch sizes: {batchSizes}")

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
        print(f"\n  --- {dKey} ---")
        print(f"    Context: {metadata['n_messages']} msgs, ~{metadata['est_tokens']:,} tok")
        print(f"    Evidence: {metadata['n_evidence']} facts")
        print(f"    Categories: {metadata.get('question_types', {})}")
        print(f"    Padding: {metadata['n_padding_sessions']} sessions, "
              f"~{metadata.get('padding_tokens', 0):,} tok")

        # Grep
        grepResults = grep_keywords(messages, facts)
        grepPresent = sum(1 for g in grepResults if g["all_present"])
        print(f"    Grep: {grepPresent}/{len(facts)} all present")

        for bs in batchSizes:
            nBatches = math.ceil(len(facts) / bs)
            totalQa += nBatches
            nJudge = math.ceil(len(facts) / 15)
            totalJudge += nJudge

    totalRequests = totalQa + totalJudge
    qaCost = totalQa * 190_000 * 0.40 / 1_000_000
    judgeCost = totalJudge * 2_000 * 0.40 / 1_000_000
    totalCost = qaCost + judgeCost

    print(f"\n  --- Cost Estimate (Batch API) ---")
    print(f"  Q&A requests:   {totalQa}")
    print(f"  Judge requests:  {totalJudge}")
    print(f"  Total requests:  {totalRequests}")
    print(f"  Estimated cost:  ~${totalCost:.2f}")


def print_summary(summary, batchSizes):
    runMode = summary.get("run_mode", "?")
    print(f"\n{'=' * 70}")
    print(f"  RECALL v5 — {runMode} — RESULTS")
    print(f"{'=' * 70}")

    results = summary.get("results", {})
    colWidth = 12

    header = f"  {'Density':<10} {'Grep':>{colWidth}}"
    for bs in batchSizes:
        header += f" {'bs=' + str(bs):>{colWidth}}"
    print(header)
    print(f"  {'-' * (10 + colWidth * (len(batchSizes) + 1))}")

    for dKey in sorted(results.keys(), key=lambda k: int(k[1:])):
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

    # Category breakdown for last density
    if results:
        lastDKey = sorted(results.keys(), key=lambda k: int(k[1:]))[-1]
        lastData = results[lastDKey]
        bs0Key = f"bs{batchSizes[0]}"
        if bs0Key in lastData and "by_category" in lastData[bs0Key]:
            print(f"\n  Category breakdown ({lastDKey}, bs={batchSizes[0]}):")
            for cat, m in sorted(lastData[bs0Key]["by_category"].items()):
                print(f"    {cat:35s}: {m['recalled']}/{m['total']} "
                      f"({m['recall']:.0%}) recall, {m['accuracy']:.0%} accuracy")

    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    defaultDensities = {
        "R1": "4,8,12,16,19",
        "R2": "4,8,12,16,19,25,30,40,60,80",
        "R3": "4,8,12,16,19,25,30",
        "R4": "4,8,12,16,19,25,30,40,60,80,120,150",
    }

    parser = argparse.ArgumentParser(
        description="Recall benchmark v5 — density sweep with 3 run modes")
    parser.add_argument("--run", type=str, required=True,
                        choices=["R1", "R2", "R3", "R4"],
                        help="Run mode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--grep-only", action="store_true")
    parser.add_argument("--densities", type=str, default=None,
                        help="Comma-separated densities (default: per-run)")
    parser.add_argument("--batch-sizes", type=str, default="1,5,10",
                        help="Q&A batch sizes (default: 1,5,10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    args = parser.parse_args()

    if args.densities is None:
        args.densities = defaultDensities[args.run]

    if args.dry_run:
        dry_run(args)
        return

    startTime = time.time()
    run_benchmark(args)
    elapsed = time.time() - startTime
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()

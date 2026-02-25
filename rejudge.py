#!python3
"""
Re-run ONLY the judge phase for a recall test run.

Use when the Q&A phase succeeded but the judge batch failed (e.g. billing issue).
Reads existing answer archives, rebuilds judge requests, submits a new batch,
and overwrites judgments + summary.

Usage:
    python rejudge.py recall_test_stage1_20260210_run3
    python rejudge.py <output-dir> [--dry-run]
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from itertools import combinations
from dotenv import load_dotenv

# Load .env from benchmark root
load_dotenv(Path(__file__).parent / ".env")

# Reuse prompts from benchmark_compaction_v2
from benchmark_compaction_v2 import BATCH_JUDGE_PROMPT, JUDGE_SYSTEM

# Force unbuffered output
_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


def load_json(filepath):
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def parse_llm_json(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(clean)


def submit_batch(client, requests, description=""):
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch submitted: {batch.id} ({len(requests)} requests) {description}")
    return batch


def wait_for_batch(client, batchId, pollInterval=30):
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
            errMsg = ""
            if hasattr(result.result, "error") and hasattr(result.result.error, "error"):
                errMsg = getattr(result.result.error.error, "message", "")
            results[cid] = {
                "status": result.result.type,
                "text": f"[{result.result.type}]",
                "error": errMsg,
            }

    succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
    print(f"  Batch complete: {succeeded}/{len(results)} succeeded")
    return results


def compute_run_metrics(verdicts, facts):
    total = len(verdicts)
    recalled = sum(1 for v in verdicts if v.get("recalled"))
    accurate = sum(1 for v in verdicts if v.get("recalled") and v.get("accurate"))
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


def compute_agreement(resultsPerBatchSize):
    recalledSets = {}
    for bs, verdicts in resultsPerBatchSize.items():
        recalledSets[bs] = set(v["fact_id"] for v in verdicts if v.get("recalled"))

    batchSizes = sorted(recalledSets.keys())
    jaccard = {}
    for bs1, bs2 in combinations(batchSizes, 2):
        s1, s2 = recalledSets[bs1], recalledSets[bs2]
        union = s1 | s2
        inter = s1 & s2
        jaccard[f"{bs1}_vs_{bs2}"] = len(inter) / len(union) if union else 1.0

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


def compute_grep_vs_judge(outputDir, densities):
    totalFacts = 0
    grepPresent = 0
    llmRecalled = 0
    grepAbsentButRecalled = 0
    grepPresentNotRecalled = 0

    for density in densities:
        dKey = f"d{density}"
        grepFile = os.path.join(outputDir, "grep", f"{dKey}_C0.json")
        if not os.path.exists(grepFile):
            continue
        grepData = load_json(grepFile)
        grepMap = {g["fact_id"]: g["all_present"] for g in grepData}

        judgmentFiles = sorted(Path(outputDir, "judgments").glob(f"{dKey}_C0_bs*_r0.json"))
        if not judgmentFiles:
            continue
        judgeData = load_json(str(judgmentFiles[0]))
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


def analyze_batch_effect(stage1, batchSizes):
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
            diffs[dKey] = {"recalls": recalls, "max_diff_pp": diff * 100}
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-run judge phase for a recall test run")
    parser.add_argument("output_dir", help="Path to the run directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--model", default=None, help="Model override (default: from config)")
    args = parser.parse_args()

    outputDir = args.output_dir
    if not os.path.isdir(outputDir):
        print(f"Error: {outputDir} is not a directory")
        sys.exit(1)

    # Load config
    configPath = os.path.join(outputDir, "config.json")
    if not os.path.exists(configPath):
        print(f"Error: no config.json in {outputDir}")
        sys.exit(1)
    config = load_json(configPath)

    densities = config["densities"]
    batchSizes = config["batch_sizes"]
    model = args.model or config.get("model", "claude-haiku-4-5-20251001")
    judgeBatchSize = 15

    print("=" * 70)
    print(f"  REJUDGE -- {outputDir}")
    print(f"  Model: {model}")
    print(f"  Densities: {densities}")
    print(f"  Batch sizes: {batchSizes}")
    print("=" * 70)

    # Load all facts (for keywords)
    allFacts = {}
    for density in densities:
        factsPath = os.path.join(outputDir, "facts", f"density_{density}.json")
        if not os.path.exists(factsPath):
            print(f"  Warning: {factsPath} not found, skipping density {density}")
            continue
        factsData = load_json(factsPath)
        # Build keyword map: {fact_id: [keywords]}
        allFacts[density] = {f["id"]: f["keywords"] for f in factsData}

    # Load all answer archives
    answersByConfig = {}
    answersDir = os.path.join(outputDir, "answers")
    for f in sorted(os.listdir(answersDir)):
        if not f.endswith(".json"):
            continue
        archive = load_json(os.path.join(answersDir, f))
        # Parse density and bs from filename: d4_C0_bs5_r0.json
        parts = f.replace(".json", "").split("_")
        density = int(parts[0][1:])  # d4 -> 4
        bs = int(parts[2][2:])       # bs5 -> 5
        answersByConfig[(density, bs)] = archive

    print(f"\n  Loaded {len(answersByConfig)} answer archives")
    print(f"  Loaded facts for densities: {sorted(allFacts.keys())}")

    # Build judge requests
    judgeRequests = []
    judgeIndex = {}

    for (density, bs), archive in sorted(answersByConfig.items()):
        if density not in allFacts:
            continue
        dKey = f"d{density}"
        keywordMap = allFacts[density]

        # Flatten all answers
        allAnswers = []
        for batch in archive["batches"]:
            allAnswers.extend(batch["answers"])

        # Build judge batches (same logic as benchmark_batch_meta.py)
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

    if args.dry_run:
        print("\n  [DRY RUN] Would submit batch with the requests above.")
        for cid, meta in sorted(judgeIndex.items()):
            nAnswers = len(meta["answers"])
            print(f"    {cid}: {nAnswers} answers to judge")
        return

    # Submit batch
    import anthropic
    client = anthropic.Anthropic()

    judgeBatch = submit_batch(client, judgeRequests, description="[Rejudge]")
    judgeResults = wait_for_batch(client, judgeBatch.id)

    # Check for errors
    errored = sum(1 for r in judgeResults.values() if r["status"] != "succeeded")
    if errored > 0:
        print(f"\n  WARNING: {errored}/{len(judgeResults)} requests errored!")
        for cid, r in judgeResults.items():
            if r["status"] != "succeeded":
                print(f"    {cid}: {r.get('error', r['status'])}")
        if errored == len(judgeResults):
            print("\n  ALL requests errored. Aborting without overwriting.")
            sys.exit(1)

    # Parse judge results (same logic as benchmark_batch_meta.py)
    print(f"\n  Parsing judge results...")
    judgeByConfig = {}

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

    # Sort and save judgments
    for (density, bs), archive in judgeByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey = f"d{density}"
        runLabel = f"{dKey}_C0_bs{bs}_r0"
        save_json(archive, os.path.join(outputDir, "judgments", f"{runLabel}.json"))

    print(f"  Saved {len(judgeByConfig)} judgment archives")

    # Recompute metrics + summary (phase 4)
    print(f"\n  Recomputing metrics...")

    # Load grep summaries
    grepSummary = {}
    for density in densities:
        dKey = f"d{density}"
        grepFile = os.path.join(outputDir, "grep", f"{dKey}_C0.json")
        if os.path.exists(grepFile):
            grepData = load_json(grepFile)
            grepSummary[dKey] = {
                "facts_total": density,
                "all_present": sum(1 for g in grepData if g["all_present"]),
                "any_present": sum(1 for g in grepData if g["any_present"]),
                "recall_upper_bound": sum(1 for g in grepData if g["all_present"]) / density,
            }

    # Build facts list for compute_run_metrics (needs list of tuples for zone ordering)
    factsLists = {}
    for density in densities:
        factsPath = os.path.join(outputDir, "facts", f"density_{density}.json")
        if os.path.exists(factsPath):
            factsData = load_json(factsPath)
            factsLists[density] = [(f["id"], f["text"], f["question"], f["keywords"]) for f in factsData]

    summary = {"stage1": {}}

    for density in densities:
        dKey = f"d{density}"
        densitySummary = {}
        if dKey in grepSummary:
            densitySummary["grep"] = grepSummary[dKey]

        verdictsPerBs = {}
        for bs in batchSizes:
            configKey = (density, bs)
            if configKey not in judgeByConfig:
                continue

            judgeArchive = judgeByConfig[configKey]
            allVerdicts = []
            for jBatch in judgeArchive["batches"]:
                allVerdicts.extend(jBatch["verdicts"])

            metrics = compute_run_metrics(allVerdicts, factsLists.get(density, []))
            densitySummary[f"batch_{bs}"] = metrics
            verdictsPerBs[bs] = allVerdicts

            print(f"    {dKey} bs={bs}: recall={metrics['recall']:.1%} "
                  f"({metrics['facts_recalled']}/{metrics['facts_total']}) "
                  f"early={metrics['recall_early']:.1%} "
                  f"mid={metrics['recall_mid']:.1%} "
                  f"late={metrics['recall_late']:.1%}")

        if len(verdictsPerBs) > 1:
            agreement = compute_agreement(verdictsPerBs)
            densitySummary["agreement"] = agreement
            print(f"    Agreement: stable={agreement['stable_count']}, "
                  f"unstable={agreement['unstable_count']}, "
                  f"never={agreement['never_count']}")

        summary["stage1"][dKey] = densitySummary

    # Grep vs Judge
    grepVsJudge = compute_grep_vs_judge(outputDir, densities)
    if grepVsJudge:
        summary["grep_vs_judge"] = grepVsJudge

    # Decision
    summary["conclusion"] = analyze_batch_effect(summary["stage1"], batchSizes)

    # Batch API stats
    oldSummary = load_json(os.path.join(outputDir, "summary.json"))
    summary["batch_api"] = oldSummary.get("batch_api", {})
    summary["batch_api"]["rejudge_batch_id"] = judgeBatch.id
    summary["batch_api"]["judge_requests"] = len(judgeRequests)

    save_json(summary, os.path.join(outputDir, "summary.json"))

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  REJUDGE RESULTS")
    print(f"{'=' * 70}")
    stage1 = summary["stage1"]
    colWidth = 12
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
                row += f" {'N/A':>{colWidth}}"
        print(row)

    conclusion = summary.get("conclusion", {})
    if conclusion:
        print(f"\n  Max diff: {conclusion.get('max_diff_pp', 0):.1f}pp")
        print(f"  Conclusion: {conclusion.get('conclusion', '?')}")

    gvj = summary.get("grep_vs_judge", {})
    if gvj:
        print(f"\n  Grep vs Judge:")
        print(f"    Grep present: {gvj.get('grep_present', 0)}/{gvj.get('total_facts', 0)}")
        print(f"    LLM recalled: {gvj.get('llm_recalled', 0)}/{gvj.get('total_facts', 0)}")
        print(f"    Lost in Middle gap: {gvj.get('lost_in_middle_gap', 0):.1%}")

    print(f"\n  Done! Updated: {outputDir}/summary.json")


if __name__ == "__main__":
    main()

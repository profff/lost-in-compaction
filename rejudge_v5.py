#!python
"""
Re-run ONLY the judge phase for a compaction v5 benchmark run.

Use when the Q&A phase succeeded but the judge batch failed (e.g. network error).
Reads saved answers, rebuilds judge requests, submits via Batch API,
overwrites judgment files, and recomputes summary.json.

Usage:
    python rejudge_v5.py compaction_v5_R4_20260314_0957 --run R4
    python rejudge_v5.py compaction_v5_R4_20260314_0957 --run R4 --judge-model claude-haiku-4-5-20251001 --dry-run
"""

import sys, json, time, re, argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# --- Prompts (same as benchmark_compaction_v5.py) ---

BATCH_JUDGE_PROMPT = """You are evaluating whether an LLM correctly recalled facts from a conversation.

For each entry below, compare the LLM's answer with the expected answer and keywords.

{entries}

For each entry, output:
- "recalled": true if the answer demonstrates knowledge of the expected information (even if imprecise). "I don't recall" or "I don't know" = false.
- "accurate": true if the specific details match (numbers, names, dates). Partial match = false.

Reply with ONLY a JSON array:
[{{"id": "LM_0001", "recalled": true/false, "accurate": true/false, "notes": "brief reason"}}]"""

JUDGE_SYSTEM = "You are an objective evaluator. Answer ONLY with valid JSON."
JUDGE_BATCH_SIZE = 15
JUDGE_CHUNK_SIZE = 50


def parse_llm_json(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```\w*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
    return json.loads(clean)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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
            errDetail = ""
            err = getattr(result.result, 'error', None)
            if err:
                innerErr = getattr(err, 'error', None)
                if innerErr:
                    errDetail = getattr(innerErr, 'message', '') or str(innerErr)
                else:
                    errDetail = str(err)
            results[cid] = {"status": result.result.type, "text": f"[{result.result.type}]: {errDetail}"}

    succeeded = sum(1 for r in results.values() if r["status"] == "succeeded")
    failed = len(results) - succeeded
    print(f"  Batch complete: {succeeded}/{len(results)} succeeded")
    if failed > 0:
        firstErr = next((r for r in results.values() if r["status"] != "succeeded"), None)
        print(f"  WARNING: {failed} failed -> {firstErr['text'] if firstErr else '?'}")
    return results


def load_c0_baseline(runMode):
    """Find the C0 baseline run for the given run mode."""
    base = Path(__file__).parent
    pattern = f"recall_v5_{runMode}_*"
    candidates = sorted(base.glob(pattern))
    for c in candidates:
        sumFile = c / "summary.json"
        if sumFile.exists():
            with open(sumFile) as f:
                return json.load(f), str(c)
    return None, None


def compute_metrics(verdicts, metadata, compactedFactIds=None):
    """Compute recall/accuracy metrics from verdicts."""
    factPositions = {fm["fact_id"]: fm["position_pct"] for fm in metadata["facts"]}
    factTypes = {fm["fact_id"]: fm.get("question_type", "unknown") for fm in metadata["facts"]}

    recalled = sum(1 for v in verdicts if v.get("recalled"))
    accurate = sum(1 for v in verdicts if v.get("accurate"))
    total = len(verdicts)

    # Spatial: early/mid/late thirds
    earlyFacts = [v for v in verdicts if factPositions.get(v["fact_id"], 50) < 33.3]
    midFacts = [v for v in verdicts if 33.3 <= factPositions.get(v["fact_id"], 50) < 66.6]
    lateFacts = [v for v in verdicts if factPositions.get(v["fact_id"], 50) >= 66.6]

    metrics = {
        "recall": recalled / total if total else 0,
        "accuracy": accurate / total if total else 0,
        "facts_recalled": recalled,
        "facts_total": total,
        "recall_early": sum(1 for v in earlyFacts if v.get("recalled")) / max(len(earlyFacts), 1),
        "recall_mid": sum(1 for v in midFacts if v.get("recalled")) / max(len(midFacts), 1),
        "recall_late": sum(1 for v in lateFacts if v.get("recalled")) / max(len(lateFacts), 1),
        "n_early": len(earlyFacts),
        "n_mid": len(midFacts),
        "n_late": len(lateFacts),
    }

    # By category
    byCategory = {}
    for v in verdicts:
        cat = factTypes.get(v["fact_id"], "unknown")
        if cat not in byCategory:
            byCategory[cat] = {"total": 0, "recalled": 0, "accurate": 0}
        byCategory[cat]["total"] += 1
        if v.get("recalled"):
            byCategory[cat]["recalled"] += 1
        if v.get("accurate"):
            byCategory[cat]["accurate"] += 1
    for cat, m in byCategory.items():
        m["recall"] = m["recalled"] / m["total"] if m["total"] else 0
        m["accuracy"] = m["accurate"] / m["total"] if m["total"] else 0
    metrics["by_category"] = byCategory

    # Compacted zone metrics
    if compactedFactIds:
        compVerdicts = [v for v in verdicts if v["fact_id"] in compactedFactIds]
        remVerdicts = [v for v in verdicts if v["fact_id"] not in compactedFactIds]
        metrics["compacted_zone"] = {
            "total": len(compVerdicts),
            "recalled": sum(1 for v in compVerdicts if v.get("recalled")),
            "recall": sum(1 for v in compVerdicts if v.get("recalled")) / max(len(compVerdicts), 1),
        }
        metrics["remaining_zone"] = {
            "total": len(remVerdicts),
            "recalled": sum(1 for v in remVerdicts if v.get("recalled")),
            "recall": sum(1 for v in remVerdicts if v.get("recalled")) / max(len(remVerdicts), 1),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Re-run judge phase for a compaction v5 run")
    parser.add_argument("run_dir", help="Path to the run output directory")
    parser.add_argument("--run", required=True, choices=["R1", "R2", "R3", "R4"],
                        help="Run mode (for C0 baseline lookup and context metadata)")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001", help="Judge model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runDir = Path(args.run_dir)
    base = Path(__file__).parent
    contextMetaDir = base / "data" / "contexts" / f"v5_{args.run}"

    # Discover answer files
    answerFiles = sorted((runDir / "answers").glob("d*_C*_bs*.json"))
    if not answerFiles:
        print(f"Error: no answer files in {runDir / 'answers'}")
        sys.exit(1)

    print(f"Found {len(answerFiles)} answer files:")
    for af in answerFiles:
        print(f"  {af.name}")

    # Load facts from original context metadata
    # Parse answer filenames to get (dKey, cLevel, bs)
    configs = []
    for af in answerFiles:
        parts = af.stem.split("_")  # d80_C1_bs1
        dKey = parts[0]
        cLevel = parts[1]
        bs = int(parts[2][2:])
        configs.append((dKey, cLevel, bs, af))

    # Load original metadata for keyword/answer maps
    allMeta = {}
    for dKey, cLevel, bs, af in configs:
        if dKey not in allMeta:
            metaFile = contextMetaDir / f"{dKey}_seed42_meta.json"
            with open(metaFile, encoding="utf-8") as f:
                allMeta[dKey] = json.load(f)

    # Build judge requests
    judgeRequests = []
    judgeIndex = {}

    for dKey, cLevel, bs, af in configs:
        with open(af, encoding="utf-8") as f:
            archive = json.load(f)

        meta = allMeta[dKey]
        keywordMap = {fm["fact_id"]: fm["keywords"] for fm in meta["facts"]}
        answerMap = {fm["fact_id"]: fm["answer"] for fm in meta["facts"]}

        # Collect all answers from batches
        allAnswers = []
        for batch in archive["batches"]:
            allAnswers.extend(batch["answers"])

        print(f"\n  {dKey}_{cLevel}_bs{bs}: {len(allAnswers)} answers")
        errored = sum(1 for a in allAnswers if a["raw_answer"] == "[errored]")
        if errored:
            print(f"    WARNING: {errored} errored answers (will be judged as not recalled)")

        for jIdx, jStart in enumerate(range(0, len(allAnswers), JUDGE_BATCH_SIZE)):
            jBatch = allAnswers[jStart:jStart + JUDGE_BATCH_SIZE]

            entriesText = "\n\n".join(
                f"[{a['fact_id']}]\n"
                f"  Expected answer: {answerMap.get(a['fact_id'], 'N/A')}\n"
                f"  Expected keywords: {', '.join(keywordMap.get(a['fact_id'], []))}\n"
                f"  LLM answer: {a['raw_answer']}"
                for a in jBatch
            )
            judgePrompt = BATCH_JUDGE_PROMPT.format(entries=entriesText)
            customId = f"judge_{dKey}_{cLevel}_bs{bs}_jb{jIdx}"

            judgeRequests.append({
                "custom_id": customId,
                "params": {
                    "model": args.judge_model,
                    "max_tokens": 4096,
                    "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": judgePrompt}],
                },
            })
            judgeIndex[customId] = {
                "dKey": dKey,
                "cLevel": cLevel,
                "bs": bs,
                "jBatchIdx": jIdx,
                "answers": jBatch,
                "judgePrompt": judgePrompt,
            }

    print(f"\nTotal judge requests: {len(judgeRequests)}")
    print(f"Judge model: {args.judge_model}")

    if args.dry_run:
        print("\n[DRY RUN] Would submit:")
        for r in judgeRequests:
            print(f"  {r['custom_id']}")
        return

    # Submit
    import anthropic
    client = anthropic.Anthropic()

    judgeBatches = submit_chunked(client, judgeRequests, JUDGE_CHUNK_SIZE, "[Rejudge v5]")

    allResults = {}
    for batch in judgeBatches:
        results = wait_for_batch(client, batch.id)
        allResults.update(results)

    errored = sum(1 for r in allResults.values() if r["status"] != "succeeded")
    if errored == len(allResults):
        print(f"\nALL {errored} requests errored. Aborting without overwriting.")
        sys.exit(1)

    # Parse verdicts and save judgment files
    judgeByConfig = {}
    for customId, indexEntry in judgeIndex.items():
        dKey = indexEntry["dKey"]
        cLevel = indexEntry["cLevel"]
        bs = indexEntry["bs"]
        jBatchIdx = indexEntry["jBatchIdx"]
        batchAnswers = indexEntry["answers"]
        judgePrompt = indexEntry["judgePrompt"]

        configKey = (dKey, cLevel, bs)
        if configKey not in judgeByConfig:
            judgeByConfig[configKey] = {"judge_batch_size": JUDGE_BATCH_SIZE, "batches": []}

        result = allResults.get(customId, {"status": "missing", "text": ""})
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

    # Sort and save
    print("\nSaving judgments:")
    for configKey, archive in judgeByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey, cLevel, bs = configKey
        outFile = runDir / "judgments" / f"{dKey}_{cLevel}_bs{bs}.json"
        save_json(archive, str(outFile))
        allVerdicts = [v for b in archive["batches"] for v in b["verdicts"]]
        recalled = sum(1 for v in allVerdicts if v.get("recalled"))
        total = len(allVerdicts)
        print(f"  {dKey}_{cLevel}_bs{bs}: {recalled}/{total} recalled ({recalled/total*100:.1f}%)")

    # Recompute summary
    print("\nRecomputing summary...")
    c0Summary, c0Dir = load_c0_baseline(args.run)

    grepDir = runDir / "grep"
    grepSummary = {}
    for gf in sorted(grepDir.glob("*.json")):
        with open(gf) as f:
            grepSummary[gf.stem] = json.load(f)

    summary = {
        "experiment": "compaction_v5",
        "run_mode": args.run,
        "c0_reference": c0Dir,
        "results": {},
        "grep": grepSummary,
    }

    batchSizes = sorted(set(bs for _, _, bs, _ in configs))

    for configKey, archive in judgeByConfig.items():
        dKey, cLevel, bs = configKey
        resultKey = f"{dKey}_{cLevel}"
        if resultKey not in summary["results"]:
            summary["results"][resultKey] = {}

        # Load compacted metadata for zone info
        compMetaFile = runDir / "contexts" / f"{dKey}_{cLevel}_meta.json"
        compMeta = None
        compactedFactIds = set()
        if compMetaFile.exists():
            with open(compMetaFile) as f:
                compMeta = json.load(f)
            compactedFactIds = set(
                fm["fact_id"] for fm in compMeta["facts"] if fm.get("compacted")
            )

        allVerdicts = [v for b in archive["batches"] for v in b["verdicts"]]
        useMeta = compMeta if compMeta else allMeta[dKey]
        metrics = compute_metrics(allVerdicts, useMeta, compactedFactIds)

        # Add C0 baseline delta
        if c0Summary:
            c0Data = c0Summary.get("results", {}).get(dKey, {})
            bsKey = f"bs{bs}"
            if bsKey in c0Data:
                c0Recall = c0Data[bsKey].get("recall", 0)
                metrics["c0_recall"] = c0Recall
                metrics["delta_vs_c0"] = metrics["recall"] - c0Recall

        # Add grep
        grepKey = f"{dKey}_{cLevel}"
        if grepKey in grepSummary:
            summary["results"][resultKey]["grep"] = grepSummary[grepKey]

        summary["results"][resultKey][f"bs{bs}"] = metrics

    save_json(summary, str(runDir / "summary.json"))
    print(f"Summary saved to {runDir / 'summary.json'}")
    print("\nDone!")


if __name__ == "__main__":
    main()

#!python3
"""
Resume v4 benchmark from a crashed run.

Retrieves Q&A batch results, saves answer archives, then submits judge batch.
Reuses all logic from benchmark_recall_v4.py.
"""
import json
import sys
import time
import math
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from benchmark_recall_v4 import (
    SYSTEM_PROMPT, BATCH_JUDGE_PROMPT, JUDGE_SYSTEM,
    CONTEXTS_DIR, load_context, extract_facts_from_meta,
    grep_keywords, parse_llm_json, submit_batch, wait_for_batch,
    compute_metrics, compute_agreement, save_json, print_summary,
)

QA_BATCH_ID = "msgbatch_018a7bKPEoDBwuGpDKFqn9ex"
DENSITIES = [4, 8, 19]
BATCH_SIZES = [1, 5, 10]
SEED = 42
MODEL = "claude-haiku-4-5-20251001"
JUDGE_BATCH_SIZE = 15


def main():
    import anthropic
    client = anthropic.Anthropic()

    # Output dir — reuse the one from crashed run or create new
    outputDir = Path("recall_v4_20260210_1703")
    if not outputDir.exists():
        outputDir = Path(f"recall_v4_resumed_{datetime.now().strftime('%Y%m%d_%H%M')}")
    for sub in ["grep", "answers", "judgments"]:
        (outputDir / sub).mkdir(parents=True, exist_ok=True)

    # Load contexts + facts
    print("  Loading contexts...")
    allContexts = {}
    allMetadata = {}
    allFacts = {}
    grepSummary = {}

    for density in DENSITIES:
        dKey = f"d{density}"
        messages, metadata = load_context(density, SEED)
        facts = extract_facts_from_meta(metadata)
        allContexts[dKey] = messages
        allMetadata[dKey] = metadata
        allFacts[dKey] = facts

        grepResults = grep_keywords(messages, facts)
        grepPresent = sum(1 for g in grepResults if g["all_present"])
        grepSummary[dKey] = {
            "facts_total": len(facts),
            "all_present": grepPresent,
            "any_present": sum(1 for g in grepResults if g["any_present"]),
            "recall_upper_bound": grepPresent / len(facts) if facts else 0,
        }
        print(f"    {dKey}: {len(facts)} facts, grep {grepPresent}/{len(facts)}")

    # Rebuild Q&A request index (same logic as benchmark_recall_v4.py)
    print(f"\n  Rebuilding Q&A index...")
    qaIndex = {}
    for dKey in sorted(allContexts.keys()):
        facts = allFacts[dKey]
        for bs in BATCH_SIZES:
            for batchIdx, batchStart in enumerate(range(0, len(facts), bs)):
                batchFacts = facts[batchStart:batchStart + bs]
                from benchmark_recall_v4 import BATCH_QUESTION_PROMPT
                questionsText = "\n".join(
                    f"- [{fid}] {question}"
                    for fid, question, _, _ in batchFacts
                )
                prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
                customId = f"qa_{dKey}_bs{bs}_b{batchIdx}"
                qaIndex[customId] = {
                    "dKey": dKey,
                    "bs": bs,
                    "batchIdx": batchIdx,
                    "facts": batchFacts,
                    "prompt": prompt,
                }

    print(f"  {len(qaIndex)} Q&A requests indexed")

    # Retrieve Q&A results
    print(f"\n  Retrieving Q&A batch {QA_BATCH_ID}...")
    qaResults = {}
    for result in client.messages.batches.results(QA_BATCH_ID):
        cid = result.custom_id
        if result.result.type == "succeeded":
            text = ""
            for block in result.result.message.content:
                if block.type == "text":
                    text = block.text
                    break
            qaResults[cid] = {"status": "succeeded", "text": text}
        else:
            qaResults[cid] = {"status": result.result.type, "text": f"[{result.result.type}]"}

    succeeded = sum(1 for r in qaResults.values() if r["status"] == "succeeded")
    print(f"  Retrieved: {succeeded}/{len(qaResults)} succeeded")

    # Parse Q&A results into answer archives
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

        for pos, (fid, question, _, _) in enumerate(batchFacts):
            batchResult["answers"].append({
                "fact_id": fid,
                "question": question,
                "raw_answer": answerMap.get(fid, f"[{result.get('status', 'error')}]"),
                "batch_position": pos,
            })

        answersByConfig[configKey]["batches"].append(batchResult)

    for configKey, archive in answersByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey, bs = configKey
        save_json(archive, str(outputDir / "answers" / f"{dKey}_bs{bs}.json"))

    print(f"  Saved {len(answersByConfig)} answer archives")

    # Build + submit judge batch
    print(f"\n  Building judge requests...")
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

        for jBatchIdx, jBatchStart in enumerate(range(0, len(allAnswers), JUDGE_BATCH_SIZE)):
            jBatch = allAnswers[jBatchStart:jBatchStart + JUDGE_BATCH_SIZE]

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
                    "model": MODEL,
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

    judgeBatch = submit_batch(client, judgeRequests, description="[Judge v2]")
    judgeResults = wait_for_batch(client, judgeBatch.id, pollInterval=30)

    # Parse judge results
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
                "judge_batch_size": JUDGE_BATCH_SIZE,
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

    for configKey, archive in judgeByConfig.items():
        archive["batches"].sort(key=lambda b: b["batch_id"])
        dKey, bs = configKey
        save_json(archive, str(outputDir / "judgments" / f"{dKey}_bs{bs}.json"))

    print(f"  Saved {len(judgeByConfig)} judgment archives")

    # Compute metrics
    print(f"\n  Computing metrics...")
    summary = {"results": {}, "grep": grepSummary}

    for dKey in sorted(allContexts.keys()):
        metadata = allMetadata[dKey]
        densityResults = {"grep": grepSummary[dKey]}
        verdictsPerBs = {}

        for bs in BATCH_SIZES:
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
                  f"accuracy={metrics['accuracy']:.1%}")

        if len(verdictsPerBs) > 1:
            agreement = compute_agreement(verdictsPerBs)
            densityResults["agreement"] = agreement

        summary["results"][dKey] = densityResults

    summary["batch_api"] = {
        "qa_batch_id": QA_BATCH_ID,
        "judge_batch_id": judgeBatch.id,
        "qa_requests": len(qaIndex),
        "judge_requests": len(judgeRequests),
        "resumed": True,
    }

    save_json(summary, str(outputDir / "summary.json"))
    print_summary(summary, BATCH_SIZES)
    print(f"\n  Results saved to: {outputDir}/")


if __name__ == "__main__":
    main()

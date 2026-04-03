#!python
"""
Re-run ONLY the judge phase for an iterative (v6) or recall (v5) benchmark run.

Use when the Q&A phase succeeded but the judge batch failed (e.g. API outage).
Reads saved answers, rebuilds judge requests, submits a new batch,
and overwrites the judgment files.

Usage (v6 iterative):
    python rejudge.py iterative_v6_R4_20260302_2224 \
        --meta data/conversations/v6_R4/nested_d160_1M_seed42_meta.json

    python rejudge.py <run-dir> --meta <metadata.json> [--model MODEL] [--dry-run]
"""

import sys, os, json, time, re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Force unbuffered output
_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# --- Prompts (same as benchmark_iterative_v6.py) ---

BATCH_JUDGE_PROMPT = """Evaluate whether each answer is correct.

{entries}

For each entry, check:
1. "recalled": Does the answer demonstrate knowledge of the expected information? "I don't recall" = false.
2. "accurate": Does the answer match the expected answer? Partial matches or wrong values = false.
   Example: expected "17 days", answer "10 days" -> recalled=true (knows about days), accurate=false (wrong number).

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-run judge phase for a benchmark run")
    parser.add_argument("run_dir", help="Path to the run output directory")
    parser.add_argument("--meta", required=True, help="Path to conversation metadata JSON (with facts)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Judge model")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    runDir = Path(args.run_dir)
    if not runDir.is_dir():
        print(f"Error: {runDir} is not a directory")
        sys.exit(1)

    # Load metadata -> facts
    with open(args.meta, encoding="utf-8") as f:
        metadata = json.load(f)

    facts = []
    for fm in metadata["facts"]:
        facts.append((
            fm["fact_id"],
            fm["question"],
            fm["answer"],
            fm["keywords"],
            fm.get("question_type", "unknown"),
        ))
    print(f"Loaded {len(facts)} facts from {Path(args.meta).name}")

    # Discover strategies from answer files
    answerFiles = sorted((runDir / "answers").glob("S*_bs*.json"))
    if not answerFiles:
        print(f"Error: no answer files found in {runDir / 'answers'}")
        sys.exit(1)

    strategyKeys = []
    bs = None
    for af in answerFiles:
        parts = af.stem.split("_")
        sk = parts[0]
        bsVal = int(parts[1][2:])
        if sk not in strategyKeys:
            strategyKeys.append(sk)
        if bs is None:
            bs = bsVal
    print(f"Strategies: {strategyKeys}, Q={bs}")

    # Load answers per strategy
    answersByStrategy = {}
    for sk in strategyKeys:
        af = runDir / "answers" / f"{sk}_bs{bs}.json"
        with open(af, encoding="utf-8") as f:
            answersByStrategy[sk] = json.load(f)
        nAnswers = len(answersByStrategy[sk])
        noAnswer = sum(1 for v in answersByStrategy[sk].values() if v == "[no answer]")
        print(f"  {sk}: {nAnswers} answers ({noAnswer} missing)")

    # Build judge requests
    judgeRequests = []
    for sk in strategyKeys:
        entries = []
        for fid, question, answer, keywords, qType in facts:
            llmAnswer = answersByStrategy[sk].get(fid, "[no answer]")
            entries.append({
                "id": fid,
                "question": question,
                "expected_answer": answer,
                "expected_keywords": keywords,
                "llm_answer": llmAnswer,
            })

        for jIdx in range(0, len(entries), JUDGE_BATCH_SIZE):
            batch = entries[jIdx:jIdx + JUDGE_BATCH_SIZE]
            entriesText = json.dumps(batch, indent=2)
            prompt = BATCH_JUDGE_PROMPT.format(entries=entriesText)
            customId = f"judge_{sk}_b{jIdx // JUDGE_BATCH_SIZE}"

            judgeRequests.append({
                "custom_id": customId,
                "params": {
                    "model": args.model,
                    "max_tokens": 4096,
                    "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })

    print(f"\nTotal judge requests: {len(judgeRequests)}")
    print(f"Model: {args.model}")

    if args.dry_run:
        print("\n[DRY RUN] Would submit the requests above.")
        for r in judgeRequests:
            print(f"  {r['custom_id']}")
        return

    # Submit
    import anthropic
    client = anthropic.Anthropic()

    judgeBatches = submit_chunked(client, judgeRequests, JUDGE_CHUNK_SIZE, "Rejudge")

    allResults = {}
    for batch in judgeBatches:
        results = wait_for_batch(client, batch.id)
        allResults.update(results)

    # Check for total failure
    errored = sum(1 for r in allResults.values() if r["status"] != "succeeded")
    if errored == len(allResults):
        print(f"\nALL {errored} requests errored. Aborting without overwriting.")
        sys.exit(1)

    # Parse verdicts per strategy
    verdictsByStrategy = {sk: [] for sk in strategyKeys}
    for cid, result in allResults.items():
        parts = cid.split("_")
        sk = parts[1]

        if result["status"] != "succeeded":
            print(f"  WARNING: {cid} failed: {result['text']}")
            continue

        try:
            verdicts = parse_llm_json(result["text"])
            for v in verdicts:
                v["fact_id"] = v.pop("id", v.get("fact_id", "?"))
            verdictsByStrategy[sk].extend(verdicts)
        except Exception as e:
            print(f"  WARNING: {cid} parse error: {e}")

    # Save judgments (overwrite)
    print()
    for sk in strategyKeys:
        outFile = runDir / "judgments" / f"{sk}_bs{bs}.json"
        with open(outFile, "w", encoding="utf-8") as f:
            json.dump({"verdicts": verdictsByStrategy[sk]}, f, indent=2)
        total = len(verdictsByStrategy[sk])
        recalled = sum(1 for v in verdictsByStrategy[sk] if v.get("recalled"))
        pct = recalled / total * 100 if total else 0
        print(f"  [{sk}] {recalled}/{total} recalled ({pct:.1f}%)")

    print(f"\nDone! Judgments saved to {runDir / 'judgments'}")


if __name__ == "__main__":
    main()

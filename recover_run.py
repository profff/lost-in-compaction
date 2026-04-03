#!python
"""
Recover a crashed benchmark run by fetching results from completed Anthropic batches.

Usage:
    python recover_run.py <run_dir> --meta <metadata.json> --qa-batches <id1> <id2> ... --judge-batch <id>
    python recover_run.py iterative_v6_R4_20260303_0838 \
        --meta data/conversations/v6_R4/nested_d160_1M_seed42_meta.json \
        --qa-batches msgbatch_011nmth... msgbatch_018Qzjsz... \
        --judge-batch msgbatch_013eyUfN...
"""

import sys, json, time, re, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


def parse_llm_json(text):
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```\w*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
    return json.loads(clean)


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recover crashed benchmark run from Anthropic batch results")
    parser.add_argument("run_dir", help="Path to the run output directory")
    parser.add_argument("--meta", required=True, help="Path to conversation metadata JSON")
    parser.add_argument("--qa-batches", nargs="+", required=True, help="QA batch IDs")
    parser.add_argument("--judge-batch", required=True, help="Judge batch ID")
    args = parser.parse_args()

    runDir = Path(args.run_dir)

    # Load config
    with open(runDir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    bs = config["qa_batch_size"]
    strategyKeys = config["strategies"]
    print(f"Run: {runDir.name}, Q={bs}, strategies={strategyKeys}")

    # Load metadata -> facts
    with open(args.meta, encoding="utf-8") as f:
        metadata = json.load(f)
    facts = []
    for fm in metadata["facts"]:
        facts.append((
            fm["fact_id"], fm["question"], fm["answer"],
            fm["keywords"], fm.get("question_type", "unknown"),
        ))
    print(f"Loaded {len(facts)} facts")

    import anthropic
    client = anthropic.Anthropic()

    # === Recover QA answers ===
    print(f"\n=== Recovering QA from {len(args.qa_batches)} batches ===")
    allQaResults = {}
    for batchId in args.qa_batches:
        results = wait_for_batch(client, batchId)
        allQaResults.update(results)

    print(f"Total QA results: {len(allQaResults)}")

    # Parse answers (same logic as benchmark_iterative_v6.py)
    answersByStrategy = {sk: {} for sk in strategyKeys}
    for cid, result in allQaResults.items():
        parts = cid.split("_")
        sk = parts[1]
        batchIdx = int(parts[3][1:])

        if result["status"] != "succeeded":
            print(f"  WARNING: {cid} failed: {result['text']}")
            continue

        try:
            answers = parse_llm_json(result["text"])
            if isinstance(answers, list):
                answerMap = {a["id"]: a.get("answer", "[parse error]") for a in answers}
            elif isinstance(answers, dict):
                answerMap = {answers["id"]: answers.get("answer", "[parse error]")}
            else:
                answerMap = {}
        except Exception as e:
            print(f"  WARNING: {cid} parse error: {e}")
            answerMap = {}

        batchStart = batchIdx * bs
        batchFacts = facts[batchStart:batchStart + bs]
        for fid, _, _, _, _ in batchFacts:
            answersByStrategy[sk][fid] = answerMap.get(fid, "[no answer]")

    # Save answers
    (runDir / "answers").mkdir(exist_ok=True)
    for sk in strategyKeys:
        outFile = runDir / "answers" / f"{sk}_bs{bs}.json"
        with open(outFile, "w", encoding="utf-8") as f:
            json.dump(answersByStrategy[sk], f, indent=2)
        answered = sum(1 for a in answersByStrategy[sk].values() if a != "[no answer]")
        print(f"  [{sk}] {answered}/{len(facts)} answers saved")

    # === Recover Judge ===
    print(f"\n=== Recovering Judge batch ===")
    judgeResults = wait_for_batch(client, args.judge_batch)

    errored = sum(1 for r in judgeResults.values() if r["status"] != "succeeded")
    if errored == len(judgeResults):
        print(f"ALL {errored} judge requests errored. Saving empty verdicts.")

    verdictsByStrategy = {sk: [] for sk in strategyKeys}
    for cid, result in judgeResults.items():
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

    # Save judgments
    (runDir / "judgments").mkdir(exist_ok=True)
    for sk in strategyKeys:
        outFile = runDir / "judgments" / f"{sk}_bs{bs}.json"
        with open(outFile, "w", encoding="utf-8") as f:
            json.dump({"verdicts": verdictsByStrategy[sk]}, f, indent=2)
        total = len(verdictsByStrategy[sk])
        recalled = sum(1 for v in verdictsByStrategy[sk] if v.get("recalled"))
        pct = recalled / total * 100 if total else 0
        print(f"  [{sk}] {recalled}/{total} recalled ({pct:.1f}%)")

    print(f"\nDone! Results saved to {runDir}")


if __name__ == "__main__":
    main()

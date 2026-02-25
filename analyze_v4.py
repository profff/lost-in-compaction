#!python3
"""Analyze v4 recall results fact by fact."""
import json
from pathlib import Path

RESULTS_DIR = Path("recall_v4_20260210_1703")
CONTEXTS_DIR = Path("data/contexts/recall_190K")

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    densities = [4, 8, 19]
    batchSizes = [1, 5, 10]

    for density in densities:
        meta = load(CONTEXTS_DIR / f"d{density}_seed42_meta.json")
        facts = {fm["fact_id"]: fm for fm in meta["facts"]}

        print(f"\n{'='*80}")
        print(f"  d{density} — {len(facts)} facts, ~{meta.get('real_tokens', meta['est_tokens']):,} tokens")
        print(f"{'='*80}")

        # Collect answers and verdicts per fact per batch size
        answersByFact = {}  # fact_id -> {bs -> raw_answer}
        verdictsByFact = {}  # fact_id -> {bs -> {recalled, accurate, notes}}

        for bs in batchSizes:
            ansFile = RESULTS_DIR / "answers" / f"d{density}_bs{bs}.json"
            judgeFile = RESULTS_DIR / "judgments" / f"d{density}_bs{bs}.json"

            answers = load(ansFile)
            judgments = load(judgeFile)

            # Flatten answers
            for batch in answers["batches"]:
                for a in batch["answers"]:
                    fid = a["fact_id"]
                    if fid not in answersByFact:
                        answersByFact[fid] = {}
                    answersByFact[fid][bs] = a["raw_answer"]

            # Flatten verdicts
            for batch in judgments["batches"]:
                for v in batch["verdicts"]:
                    fid = v["fact_id"]
                    if fid not in verdictsByFact:
                        verdictsByFact[fid] = {}
                    verdictsByFact[fid][bs] = v

        # Print fact by fact
        for fid in sorted(facts.keys()):
            fm = facts[fid]
            pos = fm["position_pct"]
            nTurns = fm["n_turns"]
            estTok = fm["est_tokens"]

            # Recall status across batch sizes
            statuses = []
            for bs in batchSizes:
                v = verdictsByFact.get(fid, {}).get(bs, {})
                recalled = v.get("recalled", False)
                accurate = v.get("accurate", False)
                if recalled and accurate:
                    statuses.append(f"bs{bs}=OK")
                elif recalled:
                    statuses.append(f"bs{bs}=RECALL(inaccurate)")
                else:
                    statuses.append(f"bs{bs}=MISS")

            # Determine overall status
            anyRecalled = any(verdictsByFact.get(fid, {}).get(bs, {}).get("recalled", False)
                             for bs in batchSizes)
            allRecalled = all(verdictsByFact.get(fid, {}).get(bs, {}).get("recalled", False)
                             for bs in batchSizes)

            if allRecalled:
                tag = "STABLE"
            elif anyRecalled:
                tag = "UNSTABLE"
            else:
                tag = "NEVER"

            print(f"\n  [{tag:8s}] {fid} @ {pos:.0f}% ({nTurns} turns, ~{estTok:,} tok)")
            print(f"  Q: {fm['question']}")
            print(f"  Expected: {fm['answer'][:100]}")
            print(f"  Keywords: {fm['keywords'][:8]}")
            print(f"  Verdicts: {' | '.join(statuses)}")

            # Show answers (abbreviated)
            for bs in batchSizes:
                ans = answersByFact.get(fid, {}).get(bs, "N/A")
                v = verdictsByFact.get(fid, {}).get(bs, {})
                notes = v.get("notes", "")
                print(f"    bs={bs:2d}: {ans[:120]}")
                if notes:
                    print(f"           judge: {notes[:100]}")

        # Summary
        print(f"\n  --- Summary d{density} ---")
        stableCount = sum(1 for fid in facts
                         if all(verdictsByFact.get(fid, {}).get(bs, {}).get("recalled", False)
                                for bs in batchSizes))
        unstableCount = sum(1 for fid in facts
                           if (any(verdictsByFact.get(fid, {}).get(bs, {}).get("recalled", False)
                                   for bs in batchSizes)
                               and not all(verdictsByFact.get(fid, {}).get(bs, {}).get("recalled", False)
                                           for bs in batchSizes)))
        neverCount = len(facts) - stableCount - unstableCount
        print(f"  Stable: {stableCount}, Unstable: {unstableCount}, Never: {neverCount}")


if __name__ == "__main__":
    main()

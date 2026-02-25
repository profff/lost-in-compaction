#!python3
"""
Step 1: Extract unique padding sessions from LongMemEval S dataset.

Input:  longmemeval_data/longmemeval_s_cleaned.json
Output: data/padding_pool.jsonl (one session per line)

Each line: {"id", "source", "turns": [{role, content}], "chars", "est_tokens"}
"""

import json
from pathlib import Path

INPUT = Path("longmemeval_data/longmemeval_s_cleaned.json")
OUTPUT = Path("data/padding_pool.jsonl")


def main():
    print(f"Loading {INPUT} ...")
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} instances loaded")

    # Extract unique padding sessions (not in answer_session_ids)
    uniqueSessions = {}  # id -> {turns, chars}
    for inst in data:
        ansIds = set(inst["answer_session_ids"])
        for sessId, sess in zip(inst["haystack_session_ids"], inst["haystack_sessions"]):
            if sessId not in ansIds and sessId not in uniqueSessions:
                # Strip has_answer metadata from turns (padding shouldn't have it,
                # but clean just in case)
                cleanTurns = [
                    {"role": t["role"], "content": t["content"]}
                    for t in sess
                ]
                chars = sum(len(t["content"]) for t in cleanTurns)
                uniqueSessions[sessId] = {
                    "id": sessId,
                    "source": _detect_source(sessId),
                    "turns": cleanTurns,
                    "chars": chars,
                    "est_tokens": chars // 3,  # rough estimate at 3.1 chars/tok
                }

    print(f"  {len(uniqueSessions)} unique padding sessions extracted")

    # Stats
    allChars = [s["chars"] for s in uniqueSessions.values()]
    totalChars = sum(allChars)
    allChars.sort()
    print(f"  Total chars: {totalChars:,} (~{totalChars // 3:,} tokens)")
    print(f"  Per session: min={min(allChars):,} median={allChars[len(allChars)//2]:,} max={max(allChars):,}")

    # Source breakdown
    sources = {}
    for s in uniqueSessions.values():
        src = s["source"]
        sources[src] = sources.get(src, 0) + 1
    print(f"  Sources: {dict(sorted(sources.items(), key=lambda x: -x[1]))}")

    # Write JSONL
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for sess in uniqueSessions.values():
            f.write(json.dumps(sess, ensure_ascii=False) + "\n")

    fileSizeMB = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n  Written: {OUTPUT} ({fileSizeMB:.1f} MB, {len(uniqueSessions)} sessions)")


def _detect_source(sessId: str) -> str:
    """Detect session source from ID prefix."""
    if sessId.startswith("sharegpt_"):
        return "sharegpt"
    elif sessId.startswith("ultrachat_"):
        return "ultrachat"
    else:
        return "longmemeval_generated"


if __name__ == "__main__":
    main()

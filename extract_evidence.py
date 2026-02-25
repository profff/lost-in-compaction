#!python3
"""
Step 2: Extract evidence sessions + questions from LongMemEval oracle dataset.

Input:  longmemeval_data/longmemeval_oracle.json
Output: data/evidence_longmemeval.json

Each entry: {fact_id, question, answer, keywords, evidence_turns,
             has_answer_indices, source, original_question_id, question_type,
             est_tokens}
"""

import json
import re
from pathlib import Path

INPUT = Path("longmemeval_data/longmemeval_oracle.json")
OUTPUT = Path("data/evidence_longmemeval.json")


def extract_keywords(answer: str, evidence_turns: list[dict]) -> list[str]:
    """
    Extract verification keywords from the expected answer.

    Strategy: take meaningful tokens from the answer (proper nouns, numbers,
    specific terms). Skip common stop words.
    """
    stopWords = {
        "the", "a", "an", "is", "was", "were", "are", "been", "be", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "and", "or", "but", "not", "no",
        "nor", "so", "yet", "both", "either", "neither", "each", "every",
        "all", "any", "few", "more", "most", "some", "such", "than", "that",
        "this", "these", "those", "it", "its", "i", "my", "me", "we", "our",
        "you", "your", "he", "his", "she", "her", "they", "their", "them",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "about", "up", "out", "if", "then", "also", "just", "only", "very",
    }

    keywords = []

    # Extract tokens from the answer
    tokens = re.findall(r"[\w']+", answer)
    for tok in tokens:
        tokLower = tok.lower()
        if tokLower not in stopWords and len(tok) >= 2:
            keywords.append(tok)

    # Also extract numbers and specific patterns
    numbers = re.findall(r"\d+(?:\.\d+)?", answer)
    for num in numbers:
        if num not in keywords:
            keywords.append(num)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for kw in keywords:
        kwLower = kw.lower()
        if kwLower not in seen:
            seen.add(kwLower)
            deduped.append(kw)

    return deduped


def main():
    print(f"Loading {INPUT} ...")
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} instances loaded")

    evidence = []
    skipped = 0

    for idx, inst in enumerate(data):
        questionId = inst["question_id"]
        question = inst["question"]
        answer = str(inst["answer"])
        questionType = inst["question_type"]

        # Collect all evidence sessions for this instance
        ansIds = set(inst["answer_session_ids"])
        allEvTurns = []
        hasAnswerIndices = []

        for sessId, sess in zip(inst["haystack_session_ids"], inst["haystack_sessions"]):
            if sessId in ansIds:
                baseIdx = len(allEvTurns)
                for i, turn in enumerate(sess):
                    cleanTurn = {"role": turn["role"], "content": turn["content"]}
                    allEvTurns.append(cleanTurn)
                    if turn.get("has_answer"):
                        hasAnswerIndices.append(baseIdx + i)

        if not allEvTurns:
            skipped += 1
            continue

        chars = sum(len(t["content"]) for t in allEvTurns)
        keywords = extract_keywords(answer, allEvTurns)

        if not keywords:
            # Fallback: use first 3 words of the answer
            keywords = answer.split()[:3]

        factId = f"LM_{idx:04d}"
        evidence.append({
            "fact_id": factId,
            "question": question,
            "answer": answer,
            "keywords": keywords,
            "evidence_turns": allEvTurns,
            "has_answer_indices": hasAnswerIndices,
            "source": "longmemeval",
            "original_question_id": questionId,
            "question_type": questionType,
            "n_turns": len(allEvTurns),
            "chars": chars,
            "est_tokens": chars // 3,
        })

    print(f"  Extracted {len(evidence)} evidence entries (skipped {skipped})")

    # Stats
    types = {}
    for e in evidence:
        qt = e["question_type"]
        types[qt] = types.get(qt, 0) + 1
    print(f"  Question types: {dict(sorted(types.items(), key=lambda x: -x[1]))}")

    tokSizes = sorted(e["est_tokens"] for e in evidence)
    totalTok = sum(tokSizes)
    print(f"  Tokens: total={totalTok:,} median={tokSizes[len(tokSizes)//2]:,} "
          f"min={tokSizes[0]:,} max={tokSizes[-1]:,}")

    kwCounts = [len(e["keywords"]) for e in evidence]
    print(f"  Keywords per fact: min={min(kwCounts)} median={sorted(kwCounts)[len(kwCounts)//2]} "
          f"max={max(kwCounts)}")

    # Show a few examples
    print(f"\n  --- Examples ---")
    for e in evidence[:3]:
        print(f"  {e['fact_id']} ({e['question_type']}):")
        print(f"    Q: {e['question']}")
        print(f"    A: {e['answer']}")
        print(f"    Keywords: {e['keywords'][:6]}")
        print(f"    Evidence: {e['n_turns']} turns, ~{e['est_tokens']} tok")
        print()

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    fileSizeMB = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  Written: {OUTPUT} ({fileSizeMB:.1f} MB)")


if __name__ == "__main__":
    main()

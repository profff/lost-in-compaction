#!python3
"""
Build a long conversation (default 5M tokens) with interleaved facts for
iterative compaction benchmarking (Phase 3 / Part C).

Adapts build_contexts_v5.py logic to produce a single massive conversation
instead of a 190K context.

Output:
  data/conversations/v6_R4/d{N}_seed{S}.json + _meta.json

Usage:
    python build_conversation_v6.py --dry-run
    python build_conversation_v6.py --density 80
    python build_conversation_v6.py --density 80 --target-tokens 500000
"""

import json
import random
import argparse
import statistics
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# PATHS
# ============================================================================

PADDING_POOL = Path("data/padding_pool.jsonl")
EVIDENCE_LM = Path("data/evidence_longmemeval.json")
OUTPUT_BASE = Path("data/conversations")

DEFAULT_TARGET = 5_000_000
CHARS_PER_TOKEN = 4.3

FILTERED_OUT_TYPES = {"temporal-reasoning", "multi-session"}


# ============================================================================
# DATA LOADING (reused from build_contexts_v5.py)
# ============================================================================

def load_padding_pool():
    sessions = []
    with open(PADDING_POOL, encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line))
    return sessions


def load_evidence():
    with open(EVIDENCE_LM, encoding="utf-8") as f:
        return json.load(f)


def estimate_tokens_chars(chars):
    return int(chars / CHARS_PER_TOKEN)


# ============================================================================
# EVIDENCE PROCESSING (reused from build_contexts_v5.py)
# ============================================================================

def chop_evidence(entry):
    turns = entry["evidence_turns"]
    indices = entry.get("has_answer_indices", [])

    if not indices:
        keptTurns = turns[:3]
    else:
        expanded = set()
        for idx in indices:
            for j in range(max(0, idx - 1), min(len(turns), idx + 2)):
                expanded.add(j)
        keptTurns = [turns[i] for i in sorted(expanded) if i < len(turns)]

    chars = sum(len(t["content"]) for t in keptTurns)
    estTokens = int(chars / CHARS_PER_TOKEN)

    chopped = dict(entry)
    chopped["evidence_turns"] = keptTurns
    chopped["chars"] = chars
    chopped["est_tokens"] = estTokens
    chopped["chopped"] = True
    chopped["original_n_turns"] = len(turns)
    chopped["kept_n_turns"] = len(keptTurns)
    return chopped


def filter_evidence(evidence, excludeTypes):
    return [e for e in evidence if e.get("question_type") not in excludeTypes]


def prepare_evidence_r4(rawEvidence):
    filtered = filter_evidence(rawEvidence, FILTERED_OUT_TYPES)
    chopped = [chop_evidence(e) for e in filtered]
    return chopped


# ============================================================================
# SELECTION
# ============================================================================

def select_evidence(evidence, density, seed):
    rng = random.Random(seed)
    pool = sorted(evidence, key=lambda e: e["fact_id"])
    rng.shuffle(pool)
    selected = pool[:density]
    selected.sort(key=lambda e: e["fact_id"])
    return selected


def select_padding(paddingPool, targetTokens, evidenceTokens, seed):
    """Select padding sessions to fill the token budget.

    For large targets (5M+), we may need most of the padding pool.
    """
    rng = random.Random(seed + 1000)
    budget = targetTokens - evidenceTokens

    if budget <= 0:
        print(f"    WARNING: evidence ({evidenceTokens:,} tok) exceeds target ({targetTokens:,})")
        return []

    pool = list(paddingPool)
    rng.shuffle(pool)

    selected = []
    usedTokens = 0
    for sess in pool:
        sessTok = estimate_tokens_chars(sess["chars"])
        if usedTokens + sessTok > budget:
            if usedTokens >= budget * 0.95:
                break
            continue
        selected.append(sess)
        usedTokens += sessTok
        if usedTokens >= budget * 0.98:
            break

    return selected


# ============================================================================
# INTERLEAVE
# ============================================================================

def interleave(paddingSessions, evidenceEntries):
    """Interleave evidence into padding at uniformly spaced positions."""
    nEvidence = len(evidenceEntries)
    nPadding = len(paddingSessions)

    if nEvidence == 0:
        messages = []
        for sess in paddingSessions:
            messages.extend(sess["turns"])
        return messages, []

    chunkSize = nPadding // (nEvidence + 1)
    remainder = nPadding % (nEvidence + 1)

    chunks = []
    idx = 0
    for i in range(nEvidence + 1):
        end = idx + chunkSize + (1 if i < remainder else 0)
        chunks.append(paddingSessions[idx:end])
        idx = end

    messages = []
    factMeta = []

    for i in range(nEvidence + 1):
        for sess in chunks[i]:
            messages.extend(sess["turns"])

        if i < nEvidence:
            ev = evidenceEntries[i]
            startIdx = len(messages)
            messages.extend(ev["evidence_turns"])
            endIdx = len(messages)

            meta = {
                "fact_id": ev["fact_id"],
                "source": ev["source"],
                "question_type": ev.get("question_type", "unknown"),
                "question": ev["question"],
                "answer": ev["answer"],
                "keywords": ev["keywords"],
                "message_start": startIdx,
                "message_end": endIdx,
                "n_turns": len(ev["evidence_turns"]),
                "est_tokens": ev["est_tokens"],
                "chopped": ev.get("chopped", False),
            }
            if ev.get("chopped"):
                meta["original_n_turns"] = ev.get("original_n_turns", 0)
                meta["kept_n_turns"] = ev.get("kept_n_turns", 0)
            factMeta.append(meta)

    totalMsgs = len(messages)
    for fm in factMeta:
        fm["position_pct"] = round(fm["message_start"] / totalMsgs * 100, 1)

    return messages, factMeta


# ============================================================================
# MAIN
# ============================================================================

def build_nested(prefixConvFile, prefixMetaFile, newDensity, targetTokens,
                 seed, outputDir, dryRun=False):
    """Build a nested conversation: existing prefix + new segment with new facts.

    The first part of the conversation is identical to the prefix.
    New facts (G2, G3, ...) are interleaved in the appended segment.
    """
    print("=" * 70)
    print(f"  BUILD NESTED CONVERSATION v6")
    print(f"  Prefix: {prefixConvFile}")
    print(f"  New facts per segment: {newDensity}")
    print(f"  Target total: {targetTokens:,} tokens ({targetTokens / 1_000_000:.1f}M)")
    print("=" * 70)

    # Load prefix
    print("\n  Loading prefix conversation...")
    with open(prefixConvFile, encoding="utf-8") as f:
        prefixMessages = json.load(f)
    with open(prefixMetaFile, encoding="utf-8") as f:
        prefixMeta = json.load(f)

    prefixTokens = prefixMeta["est_tokens"]
    prefixFacts = prefixMeta["facts"]
    prefixFactIds = set(fm["fact_id"] for fm in prefixFacts)
    generation = prefixMeta.get("max_generation", 1)

    print(f"    {len(prefixMessages)} messages, ~{prefixTokens:,} tokens")
    print(f"    {len(prefixFacts)} facts (G1–G{generation})")

    # Compute new segment budget
    segmentTokens = targetTokens - prefixTokens
    if segmentTokens <= 0:
        print(f"    ERROR: prefix ({prefixTokens:,}) >= target ({targetTokens:,})")
        return
    print(f"    New segment budget: ~{segmentTokens:,} tokens")

    # Load evidence pool and exclude already-used facts
    print("\n  Loading evidence pool...")
    rawEvidence = load_evidence()
    evidence = prepare_evidence_r4(rawEvidence)

    # Reproduce same shuffled order as original (seed=42) to pick next slice
    rng = random.Random(seed)
    pool = sorted(evidence, key=lambda e: e["fact_id"])
    rng.shuffle(pool)

    # Skip facts already used in prefix (take from position after prefix facts)
    alreadyUsed = 0
    for e in pool:
        if e["fact_id"] in prefixFactIds:
            alreadyUsed += 1
    # Find the first unused fact in shuffled order
    available = [e for e in pool if e["fact_id"] not in prefixFactIds]
    print(f"    {len(evidence)} R4 pool, {alreadyUsed} already used, {len(available)} available")

    if len(available) < newDensity:
        print(f"    WARNING: only {len(available)} available (requested {newDensity})")
        newDensity = len(available)

    newFacts = available[:newDensity]
    newEvTokens = sum(e["est_tokens"] for e in newFacts)
    print(f"    G{generation + 1}: {len(newFacts)} new facts, ~{newEvTokens:,} tok")

    typeCounts = Counter(e.get("question_type", "unknown") for e in newFacts)
    print(f"    Categories: {', '.join(f'{t}={c}' for t, c in sorted(typeCounts.items()))}")

    # Load padding pool — use different seed range to avoid overlap with prefix
    paddingPool = load_padding_pool()

    # Figure out which padding sessions were used in prefix
    # Use seed+2000 offset for new segments to get fresh padding
    padSeed = seed + 2000 * generation
    padSessions = select_padding(paddingPool, segmentTokens, newEvTokens, padSeed)
    padTokens = sum(estimate_tokens_chars(s["chars"]) for s in padSessions)
    segmentEst = newEvTokens + padTokens
    totalEst = prefixTokens + segmentEst
    print(f"    Padding: {len(padSessions)} sessions, ~{padTokens:,} tok")
    print(f"    Segment estimate: ~{segmentEst:,} tok")
    print(f"    Total estimate: ~{totalEst:,} tok ({totalEst / targetTokens * 100:.1f}% of target)")

    if dryRun:
        print("\n  Done (dry-run, no files written).")
        return

    # Build new segment
    print(f"\n  Assembling new segment...")
    segmentMessages, segmentFactMeta = interleave(padSessions, newFacts)

    # Sanitize empty messages
    segmentMessages = [m for m in segmentMessages if m.get("content", "").strip()]

    # Offset fact positions to account for prefix
    prefixMsgCount = len(prefixMessages)
    for fm in segmentFactMeta:
        fm["message_start"] += prefixMsgCount
        fm["message_end"] += prefixMsgCount
        fm["generation"] = generation + 1

    # Tag prefix facts with generation
    for fm in prefixFacts:
        if "generation" not in fm:
            fm["generation"] = fm.get("generation", 1)

    # Combine
    allMessages = prefixMessages + segmentMessages
    allFacts = list(prefixFacts) + segmentFactMeta

    # Recompute position_pct for ALL facts
    totalMsgs = len(allMessages)
    for fm in allFacts:
        fm["position_pct"] = round(fm["message_start"] / totalMsgs * 100, 1)

    actualChars = sum(len(m["content"]) for m in allMessages)
    actualTokEst = estimate_tokens_chars(actualChars)
    print(f"  Combined: {len(allMessages)} messages, ~{actualTokEst:,} tok")
    print(f"    G1–G{generation}: {len(prefixFacts)} facts (from prefix)")
    print(f"    G{generation + 1}: {len(segmentFactMeta)} new facts")

    # Build metadata
    metadata = {
        "version": "v6_nested",
        "run_mode": "R4",
        "density": len(allFacts),
        "target_tokens": targetTokens,
        "seed": seed,
        "n_messages": len(allMessages),
        "total_chars": actualChars,
        "est_tokens": actualTokEst,
        "n_evidence": len(allFacts),
        "question_types": dict(Counter(fm["question_type"] for fm in allFacts)),
        "n_padding_sessions": prefixMeta["n_padding_sessions"] + len(padSessions),
        "padding_tokens": prefixMeta.get("padding_tokens", 0) + padTokens,
        "evidence_tokens": prefixMeta.get("evidence_tokens", 0) + newEvTokens,
        "facts": allFacts,
        "max_generation": generation + 1,
        "generations": {
            **prefixMeta.get("generations", {str(i): 0 for i in range(1, generation + 1)}),
            str(generation + 1): len(segmentFactMeta),
        },
        "prefix_file": str(prefixConvFile),
        "prefix_tokens": prefixTokens,
    }

    # Fill in generation counts for prefix if not already there
    if "generations" not in prefixMeta:
        metadata["generations"]["1"] = len(prefixFacts)

    # Save
    outDir = Path(outputDir) / "v6_R4"
    outDir.mkdir(parents=True, exist_ok=True)

    tokLabel = (f"{targetTokens // 1_000_000}M" if targetTokens >= 1_000_000
                else f"{targetTokens // 1_000}K")
    tag = f"nested_d{len(allFacts)}_{tokLabel}_seed{seed}"
    contextFile = outDir / f"{tag}.json"
    metaFile = outDir / f"{tag}_meta.json"

    print(f"\n  Saving...")
    with open(contextFile, "w", encoding="utf-8") as f:
        json.dump(allMessages, f, ensure_ascii=False)

    with open(metaFile, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    sizeMB = contextFile.stat().st_size / (1024 * 1024)
    print(f"    {contextFile} ({sizeMB:.1f} MB)")
    print(f"    {metaFile}")
    print(f"\n  Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Build long conversation for iterative compaction benchmark (v6)")
    parser.add_argument("--density", type=int, default=80,
                        help="Number of facts to inject (default: 80)")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET,
                        help=f"Target conversation size in tokens (default: {DEFAULT_TARGET:,})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE))
    parser.add_argument("--dry-run", action="store_true")
    # Nested mode
    parser.add_argument("--nested", type=str, default=None,
                        help="Path to prefix conversation JSON (enables nested mode)")
    parser.add_argument("--nested-density", type=int, default=80,
                        help="Number of NEW facts to add in the nested segment (default: 80)")
    args = parser.parse_args()

    if args.nested:
        prefixConv = Path(args.nested)
        prefixMeta = prefixConv.parent / (prefixConv.stem + "_meta.json")
        if not prefixConv.exists():
            print(f"ERROR: prefix file not found: {prefixConv}")
            return
        if not prefixMeta.exists():
            print(f"ERROR: prefix meta not found: {prefixMeta}")
            return
        build_nested(
            prefixConvFile=prefixConv,
            prefixMetaFile=prefixMeta,
            newDensity=args.nested_density,
            targetTokens=args.target_tokens,
            seed=args.seed,
            outputDir=args.output_dir,
            dryRun=args.dry_run,
        )
        return

    print("=" * 70)
    print(f"  BUILD CONVERSATION v6 (iterative compaction)")
    print(f"  Mode: R4 (chopped + filtered)")
    print(f"  Density: d{args.density}")
    print(f"  Target: {args.target_tokens:,} tokens ({args.target_tokens / 1_000_000:.1f}M)")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    paddingPool = load_padding_pool()
    totalPoolTokens = sum(estimate_tokens_chars(s["chars"]) for s in paddingPool)
    print(f"    {len(paddingPool)} padding sessions (~{totalPoolTokens:,} tokens available)")

    if totalPoolTokens < args.target_tokens * 0.9:
        print(f"    WARNING: padding pool ({totalPoolTokens:,} tok) may be insufficient "
              f"for target ({args.target_tokens:,} tok)")

    rawEvidence = load_evidence()
    print(f"    {len(rawEvidence)} raw evidence entries")

    # Prepare R4 evidence
    evidence = prepare_evidence_r4(rawEvidence)
    print(f"    {len(evidence)} R4 evidence entries (filtered + chopped)")

    sizes = [e["est_tokens"] for e in evidence]
    print(f"    Token stats: min={min(sizes)}, median={statistics.median(sizes):.0f}, "
          f"mean={sum(sizes)/len(sizes):.0f}, max={max(sizes)}")

    # Select evidence
    selected = select_evidence(evidence, args.density, args.seed)
    if len(selected) < args.density:
        print(f"    WARNING: only {len(selected)} entries available (requested {args.density})")

    evTokens = sum(e["est_tokens"] for e in selected)
    print(f"\n  Evidence: {len(selected)} facts, ~{evTokens:,} tok")

    typeCounts = Counter(e.get("question_type", "unknown") for e in selected)
    print(f"  Categories: {', '.join(f'{t}={c}' for t, c in sorted(typeCounts.items()))}")

    # Select padding
    padSessions = select_padding(paddingPool, args.target_tokens, evTokens, args.seed)
    padTokens = sum(estimate_tokens_chars(s["chars"]) for s in padSessions)
    totalEst = evTokens + padTokens
    print(f"  Padding: {len(padSessions)} sessions, ~{padTokens:,} tok")
    print(f"  Total estimated: ~{totalEst:,} tok ({totalEst / args.target_tokens * 100:.1f}% of target)")

    if args.dry_run:
        # Estimate compaction cycles
        windowSize = 190_000
        highWm = 0.90
        lowWm = 0.60
        freePerCycle = int(windowSize * (highWm - lowWm))
        nCycles = max(1, (totalEst - windowSize) // freePerCycle)
        print(f"\n  --- Dry-run estimates ---")
        print(f"  Context window: {windowSize:,} tok")
        print(f"  Tokens freed per cycle: ~{freePerCycle:,}")
        print(f"  Estimated compaction cycles: ~{nCycles}")
        print(f"  Estimated compaction API calls (per strategy): ~{nCycles}")
        print(f"  Q&A batches (bs=5): {len(selected) // 5 + (1 if len(selected) % 5 else 0)}")
        print(f"\n  Conversation file size estimate: ~{totalEst * CHARS_PER_TOKEN / 1_000_000:.0f} MB")
        print("  Done (dry-run, no files written).")
        return

    # Build conversation
    print(f"\n  Assembling conversation...")
    messages, factMeta = interleave(padSessions, selected)

    # Sanitize empty messages
    beforeLen = len(messages)
    messages = [m for m in messages if m.get("content", "").strip()]
    if len(messages) < beforeLen:
        print(f"    Sanitized: removed {beforeLen - len(messages)} empty messages")
        # Recompute fact positions after sanitization
        # (positions may have shifted slightly)

    actualChars = sum(len(m["content"]) for m in messages)
    actualTokEst = estimate_tokens_chars(actualChars)
    print(f"  Assembled: {len(messages)} messages, {actualChars:,} chars, ~{actualTokEst:,} tok")

    # Show fact positions
    for fm in factMeta:
        chopStr = (f" [chopped {fm.get('original_n_turns', 0)}->"
                   f"{fm.get('kept_n_turns', 0)}t]") if fm.get("chopped") else ""
        print(f"    {fm['fact_id']} [{fm['question_type']:25s}] "
              f"@{fm['position_pct']:5.1f}%{chopStr}")

    # Build metadata
    metadata = {
        "version": "v6",
        "run_mode": "R4",
        "density": len(selected),
        "target_tokens": args.target_tokens,
        "seed": args.seed,
        "n_messages": len(messages),
        "total_chars": actualChars,
        "est_tokens": actualTokEst,
        "n_evidence": len(selected),
        "question_types": dict(Counter(fm["question_type"] for fm in factMeta)),
        "n_padding_sessions": len(padSessions),
        "padding_tokens": padTokens,
        "evidence_tokens": evTokens,
        "facts": factMeta,
    }

    # Save
    outputDir = Path(args.output_dir) / "v6_R4"
    outputDir.mkdir(parents=True, exist_ok=True)

    tokLabel = f"{args.target_tokens // 1_000_000}M" if args.target_tokens >= 1_000_000 else f"{args.target_tokens // 1_000}K"
    contextFile = outputDir / f"d{len(selected)}_{tokLabel}_seed{args.seed}.json"
    metaFile = outputDir / f"d{len(selected)}_{tokLabel}_seed{args.seed}_meta.json"

    print(f"\n  Saving...")
    with open(contextFile, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)

    with open(metaFile, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    sizeMB = contextFile.stat().st_size / (1024 * 1024)
    print(f"    {contextFile} ({sizeMB:.1f} MB)")
    print(f"    {metaFile}")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()

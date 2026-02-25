#!python3
"""
Step 4 v5: Assemble contexts for density sweep (2x2 factorial design).

Run modes (2x2: evidence format x category filter):
  R1 — full evidence, all categories        (baseline)
  R2 — chopped evidence, all categories     (truncation effect)
  R3 — full evidence, filtered categories   (filtering effect)
  R4 — chopped evidence, filtered categories (both effects, high density)

Output:
  data/contexts/v5_R{1,2,3,4}/d{N}_seed{S}.json + _meta.json

Usage:
    python build_contexts_v5.py --run R1 --dry-run
    python build_contexts_v5.py --run all --dry-run
    python build_contexts_v5.py --run R4 --densities 4,8,12,16,19,25,30,40,60,80,120,150,180
    python build_contexts_v5.py --run R1 --densities 4 --calibrate
"""

import json
import random
import argparse
import statistics
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Force unbuffered output
_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# PATHS
# ============================================================================

PADDING_POOL = Path("data/padding_pool.jsonl")
EVIDENCE_LM = Path("data/evidence_longmemeval.json")
OUTPUT_BASE = Path("data/contexts")

DEFAULT_TARGET = 190_000
CHARS_PER_TOKEN = 4.3  # calibrated on LongMemEval content

# Categories to exclude in R3
FILTERED_OUT_TYPES = {"temporal-reasoning", "multi-session"}

# Default densities per run (intermediate d12,d16,d25 for finer granularity)
DEFAULT_DENSITIES = {
    "R1": [4, 8, 12, 16, 19],
    "R2": [4, 8, 12, 16, 19, 25, 30, 40, 60, 80],
    "R3": [4, 8, 12, 16, 19, 25, 30],
    "R4": [4, 8, 12, 16, 19, 25, 30, 40, 60, 80, 120, 150],
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_padding_pool():
    """Load padding sessions from JSONL."""
    sessions = []
    with open(PADDING_POOL, encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line))
    return sessions


def load_evidence():
    """Load all evidence entries from LongMemEval."""
    with open(EVIDENCE_LM, encoding="utf-8") as f:
        evidence = json.load(f)
    return evidence


# ============================================================================
# EVIDENCE PROCESSING
# ============================================================================

def chop_evidence(entry):
    """
    Truncate evidence to has_answer turns + 1 turn of context around each.

    Preserves the turns that actually contain the answer, plus neighboring
    turns for conversational context. Much more compact than full evidence
    while keeping the relevant information.

    Returns a new entry dict with truncated evidence_turns and updated stats.
    """
    turns = entry["evidence_turns"]
    indices = entry.get("has_answer_indices", [])

    if not indices:
        # Fallback: keep first 3 turns
        keptTurns = turns[:3]
    else:
        # Expand each answer index by ±1 for context
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
    """Filter out evidence entries by question_type."""
    return [e for e in evidence if e.get("question_type") not in excludeTypes]


def prepare_evidence(evidence, runMode):
    """
    Prepare evidence pool for a given run mode.

    Returns (processedEvidence, modeInfo).
    """
    if runMode == "R1":
        # Full evidence, all categories
        return evidence, {
            "mode": "R1",
            "description": "full evidence, all categories",
            "chopped": False,
            "filtered": False,
            "n_available": len(evidence),
        }

    elif runMode == "R2":
        # Chopped evidence, all categories
        chopped = [chop_evidence(e) for e in evidence]
        return chopped, {
            "mode": "R2",
            "description": "chopped evidence (has_answer + context), all categories",
            "chopped": True,
            "filtered": False,
            "n_available": len(chopped),
        }

    elif runMode == "R3":
        # Full evidence, filtered categories
        filtered = filter_evidence(evidence, FILTERED_OUT_TYPES)
        return filtered, {
            "mode": "R3",
            "description": f"full evidence, filtered (excl {FILTERED_OUT_TYPES})",
            "chopped": False,
            "filtered": True,
            "excluded_types": sorted(FILTERED_OUT_TYPES),
            "n_available": len(filtered),
            "n_excluded": len(evidence) - len(filtered),
        }

    elif runMode == "R4":
        # Chopped evidence, filtered categories
        filtered = filter_evidence(evidence, FILTERED_OUT_TYPES)
        chopped = [chop_evidence(e) for e in filtered]
        return chopped, {
            "mode": "R4",
            "description": f"chopped evidence, filtered (excl {FILTERED_OUT_TYPES})",
            "chopped": True,
            "filtered": True,
            "excluded_types": sorted(FILTERED_OUT_TYPES),
            "n_available": len(chopped),
            "n_excluded": len(evidence) - len(filtered),
        }

    else:
        raise ValueError(f"Unknown run mode: {runMode}")


# ============================================================================
# TOKEN ESTIMATION
# ============================================================================

def estimate_tokens_chars(chars):
    return int(chars / CHARS_PER_TOKEN)


def estimate_tokens_turns(turns):
    chars = sum(len(t["content"]) for t in turns)
    return estimate_tokens_chars(chars)


# ============================================================================
# CONTEXT ASSEMBLY
# ============================================================================

def select_evidence(evidence, density, seed):
    """Select N evidence entries deterministically."""
    rng = random.Random(seed)
    pool = sorted(evidence, key=lambda e: e["fact_id"])
    rng.shuffle(pool)
    selected = pool[:density]
    selected.sort(key=lambda e: e["fact_id"])
    return selected


def select_padding(paddingPool, targetTokens, evidenceTokens, seed):
    """Select padding sessions to fill remaining token budget."""
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


def interleave(paddingSessions, evidenceEntries):
    """
    Interleave evidence into padding at uniformly spaced positions.

    Returns (messages, factMeta).
    """
    nEvidence = len(evidenceEntries)
    nPadding = len(paddingSessions)

    if nEvidence == 0:
        messages = []
        for sess in paddingSessions:
            messages.extend(sess["turns"])
        return messages, []

    # Split padding into (nEvidence + 1) chunks
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

    # Compute position percentages with final total
    totalMsgs = len(messages)
    for fm in factMeta:
        fm["position_pct"] = round(fm["message_start"] / totalMsgs * 100, 1)

    return messages, factMeta


def build_context(density, targetTokens, seed, paddingPool, evidence, dryRun=False):
    """Build a single context."""
    print(f"\n  --- d{density}, target={targetTokens:,} tok ---")

    selected = select_evidence(evidence, density, seed)
    if len(selected) < density:
        print(f"    WARNING: only {len(selected)} evidence entries (requested {density})")
        density = len(selected)

    evTokens = sum(e["est_tokens"] for e in selected)
    print(f"    Evidence: {len(selected)} entries, ~{evTokens:,} tok")

    # Category breakdown
    from collections import Counter
    typeCounts = Counter(e.get("question_type", "unknown") for e in selected)
    typeStr = ", ".join(f"{t}={c}" for t, c in sorted(typeCounts.items()))
    print(f"    Categories: {typeStr}")

    padSessions = select_padding(paddingPool, targetTokens, evTokens, seed)
    padTokens = sum(estimate_tokens_chars(s["chars"]) for s in padSessions)
    print(f"    Padding: {len(padSessions)} sessions, ~{padTokens:,} tok")
    print(f"    Total estimated: ~{evTokens + padTokens:,} tok")

    if dryRun:
        return None, None

    messages, factMeta = interleave(padSessions, selected)
    actualChars = sum(len(m["content"]) for m in messages)
    actualTokEst = int(actualChars / CHARS_PER_TOKEN)
    print(f"    Assembled: {len(messages)} msgs, {actualChars:,} chars, ~{actualTokEst:,} tok")

    for fm in factMeta:
        chopStr = f" [chopped {fm.get('original_n_turns',0)}->{fm.get('kept_n_turns',0)}t]" if fm.get("chopped") else ""
        print(f"      {fm['fact_id']} [{fm['question_type']:25s}] "
              f"@{fm['position_pct']:5.1f}%{chopStr}")

    # Build metadata
    sources = Counter(fm["source"] for fm in factMeta)
    questionTypes = Counter(fm["question_type"] for fm in factMeta)

    metadata = {
        "density": density,
        "target_tokens": targetTokens,
        "seed": seed,
        "n_messages": len(messages),
        "total_chars": actualChars,
        "est_tokens": actualTokEst,
        "n_evidence": len(selected),
        "evidence_sources": dict(sources),
        "question_types": dict(questionTypes),
        "n_padding_sessions": len(padSessions),
        "padding_tokens": padTokens,
        "evidence_tokens": evTokens,
        "facts": factMeta,
    }

    return messages, metadata


# ============================================================================
# CALIBRATION (count_tokens API — free, exact)
# ============================================================================

def calibrate_with_api(messages, metadata, paddingPool, targetTokens, seed,
                       model="claude-haiku-4-5-20251001", tolerance=5000):
    """Adjust context size using count_tokens API."""
    import anthropic
    client = anthropic.Anthropic()

    system = ("You are a helpful assistant working on a complex software project. "
              "Answer questions precisely from memory.")

    realTokens = client.messages.count_tokens(
        model=model, messages=messages, system=system
    ).input_tokens

    totalChars = sum(len(m["content"]) for m in messages)
    measuredRatio = totalChars / realTokens if realTokens > 0 else CHARS_PER_TOKEN

    print(f"    Calibration: {realTokens:,} real tok "
          f"(target: {targetTokens:,}, ratio: {measuredRatio:.2f} chars/tok)")

    delta = realTokens - targetTokens

    if abs(delta) <= tolerance:
        print(f"    Within tolerance ({tolerance:,}), OK")
        metadata["real_tokens"] = realTokens
        metadata["chars_per_token"] = round(measuredRatio, 3)
        return messages, metadata

    # Build set of evidence message indices
    evIndices = set()
    for fm in metadata["facts"]:
        for i in range(fm["message_start"], fm["message_end"]):
            evIndices.add(i)

    if delta > 0:
        charsToRemove = int(delta * measuredRatio)
        print(f"    Over by {delta:,} tok, trimming ~{charsToRemove:,} chars...")
        removedChars = 0
        while removedChars < charsToRemove and len(messages) > 10:
            lastIdx = len(messages) - 1
            if lastIdx in evIndices:
                break
            removedChars += len(messages[-1]["content"])
            messages.pop()
    else:
        charsToAdd = int(abs(delta) * measuredRatio)
        print(f"    Under by {abs(delta):,} tok, padding ~{charsToAdd:,} chars...")
        rng = random.Random(seed + 2000)
        pool = list(paddingPool)
        rng.shuffle(pool)
        usedContents = {m["content"][:100] for m in messages}
        addedChars = 0
        for sess in pool:
            if addedChars >= charsToAdd:
                break
            firstContent = sess["turns"][0]["content"][:100] if sess["turns"] else ""
            if firstContent in usedContents:
                continue
            messages.extend(sess["turns"])
            addedChars += sess["chars"]
            metadata["n_padding_sessions"] += 1

    realTokens2 = client.messages.count_tokens(
        model=model, messages=messages, system=system
    ).input_tokens
    totalChars2 = sum(len(m["content"]) for m in messages)

    print(f"    After calibration: {realTokens2:,} real tok "
          f"(delta: {realTokens2 - targetTokens:+,})")

    metadata["real_tokens"] = realTokens2
    metadata["n_messages"] = len(messages)
    metadata["total_chars"] = totalChars2
    metadata["est_tokens"] = int(totalChars2 / CHARS_PER_TOKEN)
    metadata["chars_per_token"] = round(totalChars2 / realTokens2, 3)

    return messages, metadata


# ============================================================================
# SAVE
# ============================================================================

def save_context(messages, metadata, runMode, density, seed, outputBase):
    """Save context + metadata."""
    outputDir = Path(outputBase) / f"v5_{runMode}"
    outputDir.mkdir(parents=True, exist_ok=True)

    contextFile = outputDir / f"d{density}_seed{seed}.json"
    metaFile = outputDir / f"d{density}_seed{seed}_meta.json"

    with open(contextFile, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)

    with open(metaFile, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    sizeMB = contextFile.stat().st_size / (1024 * 1024)
    print(f"    Saved: {contextFile.name} ({sizeMB:.1f} MB) + {metaFile.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build v5 recall contexts (3 run modes)")
    parser.add_argument("--run", type=str, required=True,
                        choices=["R1", "R2", "R3", "R4", "all"],
                        help="Run mode: R1 (full/all), R2 (chopped/all), R3 (full/filtered), R4 (chopped/filtered), all")
    parser.add_argument("--densities", type=str, default=None,
                        help="Comma-separated densities (default: per-run standard set)")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET,
                        help=f"Target context size (default: {DEFAULT_TARGET})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="Use count_tokens API for exact sizing (free)")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    runs = ["R1", "R2", "R3", "R4"] if args.run == "all" else [args.run]

    print("=" * 70)
    print(f"  BUILD CONTEXTS v5")
    print(f"  Runs: {runs}")
    print(f"  Target: {args.target_tokens:,} tokens")
    print(f"  Seed: {args.seed}")
    print(f"  Calibrate: {args.calibrate}")
    print("=" * 70)

    # Load raw data once
    print("\n  Loading data...")
    paddingPool = load_padding_pool()
    print(f"    {len(paddingPool)} padding sessions")

    rawEvidence = load_evidence()
    print(f"    {len(rawEvidence)} evidence entries")

    # Show category distribution
    from collections import Counter
    typeCounts = Counter(e.get("question_type", "unknown") for e in rawEvidence)
    print(f"    Categories: {dict(sorted(typeCounts.items()))}")

    for runMode in runs:
        densities = ([int(d) for d in args.densities.split(",")]
                     if args.densities else DEFAULT_DENSITIES[runMode])

        print(f"\n{'#' * 70}")
        print(f"  RUN {runMode}")
        print(f"  Densities: {densities}")
        print(f"{'#' * 70}")

        # Prepare evidence for this mode
        evidence, modeInfo = prepare_evidence(rawEvidence, runMode)
        print(f"  Mode: {modeInfo['description']}")
        print(f"  Available evidence: {modeInfo['n_available']}")

        if modeInfo.get("chopped"):
            sizes = [e["est_tokens"] for e in evidence]
            print(f"  Chopped token stats: min={min(sizes)}, "
                  f"median={statistics.median(sizes):.0f}, "
                  f"mean={sum(sizes)/len(sizes):.0f}, max={max(sizes)}")

        # Check max density
        sortedSizes = sorted(e["est_tokens"] for e in evidence)
        cumul = 0
        maxDensity = 0
        for i, t in enumerate(sortedSizes):
            cumul += t
            if cumul > args.target_tokens:
                maxDensity = i
                break
        else:
            maxDensity = len(sortedSizes)
        print(f"  Max density at {args.target_tokens:,} tok: d{maxDensity}")

        # Check requested densities
        for d in densities:
            if d > modeInfo["n_available"]:
                print(f"  WARNING: d{d} > available evidence ({modeInfo['n_available']}), will be capped")

        # Build contexts
        for density in densities:
            messages, metadata = build_context(
                density, args.target_tokens, args.seed,
                paddingPool, evidence, dryRun=args.dry_run
            )
            if messages is not None:
                # Add run mode info to metadata
                metadata["run_mode"] = modeInfo

                if args.calibrate:
                    messages, metadata = calibrate_with_api(
                        messages, metadata, paddingPool,
                        args.target_tokens, args.seed, args.model
                    )

                save_context(messages, metadata, runMode, density, args.seed,
                             args.output_dir)

    print(f"\n  Done!")


if __name__ == "__main__":
    main()

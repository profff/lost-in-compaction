#!python3
"""
Step 4: Assemble contexts from padding pool + evidence sessions.

Builds deterministic contexts for recall benchmarks:
- Pick N evidence sessions (density)
- Fill with padding sessions to reach target token count
- Interleave evidence at uniformly spaced positions
- Save context + metadata

Usage:
    python build_contexts.py --density 8 --target-tokens 190000 --seed 42
    python build_contexts.py --density 4,8,19 --target-tokens 190000
    python build_contexts.py --all-densities  (d4, d8, d19, d22 = max LM at 190K)
    python build_contexts.py --dry-run --density 8
"""

import json
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PADDING_POOL = Path("data/padding_pool.jsonl")
EVIDENCE_LM = Path("data/evidence_longmemeval.json")
EVIDENCE_SY = Path("data/evidence_synthetic.json")
OUTPUT_DIR = Path("data/contexts/recall_190K")

DEFAULT_TARGET = 190_000
# LongMemEval content: ~4.3 chars/token (measured via count_tokens API).
# Our old synthetic conversations were ~3.1, but natural English is denser.
CHARS_PER_TOKEN = 4.3


def load_padding_pool():
    """Load padding sessions from JSONL."""
    sessions = []
    with open(PADDING_POOL, encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line))
    return sessions


def load_evidence():
    """Load evidence entries (LongMemEval + synthetic if available)."""
    with open(EVIDENCE_LM, encoding="utf-8") as f:
        evidence = json.load(f)

    if EVIDENCE_SY.exists():
        with open(EVIDENCE_SY, encoding="utf-8") as f:
            synthetic = json.load(f)
        evidence.extend(synthetic)
        print(f"  Evidence: {len(evidence)} total "
              f"({len(evidence) - len(synthetic)} LM + {len(synthetic)} SY)")
    else:
        print(f"  Evidence: {len(evidence)} LongMemEval (no synthetic yet)")

    return evidence


def estimate_tokens_chars(chars):
    """Estimate token count from character count using calibrated ratio."""
    return int(chars / CHARS_PER_TOKEN)


def estimate_tokens_turns(turns):
    """Estimate token count for a list of turns."""
    chars = sum(len(t["content"]) for t in turns)
    return estimate_tokens_chars(chars)


def select_evidence(evidence, density, seed):
    """Select N evidence entries for the given density."""
    rng = random.Random(seed)

    # Sort by fact_id for determinism, then shuffle with seed
    pool = sorted(evidence, key=lambda e: e["fact_id"])
    rng.shuffle(pool)

    # Pick the first `density` entries
    selected = pool[:density]

    # Sort by fact_id again for consistent ordering in context
    selected.sort(key=lambda e: e["fact_id"])

    return selected


def select_padding(paddingPool, targetTokens, evidenceTokens, seed, excludeIds=None):
    """Select padding sessions to fill the remaining token budget."""
    rng = random.Random(seed + 1000)  # different seed stream than evidence
    budget = targetTokens - evidenceTokens

    if budget <= 0:
        print(f"  WARNING: evidence alone ({evidenceTokens:,} tok) exceeds target ({targetTokens:,})")
        return []

    # Shuffle pool
    pool = list(paddingPool)
    rng.shuffle(pool)

    selected = []
    usedTokens = 0
    for sess in pool:
        if excludeIds and sess["id"] in excludeIds:
            continue
        # Recompute tokens using calibrated ratio (data files use chars//3)
        sessTok = estimate_tokens_chars(sess["chars"])
        if usedTokens + sessTok > budget:
            # Check if we're close enough (within 5%)
            if usedTokens >= budget * 0.95:
                break
            # Try to find a smaller session
            continue
        selected.append(sess)
        usedTokens += sessTok
        if usedTokens >= budget * 0.98:
            break

    return selected


def interleave(paddingSessions, evidenceEntries):
    """
    Interleave evidence sessions into padding at uniform positions.

    Returns flat list of messages + fact position metadata.
    """
    nEvidence = len(evidenceEntries)
    nPadding = len(paddingSessions)

    if nEvidence == 0:
        # No evidence, just concatenate padding
        messages = []
        for sess in paddingSessions:
            messages.extend(sess["turns"])
        return messages, []

    # Split padding into (nEvidence + 1) roughly equal chunks
    chunkSize = nPadding // (nEvidence + 1)
    remainder = nPadding % (nEvidence + 1)

    chunks = []
    idx = 0
    for i in range(nEvidence + 1):
        end = idx + chunkSize + (1 if i < remainder else 0)
        chunks.append(paddingSessions[idx:end])
        idx = end

    # Interleave: chunk[0] + evidence[0] + chunk[1] + evidence[1] + ...
    messages = []
    factMeta = []

    for i in range(nEvidence + 1):
        # Add padding chunk
        for sess in chunks[i]:
            messages.extend(sess["turns"])

        # Add evidence (if not past the last one)
        if i < nEvidence:
            ev = evidenceEntries[i]
            startIdx = len(messages)
            messages.extend(ev["evidence_turns"])
            endIdx = len(messages)

            factMeta.append({
                "fact_id": ev["fact_id"],
                "source": ev["source"],
                "question": ev["question"],
                "answer": ev["answer"],
                "keywords": ev["keywords"],
                "message_start": startIdx,
                "message_end": endIdx,
                "position_pct": startIdx / max(len(messages), 1) * 100,
                "n_turns": len(ev["evidence_turns"]),
                "est_tokens": ev["est_tokens"],
            })

    # Update position_pct now that we know total length
    totalMsgs = len(messages)
    for fm in factMeta:
        fm["position_pct"] = round(fm["message_start"] / totalMsgs * 100, 1)

    return messages, factMeta


def build_context(density, targetTokens, seed, paddingPool, evidence, dryRun=False):
    """Build a single context for the given density."""
    print(f"\n  --- d{density}, seed={seed}, target={targetTokens:,} tokens ---")

    # Select evidence
    selected = select_evidence(evidence, density, seed)
    if len(selected) < density:
        print(f"  WARNING: only {len(selected)} evidence entries available (requested {density})")
        density = len(selected)

    # Recompute tokens using calibrated ratio (data files use chars//3)
    evTokens = sum(estimate_tokens_chars(e["chars"]) for e in selected)
    print(f"  Evidence: {len(selected)} entries, ~{evTokens:,} tokens (at {CHARS_PER_TOKEN} chars/tok)")

    # Select padding
    padSessions = select_padding(paddingPool, targetTokens, evTokens, seed)
    padTokens = sum(estimate_tokens_chars(s["chars"]) for s in padSessions)
    print(f"  Padding: {len(padSessions)} sessions, ~{padTokens:,} tokens")
    print(f"  Total estimated: ~{evTokens + padTokens:,} tokens")

    if dryRun:
        print(f"  [DRY RUN] Would build context with {len(selected)} evidence + {len(padSessions)} padding")
        return None, None, None

    # Interleave
    messages, factMeta = interleave(padSessions, selected)
    actualChars = sum(len(m["content"]) for m in messages)
    actualTokEst = int(actualChars / CHARS_PER_TOKEN)
    print(f"  Assembled: {len(messages)} messages, {actualChars:,} chars, ~{actualTokEst:,} tokens")

    # Show fact positions
    for fm in factMeta:
        print(f"    {fm['fact_id']} @ msg {fm['message_start']} ({fm['position_pct']:.0f}%)")

    # Build metadata
    sources = {}
    for fm in factMeta:
        src = fm["source"]
        sources[src] = sources.get(src, 0) + 1

    metadata = {
        "density": density,
        "target_tokens": targetTokens,
        "seed": seed,
        "n_messages": len(messages),
        "total_chars": actualChars,
        "est_tokens": actualTokEst,
        "n_evidence": len(selected),
        "evidence_sources": sources,
        "n_padding_sessions": len(padSessions),
        "facts": factMeta,
    }

    return messages, metadata, factMeta


def calibrate_with_api(messages, metadata, paddingPool, targetTokens, seed,
                       model="claude-haiku-4-5-20251001", tolerance=5000):
    """
    Calibrate context size using Anthropic count_tokens API (free, exact).

    Adds or removes padding sessions to hit targetTokens ± tolerance.
    Returns (messages, metadata) with updated token counts.
    """
    import anthropic
    client = anthropic.Anthropic()

    system = ("You are a helpful assistant working on a complex software project. "
              "Answer questions precisely from memory.")

    realTokens = client.messages.count_tokens(
        model=model, messages=messages, system=system
    ).input_tokens

    totalChars = sum(len(m["content"]) for m in messages)
    measuredRatio = totalChars / realTokens if realTokens > 0 else CHARS_PER_TOKEN

    print(f"  Calibration: {realTokens:,} real tokens "
          f"(target: {targetTokens:,}, measured ratio: {measuredRatio:.2f} chars/tok)")

    delta = realTokens - targetTokens

    if abs(delta) <= tolerance:
        print(f"  Within tolerance ({tolerance:,}), no adjustment needed")
        metadata["real_tokens"] = realTokens
        metadata["chars_per_token"] = round(measuredRatio, 3)
        return messages, metadata

    if delta > 0:
        # Over target — remove padding from the end
        charsToRemove = int(delta * measuredRatio)
        print(f"  Over by {delta:,} tokens, removing ~{charsToRemove:,} chars of padding...")

        # Find and remove non-evidence messages from the end
        factStarts = {fm["message_start"] for fm in metadata["facts"]}
        factEnds = {fm["message_end"] for fm in metadata["facts"]}
        # Build set of evidence message indices
        evIndices = set()
        for fm in metadata["facts"]:
            for i in range(fm["message_start"], fm["message_end"]):
                evIndices.add(i)

        removedChars = 0
        while removedChars < charsToRemove and len(messages) > 10:
            lastIdx = len(messages) - 1
            if lastIdx in evIndices:
                break  # Don't remove evidence
            removedChars += len(messages[-1]["content"])
            messages.pop()

    else:
        # Under target — add more padding
        charsToAdd = int(abs(delta) * measuredRatio)
        print(f"  Under by {abs(delta):,} tokens, adding ~{charsToAdd:,} chars of padding...")

        rng = random.Random(seed + 2000)
        pool = list(paddingPool)
        rng.shuffle(pool)

        # Find sessions not already used (by content hash)
        usedContents = set()
        for m in messages:
            usedContents.add(m["content"][:100])  # crude dedup

        addedChars = 0
        for sess in pool:
            if addedChars >= charsToAdd:
                break
            firstContent = sess["turns"][0]["content"][:100] if sess["turns"] else ""
            if firstContent in usedContents:
                continue
            # Append at end (before last evidence if possible)
            messages.extend(sess["turns"])
            addedChars += sess["chars"]
            metadata["n_padding_sessions"] += 1

    # Re-measure
    realTokens2 = client.messages.count_tokens(
        model=model, messages=messages, system=system
    ).input_tokens
    totalChars2 = sum(len(m["content"]) for m in messages)

    print(f"  After calibration: {realTokens2:,} real tokens "
          f"(delta: {realTokens2 - targetTokens:+,})")

    metadata["real_tokens"] = realTokens2
    metadata["n_messages"] = len(messages)
    metadata["total_chars"] = totalChars2
    metadata["est_tokens"] = int(totalChars2 / CHARS_PER_TOKEN)
    metadata["chars_per_token"] = round(totalChars2 / realTokens2, 3)

    # Update fact positions (if messages were added/removed, positions may shift)
    # Note: for trim from end, evidence positions don't change
    # For additions at end, evidence positions don't change either

    return messages, metadata


def save_context(messages, metadata, density, seed, outputDir):
    """Save context and metadata files."""
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    contextFile = outputDir / f"d{density}_seed{seed}.json"
    metaFile = outputDir / f"d{density}_seed{seed}_meta.json"

    with open(contextFile, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)

    with open(metaFile, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    sizeMB = contextFile.stat().st_size / (1024 * 1024)
    print(f"  Saved: {contextFile} ({sizeMB:.1f} MB)")
    print(f"  Saved: {metaFile}")


def main():
    parser = argparse.ArgumentParser(description="Build recall benchmark contexts")
    parser.add_argument("--density", type=str, default=None,
                        help="Comma-separated densities (e.g. 4,8,19)")
    parser.add_argument("--all-densities", action="store_true",
                        help="Build for all standard densities")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET,
                        help=f"Target context size in tokens (default: {DEFAULT_TARGET})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="Use count_tokens API to hit exact token target (free)")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="Model for count_tokens calibration")
    args = parser.parse_args()

    # Determine densities
    if args.all_densities:
        densities = [4, 8, 19]  # safe LongMemEval-only densities at 190K
    elif args.density:
        densities = [int(d) for d in args.density.split(",")]
    else:
        parser.error("Specify --density or --all-densities")

    print("=" * 70)
    print(f"  BUILD CONTEXTS")
    print(f"  Densities: {densities}")
    print(f"  Target: {args.target_tokens:,} tokens")
    print(f"  Chars/token ratio: {CHARS_PER_TOKEN}")
    print(f"  Seed: {args.seed}")
    print(f"  Calibrate: {args.calibrate}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # Load data
    print("\n  Loading padding pool...")
    paddingPool = load_padding_pool()
    print(f"  {len(paddingPool)} padding sessions loaded")

    print("  Loading evidence...")
    evidence = load_evidence()

    # Build contexts
    for density in densities:
        messages, metadata, factMeta = build_context(
            density, args.target_tokens, args.seed,
            paddingPool, evidence, dryRun=args.dry_run
        )
        if messages is not None:
            if args.calibrate:
                messages, metadata = calibrate_with_api(
                    messages, metadata, paddingPool,
                    args.target_tokens, args.seed, args.model
                )
            save_context(messages, metadata, density, args.seed, args.output_dir)

    print(f"\n  Done!")


if __name__ == "__main__":
    main()

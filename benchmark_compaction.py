#!python3
"""
Benchmark: Incremental vs Brutal compaction.

"Needle in a Compacted Haystack"

Measures fact recall after multiple compaction cycles,
comparing dual-watermark incremental compaction vs single-shot brutal summary.

Usage:
    ./benchmark_compaction.py [--dry-run] [--cycles N] [--max-tokens N]
"""

import sys
import os
import json
import time
import argparse
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# Add francine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "francine"))

from compaction import ContextCompactor, estimate_tokens, messages_to_text, COMPACT_SYSTEM, COMPACT_PROMPT


# ============================================================================
# PLANTED FACTS — each has a unique ID and expected answer
# These are the "needles" we'll look for after compaction
# ============================================================================

PLANTED_FACTS = [
    # (fact_id, message_content, question, expected_keywords)
    ("F01", "The project uses PostgreSQL 15.3 on port 5433, not the default 5432.",
     "What PostgreSQL version and port does the project use?",
     ["15.3", "5433"]),

    ("F02", "The deployment server is at 10.0.42.17, hostname 'kraken', SSH on port 2222.",
     "What is the deployment server IP and SSH port?",
     ["10.0.42.17", "2222"]),

    ("F03", "We decided to use Redis for session storage instead of JWT tokens.",
     "What technology was chosen for session storage?",
     ["redis"]),

    ("F04", "The CI pipeline runs on GitHub Actions, with a self-hosted runner named 'buildbox-arm64'.",
     "What CI system is used and what's the runner name?",
     ["github actions", "buildbox-arm64"]),

    ("F05", "Max upload file size is 25MB, configured in nginx.conf AND app settings.",
     "What is the maximum upload file size?",
     ["25"]),

    ("F06", "The API rate limit is 100 requests per minute per user, with a burst of 20.",
     "What is the API rate limit?",
     ["100", "minute"]),

    ("F07", "The user's name is Olivier and he prefers PascalCase for globals, camelCase for locals.",
     "What naming convention does the user prefer?",
     ["pascalcase", "camelcase"]),

    ("F08", "The codebase has a critical bug in src/auth/oauth.py line 247: token refresh race condition.",
     "Where is the critical bug located?",
     ["oauth.py", "247", "race condition"]),

    ("F09", "Database migrations use Alembic, and the current head revision is a3f7c2b.",
     "What migration tool is used and what's the current revision?",
     ["alembic", "a3f7c2b"]),

    ("F10", "The project uses Python 3.12.1 in a venv at /opt/app/.venv, NOT conda.",
     "What Python version and virtual environment path?",
     ["3.12", "/opt/app/.venv"]),

    ("F11", "Log rotation is set to 7 days, max 500MB per file, stored in /var/log/app/.",
     "What are the log rotation settings?",
     ["7 days", "500"]),

    ("F12", "The frontend uses Vue 3.4 with Pinia for state management, NOT Vuex.",
     "What frontend framework and state management library?",
     ["vue", "pinia"]),

    ("F13", "CORS is configured to allow only app.example.com and staging.example.com.",
     "What domains are allowed in CORS?",
     ["app.example.com", "staging.example.com"]),

    ("F14", "The backup runs daily at 03:00 UTC via a cron job on the kraken server.",
     "When does the backup run?",
     ["03:00", "daily"]),

    ("F15", "The websocket endpoint is /ws/notifications, using socket.io protocol v4.",
     "What is the websocket endpoint and protocol version?",
     ["/ws/notifications", "socket.io", "v4"]),
]


# ============================================================================
# PADDING GENERATORS — realistic filler to inflate context
# ============================================================================

def make_file_read_padding(filename: str, lines: int = 80) -> list[dict]:
    """Simulate a file read exchange (user asks, assistant reads, shows content)."""
    fake_code = "\n".join(
        f"    line_{i}: x = process(data[{i}])  # computation step {i}"
        for i in range(lines)
    )
    return [
        {"role": "user", "content": f"Can you read {filename} for me?"},
        {"role": "assistant", "content": f"Here's the content of `{filename}` ({lines} lines):\n\n```\n{fake_code}\n```\n\nThe file contains {lines} lines of processing logic. Let me know if you want me to analyze any specific part."},
    ]


def make_tool_chain_padding(task: str) -> list[dict]:
    """Simulate a tool_use/tool_result exchange (realistic padding)."""
    return [
        {"role": "user", "content": f"Can you {task}?"},
        {"role": "assistant", "content": f"I'll {task} for you. Let me check the current state first.\n\nI've looked at the relevant files and here's what I found:\n- The current implementation handles the basic case\n- There are a few edge cases to consider\n- The tests cover the main path but not error scenarios\n\nI've made the necessary changes. The modification touches 3 files and adds proper error handling. Would you like me to run the tests?"},
    ]


def make_discussion_padding(topic: str) -> list[dict]:
    """Simulate a back-and-forth discussion."""
    return [
        {"role": "user", "content": f"What do you think about {topic}? I'm considering several approaches."},
        {"role": "assistant", "content": f"Regarding {topic}, there are several considerations:\n\n1. **Performance**: The current approach has O(n²) complexity which could be an issue at scale. A hash-based approach would bring it down to O(n).\n\n2. **Maintainability**: The simpler approach is easier to understand and debug, even if slightly slower.\n\n3. **Testing**: Whatever we choose, we should add comprehensive tests.\n\nI'd recommend starting with the simpler approach and optimizing only if profiling shows it's a bottleneck. Premature optimization and all that.\n\nWhat's your current scale? That would help narrow down the recommendation."},
        {"role": "user", "content": f"Good point. Let's go with the simple approach for now. We handle maybe 10K records max."},
        {"role": "assistant", "content": f"10K records is well within the comfortable range for the simpler O(n²) approach — we're talking milliseconds, not seconds. Good call. I'll implement it the straightforward way."},
    ]


# ============================================================================
# CONVERSATION BUILDER — assemble facts + padding into a realistic convo
# ============================================================================

def build_scripted_conversation() -> list[dict]:
    """
    Build a realistic conversation with planted facts interspersed with padding.

    Structure:
    - Facts are spread across the conversation (early, mid, late)
    - Padding simulates real work: file reads, tool chains, discussions
    - Total size should be enough to trigger multiple compaction cycles
    """
    messages = []

    # --- Opening ---
    messages.extend([
        {"role": "user", "content": "Hey, I need help setting up the deployment pipeline for our project. Let me give you some context first."},
        {"role": "assistant", "content": "Sure! I'm ready to help. Go ahead and share the project details."},
    ])

    # --- Early facts (F01-F05) with padding ---
    for i, (fid, fact, _, _) in enumerate(PLANTED_FACTS[:5]):
        messages.extend([
            {"role": "user", "content": fact},
            {"role": "assistant", "content": f"Got it, noted. {_generate_ack(fid, fact)}"},
        ])
        # Add padding between facts
        if i % 2 == 0:
            messages.extend(make_file_read_padding(f"src/module_{i}.py", lines=60))
        else:
            messages.extend(make_tool_chain_padding(f"check the {['config', 'tests', 'docs'][i % 3]}"))

    # --- Heavy padding block (simulates deep work session) ---
    messages.extend(make_file_read_padding("src/core/engine.py", lines=120))
    messages.extend(make_discussion_padding("the database schema design"))
    messages.extend(make_file_read_padding("config/production.yaml", lines=80))
    messages.extend(make_tool_chain_padding("refactor the authentication module"))
    messages.extend(make_discussion_padding("error handling strategy"))
    messages.extend(make_file_read_padding("src/api/routes.py", lines=100))

    # --- Mid facts (F06-F10) with padding ---
    for i, (fid, fact, _, _) in enumerate(PLANTED_FACTS[5:10]):
        messages.extend([
            {"role": "user", "content": fact},
            {"role": "assistant", "content": f"Understood. {_generate_ack(fid, fact)}"},
        ])
        if i % 2 == 0:
            messages.extend(make_discussion_padding(f"the {['caching', 'monitoring', 'logging'][i % 3]} approach"))
        else:
            messages.extend(make_file_read_padding(f"tests/test_module_{i}.py", lines=70))

    # --- More heavy padding ---
    messages.extend(make_file_read_padding("docker/Dockerfile", lines=50))
    messages.extend(make_tool_chain_padding("set up the CI pipeline"))
    messages.extend(make_file_read_padding("src/services/notification.py", lines=90))
    messages.extend(make_discussion_padding("microservices vs monolith"))

    # --- Late facts (F11-F15) with padding ---
    for i, (fid, fact, _, _) in enumerate(PLANTED_FACTS[10:15]):
        messages.extend([
            {"role": "user", "content": fact},
            {"role": "assistant", "content": f"Noted. {_generate_ack(fid, fact)}"},
        ])
        if i % 2 == 0:
            messages.extend(make_tool_chain_padding(f"update the {['docs', 'config', 'tests'][i % 3]}"))

    # --- Final padding ---
    messages.extend(make_file_read_padding("README.md", lines=40))

    return messages


def _generate_ack(fid: str, fact: str) -> str:
    """Generate a realistic acknowledgment for a planted fact."""
    acks = {
        "F01": "I'll keep that non-standard port in mind for the connection configs.",
        "F02": "I'll configure SSH with the custom port for deployments.",
        "F03": "Redis is a solid choice for sessions — fast and supports TTL natively.",
        "F04": "ARM64 self-hosted runner, nice. I'll make sure the Dockerfile uses multi-arch builds.",
        "F05": "25MB limit noted — I'll check both nginx client_max_body_size and the app validator.",
        "F06": "100/min with burst 20 — I'll set up the rate limiter middleware accordingly.",
        "F07": "PascalCase globals, camelCase locals — I'll follow that convention.",
        "F08": "Race condition on token refresh — that's a classic. I'll look at adding a mutex or atomic refresh.",
        "F09": "Alembic with head at a3f7c2b — I'll make sure new migrations chain from there.",
        "F10": "Python 3.12.1 in /opt/app/.venv — noted, no conda.",
        "F11": "7 days retention, 500MB max — I'll verify logrotate is configured to match.",
        "F12": "Vue 3.4 + Pinia, got it. Modern stack.",
        "F13": "Two CORS origins whitelisted — I'll make sure the API middleware reflects that exactly.",
        "F14": "Daily backup at 03:00 UTC on kraken — I'll check the cron entry.",
        "F15": "Socket.io v4 on /ws/notifications — I'll use the compatible client library.",
    }
    return acks.get(fid, "Noted, I'll factor that in.")


# ============================================================================
# COMPACTION STRATEGIES
# ============================================================================

class BrutalCompactor:
    """
    Single-shot compaction: summarize everything at once.
    Mimics Claude Code's approach — when context is too big, summarize all.
    """

    def __init__(self, maxContextTokens: int = 10_000):
        self.maxContextTokens = maxContextTokens
        self.compactionCount = 0

    def should_compact(self, messages: list[dict], system: str = "") -> bool:
        tokens = estimate_tokens(messages, system)
        return tokens >= self.maxContextTokens * 0.90

    def compact(self, messages: list[dict], llm, system: str = "") -> dict:
        """Compact ALL messages (except last 2) into a single summary."""
        tokensBefore = estimate_tokens(messages, system)

        # Keep only last 2 messages (current exchange)
        keepRecent = 2
        if len(messages) <= keepRecent + 2:
            return {"compacted": False, "reason": "not enough messages"}

        toCompact = messages[:-keepRecent]
        conversationText = messages_to_text(toCompact)

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": f"{COMPACT_PROMPT}{conversationText}"}],
                tools=None,
                system=COMPACT_SYSTEM,
            )
            summary = None
            for block in response.content:
                if block.type == "text":
                    summary = block.text
                    break

            if not summary:
                return {"compacted": False, "reason": "summarization failed"}

        except Exception as e:
            return {"compacted": False, "reason": str(e)}

        summaryPair = [
            {"role": "user", "content": f"[Summary of {len(toCompact)} earlier messages]\n\n{summary}"},
            {"role": "assistant", "content": "Understood. I have the context from our earlier conversation."},
        ]

        messages[:-keepRecent] = summaryPair

        tokensAfter = estimate_tokens(messages, system)
        self.compactionCount += 1

        return {
            "compacted": True,
            "messagesCompacted": len(toCompact),
            "messagesRemaining": len(messages),
            "tokensBefore": tokensBefore,
            "tokensAfterEst": tokensAfter,
            "tokensFreed": tokensBefore - tokensAfter,
            "compactionNumber": self.compactionCount,
        }


# ============================================================================
# INCREMENTAL FEED — simulates gradual message accumulation
# ============================================================================

def feed_incremental(allMessages: list[dict], compactor: ContextCompactor, llm,
                     batchSize: int = 6, system: str = "") -> list[dict]:
    """
    Feed messages in batches, compacting when watermark is hit.
    Returns the final messages list after all compaction cycles.
    """
    messages = []
    stats = []

    for i in range(0, len(allMessages), batchSize):
        batch = allMessages[i:i + batchSize]
        messages.extend(batch)

        # Update token estimate
        compactor.lastInputTokens = estimate_tokens(messages, system)

        # Check watermark and compact if needed
        if compactor.should_compact(messages, system):
            result = compactor.compact(messages, llm, system)
            if result.get("compacted"):
                stats.append(result)
                print(f"  [incremental] Cycle {result['compactionNumber']}: "
                      f"{result['messagesCompacted']} msgs compacted, "
                      f"{result['tokensFreed']} tokens freed")

    return messages, stats


def feed_brutal(allMessages: list[dict], compactor: BrutalCompactor, llm,
                batchSize: int = 6, system: str = "") -> list[dict]:
    """
    Feed messages in batches, compacting brutally when threshold is hit.
    Returns the final messages list after all compaction cycles.
    """
    messages = []
    stats = []

    for i in range(0, len(allMessages), batchSize):
        batch = allMessages[i:i + batchSize]
        messages.extend(batch)

        if compactor.should_compact(messages, system):
            result = compactor.compact(messages, llm, system)
            if result.get("compacted"):
                stats.append(result)
                print(f"  [brutal] Cycle {result['compactionNumber']}: "
                      f"{result['messagesCompacted']} msgs compacted, "
                      f"{result['tokensFreed']} tokens freed")

    return messages, stats


# ============================================================================
# EVALUATION — ask questions, judge answers
# ============================================================================

JUDGE_SYSTEM = "You are an objective evaluator. Answer ONLY with a JSON object."

JUDGE_PROMPT = """Given this conversation context and a question about it, evaluate the assistant's answer.

The expected answer should contain these keywords: {keywords}

Assistant's answer:
{answer}

Evaluate:
1. Does the answer contain the expected information?
2. Is the information accurate (not hallucinated or confused with other facts)?

Respond with ONLY this JSON:
{{"fact_id": "{fact_id}", "recalled": true/false, "accurate": true/false, "notes": "brief explanation"}}"""


def ask_questions(messages: list[dict], llm, system: str = "") -> list[dict]:
    """
    Ask all planted-fact questions to the LLM with the compacted context.
    Returns list of {fact_id, question, answer}.
    """
    results = []

    for factId, _, question, _ in PLANTED_FACTS:
        # Add question to conversation
        questionMessages = messages + [
            {"role": "user", "content": f"Quick question from memory: {question} Be specific with numbers/names."}
        ]

        try:
            response = llm.chat_raw(questionMessages, tools=None, system=system)
            answer = ""
            for block in response.content:
                if block.type == "text":
                    answer = block.text
                    break
        except Exception as e:
            answer = f"[ERROR: {e}]"

        results.append({
            "fact_id": factId,
            "question": question,
            "answer": answer,
        })
        print(f"  {factId}: answered ({len(answer)} chars)")

    return results


def judge_answers(answers: list[dict], llm) -> list[dict]:
    """
    Use LLM-as-judge to evaluate if answers contain the planted facts.
    Returns list of {fact_id, recalled, accurate, notes}.
    """
    evaluations = []

    for answer_data in answers:
        factId = answer_data["fact_id"]
        answer = answer_data["answer"]

        # Find expected keywords
        keywords = []
        for fid, _, _, kw in PLANTED_FACTS:
            if fid == factId:
                keywords = kw
                break

        prompt = JUDGE_PROMPT.format(
            keywords=", ".join(keywords),
            answer=answer,
            fact_id=factId,
        )

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": prompt}],
                tools=None,
                system=JUDGE_SYSTEM,
            )
            judge_text = ""
            for block in response.content:
                if block.type == "text":
                    judge_text = block.text
                    break

            # Parse JSON from judge response
            # Handle markdown code blocks
            judge_text = judge_text.strip()
            if judge_text.startswith("```"):
                judge_text = judge_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            evaluation = json.loads(judge_text)

        except (json.JSONDecodeError, Exception) as e:
            evaluation = {
                "fact_id": factId,
                "recalled": False,
                "accurate": False,
                "notes": f"Judge parse error: {e}",
            }

        evaluations.append(evaluation)
        status = "OK" if evaluation.get("recalled") and evaluation.get("accurate") else "MISS"
        print(f"  {factId}: {status} — {evaluation.get('notes', '')[:60]}")

    return evaluations


# ============================================================================
# METRICS — compute and display results
# ============================================================================

def compute_metrics(evaluations: list[dict], compactionStats: list[dict],
                    label: str) -> dict:
    """Compute recall metrics from judge evaluations."""
    total = len(evaluations)
    recalled = sum(1 for e in evaluations if e.get("recalled"))
    accurate = sum(1 for e in evaluations if e.get("recalled") and e.get("accurate"))

    # Recall by zone (early=F01-F05, mid=F06-F10, late=F11-F15)
    zones = {"early": [], "mid": [], "late": []}
    for e in evaluations:
        fid = e.get("fact_id", "")
        num = int(fid[1:]) if fid.startswith("F") and fid[1:].isdigit() else 0
        if num <= 5:
            zones["early"].append(e)
        elif num <= 10:
            zones["mid"].append(e)
        else:
            zones["late"].append(e)

    zoneRecall = {}
    for zone, evals in zones.items():
        if evals:
            zoneRecall[zone] = sum(1 for e in evals if e.get("recalled")) / len(evals)
        else:
            zoneRecall[zone] = 0.0

    totalCycles = len(compactionStats)
    totalTokensFreed = sum(s.get("tokensFreed", 0) for s in compactionStats)

    return {
        "label": label,
        "recall_global": recalled / total if total > 0 else 0,
        "accuracy": accurate / total if total > 0 else 0,
        "recall_early": zoneRecall["early"],
        "recall_mid": zoneRecall["mid"],
        "recall_late": zoneRecall["late"],
        "facts_recalled": recalled,
        "facts_total": total,
        "compaction_cycles": totalCycles,
        "tokens_freed": totalTokensFreed,
        "efficiency": totalTokensFreed / max(1, total - recalled)
                      if recalled < total else float("inf"),
        "details": evaluations,
    }


def print_results(metricsA: dict, metricsB: dict):
    """Pretty-print comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: Incremental vs Brutal Compaction")
    print("=" * 70)

    headers = f"{'Metric':<30} {'Incremental':>15} {'Brutal':>15}"
    print(headers)
    print("-" * 70)

    rows = [
        ("Recall global",
         f"{metricsA['recall_global']:.0%}", f"{metricsB['recall_global']:.0%}"),
        ("Accuracy",
         f"{metricsA['accuracy']:.0%}", f"{metricsB['accuracy']:.0%}"),
        ("Recall (early facts)",
         f"{metricsA['recall_early']:.0%}", f"{metricsB['recall_early']:.0%}"),
        ("Recall (mid facts)",
         f"{metricsA['recall_mid']:.0%}", f"{metricsB['recall_mid']:.0%}"),
        ("Recall (late facts)",
         f"{metricsA['recall_late']:.0%}", f"{metricsB['recall_late']:.0%}"),
        ("Facts recalled / total",
         f"{metricsA['facts_recalled']}/{metricsA['facts_total']}",
         f"{metricsB['facts_recalled']}/{metricsB['facts_total']}"),
        ("Compaction cycles",
         str(metricsA['compaction_cycles']), str(metricsB['compaction_cycles'])),
        ("Total tokens freed",
         str(metricsA['tokens_freed']), str(metricsB['tokens_freed'])),
    ]

    for label, valA, valB in rows:
        print(f"{label:<30} {valA:>15} {valB:>15}")

    print("=" * 70)

    # Per-fact detail
    print("\nPer-fact breakdown:")
    print(f"{'Fact':<6} {'Incremental':>12} {'Brutal':>12}  Question")
    print("-" * 70)

    for i, (fid, _, question, _) in enumerate(PLANTED_FACTS):
        detA = metricsA["details"][i] if i < len(metricsA["details"]) else {}
        detB = metricsB["details"][i] if i < len(metricsB["details"]) else {}

        statusA = "OK" if detA.get("recalled") and detA.get("accurate") else "MISS"
        statusB = "OK" if detB.get("recalled") and detB.get("accurate") else "MISS"

        print(f"{fid:<6} {statusA:>12} {statusB:>12}  {question[:45]}")

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark: incremental vs brutal compaction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build conversation and show stats without calling LLM")
    parser.add_argument("--max-tokens", type=int, default=10_000,
                        help="Max context tokens (default: 10000 for testing)")
    parser.add_argument("--high-watermark", type=float, default=0.90,
                        help="High watermark for compaction trigger (default: 0.90)")
    parser.add_argument("--low-watermark", type=float, default=0.60,
                        help="Low watermark target after compaction (default: 0.60)")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="Messages per batch when feeding (default: 6)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # --- Build conversation ---
    print("Building scripted conversation...")
    conversation = build_scripted_conversation()
    totalTokens = estimate_tokens(conversation)
    print(f"  {len(conversation)} messages, ~{totalTokens} estimated tokens")
    print(f"  {len(PLANTED_FACTS)} planted facts (F01-F15)")
    print(f"  Max context: {args.max_tokens} tokens")
    print(f"  Watermarks: high={args.high_watermark:.0%}, low={args.low_watermark:.0%}")

    if args.dry_run:
        print("\n[DRY RUN] Conversation structure:")
        for i, msg in enumerate(conversation):
            role = msg["role"]
            content = msg["content"][:80].replace("\n", " ")
            isFact = any(fact in msg["content"] for _, fact, _, _ in PLANTED_FACTS)
            marker = " *** FACT ***" if isFact else ""
            print(f"  [{i:3d}] {role:10s} {content}...{marker}")
        print(f"\n  Would trigger ~{totalTokens // args.max_tokens + 1} compaction cycles")
        return

    # --- Initialize LLM ---
    print("\nInitializing LLM backend...")
    from llm.anthropic import AnthropicLLM
    llm = AnthropicLLM()
    print(f"  Model: {llm.model}")

    system = "You are a helpful assistant working on a software project. Answer questions precisely."

    # --- Run INCREMENTAL compaction ---
    print("\n--- INCREMENTAL COMPACTION ---")
    incCompactor = ContextCompactor(
        maxContextTokens=args.max_tokens,
        highWatermark=args.high_watermark,
        lowWatermark=args.low_watermark,
        minKeepRecent=4,
    )
    incMessages, incStats = feed_incremental(
        deepcopy(conversation), incCompactor, llm,
        batchSize=args.batch_size, system=system,
    )
    print(f"  Final: {len(incMessages)} messages, ~{estimate_tokens(incMessages)} tokens")
    print(f"  Compaction cycles: {len(incStats)}")

    print("\n  Asking questions...")
    incAnswers = ask_questions(incMessages, llm, system=system)

    print("\n  Judging answers...")
    incEvals = judge_answers(incAnswers, llm)

    # --- Run BRUTAL compaction ---
    print("\n--- BRUTAL COMPACTION ---")
    brutalCompactor = BrutalCompactor(maxContextTokens=args.max_tokens)
    brutalMessages, brutalStats = feed_brutal(
        deepcopy(conversation), brutalCompactor, llm,
        batchSize=args.batch_size, system=system,
    )
    print(f"  Final: {len(brutalMessages)} messages, ~{estimate_tokens(brutalMessages)} tokens")
    print(f"  Compaction cycles: {len(brutalStats)}")

    print("\n  Asking questions...")
    brutalAnswers = ask_questions(brutalMessages, llm, system=system)

    print("\n  Judging answers...")
    brutalEvals = judge_answers(brutalAnswers, llm)

    # --- Compute and display metrics ---
    incMetrics = compute_metrics(incEvals, incStats, "Incremental")
    brutalMetrics = compute_metrics(brutalEvals, brutalStats, "Brutal")

    print_results(incMetrics, brutalMetrics)

    # --- Save to JSON if requested ---
    if args.output:
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "max_tokens": args.max_tokens,
                "high_watermark": args.high_watermark,
                "low_watermark": args.low_watermark,
                "batch_size": args.batch_size,
                "conversation_messages": len(conversation),
                "conversation_tokens": totalTokens,
                "planted_facts": len(PLANTED_FACTS),
            },
            "incremental": incMetrics,
            "brutal": brutalMetrics,
        }
        # Remove non-serializable stuff
        for m in [results["incremental"], results["brutal"]]:
            if "efficiency" in m and m["efficiency"] == float("inf"):
                m["efficiency"] = "perfect"

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

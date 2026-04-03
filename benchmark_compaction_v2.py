#!python3
"""
Benchmark v2: Incremental vs Brutal compaction — Large scale.

"Needle in a Compacted Haystack" — scaled up.

Auto-generates N facts, builds a conversation with realistic padding
to reach a target token count, forces multiple compaction cycles,
then evaluates recall via batched LLM-as-judge.

Usage:
    ./benchmark_compaction_v2.py --dry-run
    ./benchmark_compaction_v2.py --max-tokens 30000 --target-tokens 150000 --facts 200
    ./benchmark_compaction_v2.py --preset small   # 50 facts, 50K conv, 15K ctx
    ./benchmark_compaction_v2.py --preset medium  # 200 facts, 150K conv, 30K ctx
    ./benchmark_compaction_v2.py --preset large   # 500 facts, 400K conv, 80K ctx
"""

import sys
import os
import json
import time
import random
import argparse
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv

# Load .env from benchmark root
load_dotenv(Path(__file__).parent / ".env")

from compaction import (
    ContextCompactor, FrozenCompactor, FrozenRankedCompactor,
    estimate_tokens, messages_to_text,
    COMPACT_SYSTEM, COMPACT_PROMPT,
)

# Force unbuffered output (critical for progress monitoring)
_original_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)


# ============================================================================
# LLM WRAPPERS — Anthropic (rate-limited) + Ollama (local)
# ============================================================================

class _TextBlock:
    """Mimics Anthropic's ContentBlock for compatibility with compactor code."""
    def __init__(self, text: str):
        self.type = "text"
        self.text = text

class _FakeResponse:
    """Mimics Anthropic's Message response for compatibility."""
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class RateLimitedLLM:
    """Calls Anthropic API directly with automatic retry on 429 + rate-limit throttling."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", minDelay: float = 2.0):
        import anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        self.minDelay = minDelay
        self._baseDelay = minDelay  # remember original for reset
        self._lastCall = 0.0
        self.totalCalls = 0
        self.totalRetries = 0

    def chat_raw(self, messages, tools=None, system=None):
        """Call LLM with rate-limit awareness and retry on 429."""
        import anthropic

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        maxRetries = 5
        for attempt in range(maxRetries):
            elapsed = time.time() - self._lastCall
            if elapsed < self.minDelay:
                time.sleep(self.minDelay - elapsed)

            try:
                self._lastCall = time.time()
                self.totalCalls += 1
                result = self._client.messages.create(**kwargs)
                # Success: reset to base delay
                self.minDelay = self._baseDelay
                return result

            except anthropic.RateLimitError as e:
                self.totalRetries += 1
                retryAfter = getattr(e.response, 'headers', {}).get('retry-after')
                if retryAfter:
                    waitTime = float(retryAfter)
                else:
                    waitTime = min(120, (2 ** attempt) * 10)

                print(f"  [429] Rate limited, waiting {waitTime:.0f}s "
                      f"(attempt {attempt+1}/{maxRetries}, delay={self.minDelay:.1f}s)...")
                time.sleep(waitTime)
                self.minDelay = min(30, self.minDelay * 1.5)

            except Exception:
                raise

        raise RuntimeError(f"Rate limited after {maxRetries} retries")


class OllamaLLM:
    """Direct Ollama API backend. No rate limits, no cost."""

    def __init__(self, model: str = "qwen2.5:3b", baseUrl: str = "http://localhost:11434",
                 timeout=None):
        import httpx
        self.model = model
        self.baseUrl = baseUrl
        self._client = httpx.Client(timeout=timeout)
        self.totalCalls = 0
        self.totalRetries = 0  # for interface compat

        # Verify connection
        try:
            resp = self._client.get(f"{baseUrl}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(model in m for m in models):
                available = ", ".join(models) or "(none)"
                raise ValueError(f"Model '{model}' not found. Available: {available}")
        except httpx.ConnectError:
            raise ConnectionError(f"Cannot connect to Ollama at {baseUrl}. Is it running?")

    def chat_raw(self, messages, tools=None, system=None):
        """Call Ollama /api/chat and return Anthropic-compatible response."""
        ollamaMessages = []

        # Inject system as first message if provided
        if system:
            ollamaMessages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Flatten content blocks to text
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        text_parts.append(block.get("text", "") or block.get("content", ""))
                content = "\n".join(text_parts)
            ollamaMessages.append({"role": role, "content": content})

        self.totalCalls += 1
        resp = self._client.post(
            f"{self.baseUrl}/api/chat",
            json={
                "model": self.model,
                "messages": ollamaMessages,
                "stream": False,
                "options": {"num_ctx": 32768},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return _FakeResponse(text)


class WrapperLLM:
    """OpenAI-compatible wrapper (claude-code-openai-wrapper). No API cost."""

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 baseUrl: str = "http://localhost:8082/v1", minDelay: float = 0.5):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(base_url=baseUrl, api_key="none", timeout=None)
        self.minDelay = minDelay
        self._lastCall = 0.0
        self.totalCalls = 0
        self.totalRetries = 0

    def chat_raw(self, messages, tools=None, system=None):
        """Call wrapper and return Anthropic-compatible response."""
        oaiMessages = []
        if system:
            oaiMessages.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", "") or block.get("content", ""))
                content = "\n".join(parts)
            oaiMessages.append({"role": role, "content": content})

        elapsed = time.time() - self._lastCall
        if elapsed < self.minDelay:
            time.sleep(self.minDelay - elapsed)

        self._lastCall = time.time()
        self.totalCalls += 1

        response = self._client.chat.completions.create(
            model=self.model,
            messages=oaiMessages,
            max_tokens=4096,
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        return _FakeResponse(text)


# ============================================================================
# FACT GENERATORS — auto-generate diverse, verifiable facts
# ============================================================================

# Templates: (category, template, question_template, keyword_extractor)
# Each generator produces: (fact_id, fact_text, question, expected_keywords)

FACT_TEMPLATES = [
    # --- Server / Infra ---
    {
        "category": "server",
        "make": lambda i, r: {
            "text": f"Server '{r.choice(['atlas', 'kronos', 'helios', 'mantis', 'phoenix', 'neptune', 'vega', 'orion', 'titan', 'pluto'])}-{i:03d}' runs on {r.randint(10,172)}.{r.randint(0,255)}.{r.randint(1,254)}.{r.randint(1,254)}, SSH port {r.choice([22, 2222, 2200, 8022, 9022])}.",
            "q": f"What is the IP address and SSH port of server *-{i:03d}?",
        },
    },
    # --- Software versions ---
    {
        "category": "version",
        "make": lambda i, r: {
            "text": f"The {r.choice(['backend', 'frontend', 'API', 'worker', 'scheduler', 'gateway'])} service uses {r.choice(['Python', 'Node.js', 'Go', 'Rust', 'Java'])} {r.randint(1,3)}.{r.randint(0,22)}.{r.randint(0,9)} in {r.choice(['production', 'staging', 'development'])}.",
            "q": f"What language and version does service #{i} use?",
        },
    },
    # --- File paths + line numbers ---
    {
        "category": "code_location",
        "make": lambda i, r: {
            "text": f"There's a {r.choice(['bug', 'TODO', 'performance issue', 'security vulnerability', 'deprecated call'])} in {r.choice(['src', 'lib', 'app', 'core'])}/{r.choice(['auth', 'api', 'db', 'cache', 'queue', 'parser', 'config', 'handler'])}.{r.choice(['py', 'js', 'go', 'rs'])} at line {r.randint(10, 999)}.",
            "q": f"Where is the issue #{i} located (file and line)?",
        },
    },
    # --- Config values ---
    {
        "category": "config",
        "make": lambda i, r: {
            "text": f"Config '{r.choice(['MAX_CONNECTIONS', 'TIMEOUT_MS', 'RETRY_COUNT', 'BATCH_SIZE', 'CACHE_TTL', 'POOL_SIZE', 'RATE_LIMIT', 'BUFFER_SIZE'])}_{i}' is set to {r.randint(1, 10000)} in {r.choice(['production', 'staging'])}.",
            "q": f"What is the value of config parameter #{i}?",
        },
    },
    # --- Technical decisions ---
    {
        "category": "decision",
        "make": lambda i, r: {
            "chosen": r.choice(['Redis', 'Memcached', 'PostgreSQL', 'MongoDB', 'SQLite', 'DynamoDB', 'Kafka', 'RabbitMQ', 'gRPC', 'REST', 'GraphQL', 'WebSocket']),
            "rejected": r.choice(['MySQL', 'MariaDB', 'CouchDB', 'Cassandra', 'NATS', 'ZeroMQ', 'SOAP', 'Thrift']),
            "text": lambda s: f"For the {r.choice(['caching', 'messaging', 'storage', 'communication', 'queuing', 'indexing'])} layer, we chose {s['chosen']} over {s['rejected']}.",
            "q": f"What technology was chosen (and what was rejected) for decision #{i}?",
        },
    },
    # --- Dates and schedules ---
    {
        "category": "schedule",
        "make": lambda i, r: {
            "text": f"The {r.choice(['backup', 'cleanup', 'report', 'sync', 'health-check', 'rotation'])} job runs {r.choice(['daily', 'weekly', 'hourly', 'every 6 hours'])} at {r.randint(0,23):02d}:{r.choice(['00', '15', '30', '45'])} UTC.",
            "q": f"When does scheduled job #{i} run?",
        },
    },
    # --- Credentials / endpoints ---
    {
        "category": "endpoint",
        "make": lambda i, r: {
            "text": f"The {r.choice(['internal', 'external', 'admin', 'public', 'webhook'])} API endpoint is /{r.choice(['api', 'v2', 'internal'])}/{r.choice(['users', 'orders', 'events', 'metrics', 'health', 'config', 'auth'])}/{r.choice(['list', 'create', 'status', 'export', 'sync'])} on port {r.choice([8080, 8443, 3000, 5000, 9090, 4000])}.",
            "q": f"What is the path and port of endpoint #{i}?",
        },
    },
    # --- Team / people ---
    {
        "category": "people",
        "make": lambda i, r: {
            "text": f"{r.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hugo', 'Iris', 'Jules'])} is responsible for the {r.choice(['authentication', 'deployment', 'monitoring', 'database', 'frontend', 'API', 'security', 'testing'])} module and prefers {r.choice(['tabs', 'spaces', '2-space indent', '4-space indent'])}.",
            "q": f"Who is responsible for module #{i} and what's their preference?",
        },
    },
]


def generate_facts(n: int, seed: int = 42) -> list[tuple]:
    """
    Auto-generate n unique, verifiable facts.
    Returns: [(fact_id, fact_text, question, expected_keywords), ...]
    """
    r = random.Random(seed)
    facts = []

    for i in range(n):
        template = FACT_TEMPLATES[i % len(FACT_TEMPLATES)]
        raw = template["make"](i, r)

        # Handle the "decision" template special case
        if "chosen" in raw:
            text = raw["text"](raw)
            keywords = [raw["chosen"].lower(), raw["rejected"].lower()]
        else:
            text = raw["text"]
            # Extract keywords: numbers, IPs, paths, names
            keywords = _extract_keywords(text)

        factId = f"F{i:04d}"
        question = raw["q"]
        facts.append((factId, text, question, keywords))

    return facts


def _extract_keywords(text: str) -> list[str]:
    """Extract verifiable keywords from a fact text."""
    import re
    keywords = []

    # IP addresses
    ips = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text)
    keywords.extend(ips)

    # Port numbers (after "port")
    ports = re.findall(r'port\s+(\d+)', text, re.IGNORECASE)
    keywords.extend(ports)

    # Version numbers (X.Y.Z)
    versions = re.findall(r'(\d+\.\d+\.\d+)', text)
    keywords.extend(versions)

    # Quoted names
    quoted = re.findall(r"'([^']+)'", text)
    keywords.extend([q.lower() for q in quoted])

    # File paths
    paths = re.findall(r'([\w/]+\.(?:py|js|go|rs|ts|java))', text)
    keywords.extend(paths)

    # Line numbers (after "line")
    lines = re.findall(r'line\s+(\d+)', text, re.IGNORECASE)
    keywords.extend(lines)

    # Numbers after "set to" or "is"
    values = re.findall(r'set to\s+(\d+)', text)
    keywords.extend(values)

    # Times (HH:MM)
    times = re.findall(r'(\d{2}:\d{2})', text)
    keywords.extend(times)

    # Named tech (capitalized words that aren't common English)
    techs = re.findall(r'\b(Redis|Memcached|PostgreSQL|MongoDB|SQLite|DynamoDB|Kafka|RabbitMQ|'
                       r'gRPC|REST|GraphQL|WebSocket|Python|Node\.js|Go|Rust|Java)\b', text)
    keywords.extend([t.lower() for t in techs])

    # API paths
    api_paths = re.findall(r'(/[\w/]+)', text)
    keywords.extend(api_paths)

    # Frequency
    freqs = re.findall(r'(daily|weekly|hourly|every \d+ hours)', text, re.IGNORECASE)
    keywords.extend([f.lower() for f in freqs])

    # Names (capitalized, likely person names)
    names = re.findall(r'\b(Alice|Bob|Charlie|Diana|Eve|Frank|Grace|Hugo|Iris|Jules)\b', text)
    keywords.extend([n.lower() for n in names])

    # Deduplicate, keep non-empty
    seen = set()
    unique = []
    for kw in keywords:
        kw = kw.strip()
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            unique.append(kw)

    return unique[:5]  # Cap at 5 keywords per fact


# ============================================================================
# PADDING GENERATORS — scaled up
# ============================================================================

def make_code_block(filename: str, lines: int, r: random.Random) -> str:
    """Generate a realistic-looking code block."""
    langs = {
        ".py": ["def ", "class ", "import ", "    return ", "    if ", "    for ", "# "],
        ".js": ["function ", "const ", "import ", "  return ", "  if ", "  for ", "// "],
        ".go": ["func ", "type ", "import ", "    return ", "    if ", "    for ", "// "],
        ".rs": ["fn ", "struct ", "use ", "    return ", "    if ", "    for ", "// "],
    }
    ext = Path(filename).suffix
    prefixes = langs.get(ext, langs[".py"])

    code_lines = []
    for i in range(lines):
        prefix = r.choice(prefixes)
        varname = r.choice(["data", "result", "config", "value", "item", "buffer", "ctx"])
        code_lines.append(f"{prefix}{varname}_{i} = process(input[{i}])  # step {i}")
    return "\n".join(code_lines)


def make_padding_exchange(kind: str, r: random.Random, targetChars: int = 2000) -> list[dict]:
    """Generate a padding exchange of approximately targetChars total."""
    if kind == "file_read":
        filename = f"src/{r.choice(['core', 'api', 'lib', 'utils', 'services'])}/{r.choice(['handler', 'processor', 'manager', 'controller', 'engine'])}.{r.choice(['py', 'js', 'go', 'rs'])}"
        lines = max(20, targetChars // 60)
        code = make_code_block(filename, lines, r)
        return [
            {"role": "user", "content": f"Can you read {filename}?"},
            {"role": "assistant", "content": f"Here's `{filename}` ({lines} lines):\n\n```\n{code}\n```\n\nThe file contains {lines} lines of processing logic with {r.choice(['error handling', 'data validation', 'caching', 'async operations'])}. Let me know if you need changes."},
        ]

    elif kind == "discussion":
        topic = r.choice([
            "database indexing strategy", "API versioning approach", "error handling patterns",
            "caching invalidation", "deployment rollback strategy", "log aggregation",
            "service mesh architecture", "feature flag implementation", "load balancing config",
            "secret management", "CI/CD pipeline optimization", "test coverage strategy",
        ])
        filler = f"Regarding {topic}, " + " ".join(
            r.choice(["the current approach", "we should consider", "one option is",
                      "the trade-off here", "performance-wise", "from a maintenance perspective",
                      "the key constraint", "given our scale", "based on profiling"])
            + " " + r.choice(["is suboptimal", "works well", "needs rethinking",
                             "has proven reliable", "could be improved", "is the bottleneck"])
            + "." for _ in range(targetChars // 80)
        )
        return [
            {"role": "user", "content": f"What's your take on {topic}? We need to decide soon."},
            {"role": "assistant", "content": filler[:targetChars]},
            {"role": "user", "content": r.choice(["Makes sense, let's go with that.", "Good analysis. Let's revisit after the sprint.", "Agreed, option A sounds right."])},
            {"role": "assistant", "content": r.choice(["Sounds good. I'll factor that into the implementation.", "Perfect, I'll proceed accordingly.", "Noted. Let's move on to the next item."])},
        ]

    elif kind == "tool_chain":
        task = r.choice([
            "refactor the authentication flow", "update the database schema",
            "fix the failing test suite", "optimize the query performance",
            "add input validation", "update the API documentation",
            "configure the monitoring alerts", "set up the staging environment",
        ])
        detail = " ".join(
            f"Modified {r.choice(['src', 'lib', 'app'])}/{r.choice(['auth', 'db', 'api', 'core'])}.{r.choice(['py', 'js', 'go'])} — "
            + r.choice(["added error handling", "fixed edge case", "optimized query", "updated config", "refactored logic"])
            + "." for _ in range(targetChars // 120)
        )
        return [
            {"role": "user", "content": f"Can you {task}?"},
            {"role": "assistant", "content": f"I'll {task}. Here's what I did:\n\n{detail[:targetChars]}\n\nAll changes are consistent with the existing patterns."},
        ]

    return []


# ============================================================================
# CONVERSATION BUILDER — scalable
# ============================================================================

def build_conversation(facts: list[tuple], targetTokens: int, seed: int = 42) -> list[dict]:
    """
    Build a conversation interleaving facts with padding to reach targetTokens.

    Strategy:
    - Distribute facts evenly across the conversation
    - Fill gaps with padding of varying types and sizes
    - Ensure user/assistant alternation is maintained
    """
    r = random.Random(seed)
    messages = []

    # Opening
    messages.extend([
        {"role": "user", "content": "Hey, I need help with a complex project. Let me give you context as we go."},
        {"role": "assistant", "content": "Sure! I'm ready. Share the details and I'll help wherever I can."},
    ])

    nFacts = len(facts)
    currentTokens = estimate_tokens(messages)
    paddingKinds = ["file_read", "discussion", "tool_chain"]

    # Token budget per fact (padding + fact itself)
    tokensPerFact = max(300, (targetTokens - currentTokens) // max(1, nFacts))

    for i, (factId, factText, _, _) in enumerate(facts):
        # Add padding BEFORE the fact to reach the per-fact token target
        tokenTarget = currentTokens + tokensPerFact - 50  # reserve ~50 for the fact exchange
        while estimate_tokens(messages) < tokenTarget:
            kind = r.choice(paddingKinds)
            chunkChars = r.randint(1500, 5000)
            padding = make_padding_exchange(kind, r, targetChars=chunkChars)
            messages.extend(padding)

        # Add the fact
        messages.extend([
            {"role": "user", "content": factText},
            {"role": "assistant", "content": f"Got it, noted. I'll keep that in mind for the implementation."},
        ])

        currentTokens = estimate_tokens(messages)

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Built {i+1}/{nFacts} facts, ~{currentTokens:,} tokens so far...")

    return messages


# ============================================================================
# COMPACTION STRATEGIES (same as v1, with progress)
# ============================================================================

class BrutalCompactor:
    """Single-shot: summarize everything except last 2 messages."""

    def __init__(self, maxContextTokens: int):
        self.maxContextTokens = maxContextTokens
        self.compactionCount = 0
        # Scale maxChars with context: leave room for system prompt + COMPACT_PROMPT + max_tokens
        # Real ratio is ~3.17 chars/token (not 4), so be conservative
        self.maxChars = int(maxContextTokens * 2.4)  # ~150K real tokens for 200K context

    def should_compact(self, messages: list[dict], system: str = "") -> bool:
        return estimate_tokens(messages, system) >= self.maxContextTokens * 0.90

    def compact(self, messages: list[dict], llm, system: str = "") -> dict:
        tokensBefore = estimate_tokens(messages, system)
        keepRecent = 2
        if len(messages) <= keepRecent + 2:
            return {"compacted": False, "reason": "not enough messages"}

        toCompact = messages[:-keepRecent]
        conversationText = messages_to_text(toCompact)

        # Truncate if text exceeds model's capacity
        if len(conversationText) > self.maxChars:
            conversationText = conversationText[:self.maxChars] + "\n\n[...truncated...]"

        inputChars = len(COMPACT_PROMPT) + len(conversationText)
        print(f"    [brutal] Sending {inputChars:,} chars (~{inputChars//4:,} tokens) to summarize...")

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": f"{COMPACT_PROMPT}{conversationText}"}],
                tools=None, system=COMPACT_SYSTEM,
            )
            summary = None
            for block in response.content:
                if block.type == "text":
                    summary = block.text
                    break
            if not summary:
                return {"compacted": False, "reason": f"summarization failed (stop={getattr(response, 'stop_reason', '?')}, blocks={len(response.content)})"}
        except Exception as e:
            return {"compacted": False, "reason": f"{type(e).__name__}: {e}"}

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
# FEED STRATEGIES
# ============================================================================

def feed_strategy(allMessages: list[dict], compactor, llm, label: str,
                  batchSize: int = 10, system: str = "") -> tuple[list[dict], list[dict]]:
    """Feed messages in batches, compact using any compactor at watermark.

    Works with ContextCompactor, FrozenCompactor, and BrutalCompactor.
    """
    messages = []
    stats = []
    consecutiveFailures = 0
    maxConsecutiveFailures = 3  # stop trying after 3 failures in a row

    total = len(allMessages)
    for i in range(0, total, batchSize):
        batch = allMessages[i:i + batchSize]
        messages.extend(batch)

        # Update token estimate for compactors that track it
        if hasattr(compactor, 'lastInputTokens'):
            compactor.lastInputTokens = estimate_tokens(messages, system)

        if consecutiveFailures >= maxConsecutiveFailures:
            continue  # stop trying, just feed remaining messages

        if compactor.should_compact(messages, system):
            result = compactor.compact(messages, llm, system)
            if result.get("compacted"):
                stats.append(result)
                extra = ""
                if result.get("type") == "freeze":
                    extra = f", frozen#{result.get('frozenSummaries', '?')}"
                if result.get("merge"):
                    m = result["merge"]
                    extra += f" +merge#{m.get('mergeNumber', '?')}"
                    if "newRank" in m:
                        extra += f" R{m.get('mergedRanks','?')}+R{m.get('mergedRanks','?')}->R{m['newRank']}"
                print(f"  [{label}] Cycle {result.get('compactionNumber', '?')}: "
                      f"{result.get('messagesCompacted', '?')} msgs -> "
                      f"{result.get('messagesRemaining', '?')} remaining, "
                      f"{result.get('tokensFreed', '?')} tokens freed{extra} "
                      f"(fed {min(i + batchSize, total)}/{total} msgs)")
                consecutiveFailures = 0
            else:
                consecutiveFailures += 1
                reason = result.get("reason", "unknown")
                print(f"  [{label}] Compact FAILED ({consecutiveFailures}/{maxConsecutiveFailures}): "
                      f"{reason} (msgs={len(messages)}, ~{estimate_tokens(messages):,} tokens)")
                if consecutiveFailures >= maxConsecutiveFailures:
                    print(f"  [{label}] Giving up compaction after {maxConsecutiveFailures} failures")

    return messages, stats


# ============================================================================
# BATCHED EVALUATION
# ============================================================================

BATCH_QUESTION_PROMPT = """Answer each of the following questions based ONLY on what you know from our conversation.
Be specific: include exact numbers, names, paths, versions.
If you don't remember or aren't sure, say "I don't recall".

{questions}

Reply with a JSON array of objects, one per question:
[{{"id": "F0001", "answer": "your answer"}}, ...]

IMPORTANT: Return ONLY the JSON array, no other text."""


BATCH_JUDGE_PROMPT = """Evaluate whether each answer contains the expected information.

{entries}

For each entry, check:
1. Does the answer contain the expected keywords?
2. Is the information accurate (not hallucinated)?

Reply with ONLY a JSON array:
[{{"id": "F0001", "recalled": true/false, "accurate": true/false, "notes": "brief reason"}}, ...]"""

JUDGE_SYSTEM = "You are an objective evaluator. Answer ONLY with valid JSON."


def ask_questions_batched(messages: list[dict], facts: list[tuple], llm,
                          system: str = "", batchSize: int = 10) -> list[dict]:
    """Ask all questions in batches. Returns [{fact_id, question, answer}, ...]."""
    allResults = []

    for batchStart in range(0, len(facts), batchSize):
        batch = facts[batchStart:batchStart + batchSize]

        questionsText = "\n".join(
            f"- [{fid}] {question}"
            for fid, _, question, _ in batch
        )

        prompt = BATCH_QUESTION_PROMPT.format(questions=questionsText)
        questionMessages = messages + [{"role": "user", "content": prompt}]

        try:
            response = llm.chat_raw(questionMessages, tools=None, system=system)
            responseText = ""
            for block in response.content:
                if block.type == "text":
                    responseText = block.text
                    break

            # Parse JSON response
            responseText = responseText.strip()
            if responseText.startswith("```"):
                responseText = responseText.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            answers = json.loads(responseText)

            # Map answers back to facts
            answerMap = {a["id"]: a.get("answer", "") for a in answers}
            for fid, _, question, _ in batch:
                allResults.append({
                    "fact_id": fid,
                    "question": question,
                    "answer": answerMap.get(fid, "[no answer in batch response]"),
                })

        except (json.JSONDecodeError, Exception) as e:
            # Fallback: mark all in batch as unanswered
            for fid, _, question, _ in batch:
                allResults.append({
                    "fact_id": fid,
                    "question": question,
                    "answer": f"[batch error: {e}]",
                })

        answered = batchStart + len(batch)
        print(f"  Questions: {answered}/{len(facts)} answered")
        time.sleep(0.5)  # Light rate limiting

    return allResults


def judge_answers_batched(answers: list[dict], facts: list[tuple], llm,
                           batchSize: int = 15) -> list[dict]:
    """Judge all answers in batches. Returns [{fact_id, recalled, accurate, notes}, ...]."""
    # Build keyword map
    keywordMap = {fid: kw for fid, _, _, kw in facts}

    allEvals = []

    for batchStart in range(0, len(answers), batchSize):
        batch = answers[batchStart:batchStart + batchSize]

        entriesText = "\n\n".join(
            f"[{a['fact_id']}]\n"
            f"  Expected keywords: {', '.join(keywordMap.get(a['fact_id'], []))}\n"
            f"  Answer: {a['answer']}"
            for a in batch
        )

        prompt = BATCH_JUDGE_PROMPT.format(entries=entriesText)

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": prompt}],
                tools=None, system=JUDGE_SYSTEM,
            )
            responseText = ""
            for block in response.content:
                if block.type == "text":
                    responseText = block.text
                    break

            responseText = responseText.strip()
            if responseText.startswith("```"):
                responseText = responseText.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            evals = json.loads(responseText)
            evalMap = {e["id"]: e for e in evals}

            for a in batch:
                fid = a["fact_id"]
                ev = evalMap.get(fid, {"recalled": False, "accurate": False, "notes": "missing from judge response"})
                ev["fact_id"] = fid  # Ensure present
                allEvals.append(ev)

        except (json.JSONDecodeError, Exception) as e:
            for a in batch:
                allEvals.append({
                    "fact_id": a["fact_id"],
                    "recalled": False,
                    "accurate": False,
                    "notes": f"judge error: {e}",
                })

        judged = batchStart + len(batch)
        print(f"  Judged: {judged}/{len(answers)}")
        time.sleep(0.5)

    return allEvals


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(evaluations: list[dict], compactionStats: list[dict],
                    facts: list[tuple], label: str) -> dict:
    """Compute recall metrics with zone breakdown."""
    total = len(evaluations)
    recalled = sum(1 for e in evaluations if e.get("recalled"))
    accurate = sum(1 for e in evaluations if e.get("recalled") and e.get("accurate"))

    # Divide into zones based on position
    zoneSize = total // 3
    zones = {
        "early": evaluations[:zoneSize],
        "mid": evaluations[zoneSize:2 * zoneSize],
        "late": evaluations[2 * zoneSize:],
    }

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
        "details": evaluations,
    }


def print_results(allMetrics: list[dict], facts: list[tuple]):
    """Pretty-print comparison table for N strategies."""
    nStrats = len(allMetrics)
    colWidth = 14

    print("\n" + "=" * (40 + colWidth * nStrats))
    print("  BENCHMARK v2: Compaction Strategy Comparison")
    print("=" * (40 + colWidth * nStrats))

    # Header
    header = f"  {'Metric':<35}"
    for m in allMetrics:
        header += f" {m['label']:>{colWidth}}"
    print(header)
    print("  " + "-" * (35 + colWidth * nStrats))

    # Rows
    rowDefs = [
        ("Recall global", lambda m: f"{m['recall_global']:.1%}"),
        ("Accuracy (recalled & correct)", lambda m: f"{m['accuracy']:.1%}"),
        ("Recall — early (first 1/3)", lambda m: f"{m['recall_early']:.1%}"),
        ("Recall — mid (middle 1/3)", lambda m: f"{m['recall_mid']:.1%}"),
        ("Recall — late (last 1/3)", lambda m: f"{m['recall_late']:.1%}"),
        ("Facts recalled / total", lambda m: f"{m['facts_recalled']}/{m['facts_total']}"),
        ("Compaction cycles", lambda m: str(m['compaction_cycles'])),
        ("Total tokens freed", lambda m: f"{m['tokens_freed']:,}"),
    ]

    for label, fmt in rowDefs:
        row = f"  {label:<35}"
        for m in allMetrics:
            row += f" {fmt(m):>{colWidth}}"
        print(row)

    print("  " + "=" * (35 + colWidth * nStrats))

    # Zone heatmap
    print("\n  Zone recall heatmap:")
    for zone in ["early", "mid", "late"]:
        line = f"  {zone:>8}: "
        for m in allMetrics:
            rec = m[f"recall_{zone}"]
            bar = "#" * int(rec * 20) + "." * (20 - int(rec * 20))
            shortLabel = m["label"][:3]
            line += f" {shortLabel} [{bar}] {rec:.0%} "
        print(line)

    # Find best strategy
    best = max(allMetrics, key=lambda m: m["recall_global"])
    print(f"\n  Best recall: {best['label']} ({best['recall_global']:.1%})")

    print()


# ============================================================================
# PRESETS
# ============================================================================

PRESETS = {
    "small": {
        "facts": 50,
        "target_tokens": 50_000,
        "max_tokens": 15_000,
        "batch_size": 10,
        "description": "Quick test: ~50K conv, 15K ctx, ~3 cycles",
    },
    "medium": {
        "facts": 200,
        "target_tokens": 150_000,
        "max_tokens": 30_000,
        "batch_size": 10,
        "description": "Standard: ~150K conv, 30K ctx, ~5 cycles",
    },
    "large": {
        "facts": 500,
        "target_tokens": 400_000,
        "max_tokens": 80_000,
        "batch_size": 14,
        "description": "Heavy: ~400K conv, 80K ctx, ~5-6 cycles",
    },
    "full": {
        "facts": 150,
        "target_tokens": 1_500_000,
        "max_tokens": 200_000,
        "batch_size": 10,
        "description": "Full 200K: ~1.5M conv, 200K ctx, 150 facts (1/10K), ~22 inc cycles",
    },
    "xxl": {
        "facts": 1000,
        "target_tokens": 1_000_000,
        "max_tokens": 200_000,
        "batch_size": 20,
        "description": "Full scale: ~1M conv, 200K ctx (real Claude), ~13 cycles",
    },
    "heavy": {
        "facts": 300,
        "target_tokens": 3_000_000,
        "max_tokens": 200_000,
        "batch_size": 10,
        "description": "Heavy 200K: ~3M conv, 200K ctx, 300 facts (1/10K), ~47 inc cycles",
    },
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark v2: incremental vs brutal compaction (large scale)")
    parser.add_argument("--preset", choices=PRESETS.keys(), default=None,
                        help="Use a preset config (small/medium/large)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build conversation and show stats, no LLM calls")
    parser.add_argument("--facts", type=int, default=200,
                        help="Number of facts to plant (default: 200)")
    parser.add_argument("--target-tokens", type=int, default=150_000,
                        help="Target conversation size in tokens (default: 150000)")
    parser.add_argument("--max-tokens", type=int, default=30_000,
                        help="Max context window tokens (default: 30000)")
    parser.add_argument("--high-watermark", type=float, default=0.90)
    parser.add_argument("--low-watermark", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Messages per feed batch (default: 10)")
    parser.add_argument("--question-batch", type=int, default=10,
                        help="Questions per LLM call (default: 10)")
    parser.add_argument("--strategy", choices=["all", "incremental", "brutal", "frozen", "frozen_ranked"],
                        default="all",
                        help="Which strategy to run (default: all)")
    parser.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic",
                        help="LLM backend (default: anthropic)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (default: haiku for anthropic, qwen2.5:3b for ollama)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--min-delay", type=float, default=2.0,
                        help="Min seconds between API calls, anthropic only (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Apply preset if specified
    if args.preset:
        p = PRESETS[args.preset]
        args.facts = p["facts"]
        args.target_tokens = p["target_tokens"]
        args.max_tokens = p["max_tokens"]
        args.batch_size = p["batch_size"]
        print(f"Using preset '{args.preset}': {p['description']}")

    # Which strategies to run
    if args.strategy == "all":
        strategies = ["incremental", "brutal", "frozen", "frozen_ranked"]
    else:
        strategies = [args.strategy]

    # --- Generate facts ---
    print(f"\nGenerating {args.facts} facts (seed={args.seed})...")
    facts = generate_facts(args.facts, seed=args.seed)
    print(f"  {len(facts)} facts across {len(FACT_TEMPLATES)} categories")

    print("  Sample facts:")
    for fid, text, q, kw in facts[:3]:
        print(f"    {fid}: {text[:70]}...")
        print(f"          Q: {q}")
        print(f"          Keywords: {kw}")

    # --- Build conversation ---
    print(f"\nBuilding conversation (target: ~{args.target_tokens:,} tokens)...")
    conversation = build_conversation(facts, args.target_tokens, seed=args.seed)
    actualTokens = estimate_tokens(conversation)
    print(f"  {len(conversation)} messages, ~{actualTokens:,} estimated tokens")

    incCycles = max(1, (actualTokens - args.max_tokens) //
                   int(args.max_tokens * (args.high_watermark - args.low_watermark)))
    brutalFreePerCycle = int(args.max_tokens * (args.high_watermark - 0.05))
    brutalCycles = max(1, (actualTokens - args.max_tokens) // brutalFreePerCycle)
    print(f"  Max context: {args.max_tokens:,} tokens")
    print(f"  Watermarks: high={args.high_watermark:.0%}, low={args.low_watermark:.0%}")
    print(f"  Expected cycles: ~{incCycles} incremental, ~{brutalCycles} brutal")

    nStrats = len(strategies)
    questionBatches = args.facts // args.question_batch
    apiCalls = (incCycles + brutalCycles + incCycles) + questionBatches * nStrats * 2
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Estimated API calls: ~{apiCalls}")

    if args.dry_run:
        print("\n[DRY RUN] Fact distribution:")
        factPositions = []
        for i, msg in enumerate(conversation):
            for fid, ftext, _, _ in facts:
                if ftext == msg.get("content"):
                    factPositions.append((i, fid))
                    break

        nMsgs = len(conversation)
        for pos, fid in factPositions[:10]:
            zone = "early" if pos < nMsgs // 3 else "mid" if pos < 2 * nMsgs // 3 else "late"
            print(f"  msg [{pos:4d}/{nMsgs}] {zone:>5} — {fid}")
        if len(factPositions) > 10:
            print(f"  ... and {len(factPositions) - 10} more facts")

        print(f"\n  Total: {len(factPositions)} facts positioned")
        print(f"  Zone distribution: early={sum(1 for p, _ in factPositions if p < nMsgs//3)}, "
              f"mid={sum(1 for p, _ in factPositions if nMsgs//3 <= p < 2*nMsgs//3)}, "
              f"late={sum(1 for p, _ in factPositions if p >= 2*nMsgs//3)}")
        return

    # --- Initialize LLM ---
    if args.backend == "ollama":
        modelName = args.model or "qwen2.5:3b"
        print(f"\nInitializing Ollama backend (model: {modelName})...")
        llm = OllamaLLM(model=modelName, baseUrl=args.ollama_url)
        print(f"  Model: {llm.model} (local, no rate limit)")
    else:
        modelName = args.model or "claude-haiku-4-5-20251001"
        print(f"\nInitializing Anthropic backend (model: {modelName})...")
        llm = RateLimitedLLM(model=modelName, minDelay=args.min_delay)
        print(f"  Model: {llm.model}")
        print(f"  Min delay between calls: {args.min_delay}s (auto-increases on 429)")

    system = "You are a helpful assistant working on a complex software project. Answer questions precisely from memory."
    startTime = time.time()

    # --- Run each strategy ---
    allMetrics = []
    allResults = {}
    checkpointBase = args.output.rsplit(".", 1)[0] if args.output else "results"

    for stratName in strategies:
        phaseNum = strategies.index(stratName) + 1
        print(f"\n{'=' * 60}")
        print(f"  PHASE {phaseNum}/{nStrats}: {stratName.upper()} COMPACTION")
        print(f"{'=' * 60}")

        # Create compactor
        if stratName == "incremental":
            compactor = ContextCompactor(
                maxContextTokens=args.max_tokens,
                highWatermark=args.high_watermark,
                lowWatermark=args.low_watermark,
                minKeepRecent=4,
            )
        elif stratName == "brutal":
            compactor = BrutalCompactor(maxContextTokens=args.max_tokens)
        elif stratName == "frozen":
            compactor = FrozenCompactor(
                maxContextTokens=args.max_tokens,
                highWatermark=args.high_watermark,
                lowWatermark=args.low_watermark,
                minKeepRecent=4,
            )
        elif stratName == "frozen_ranked":
            compactor = FrozenRankedCompactor(
                maxContextTokens=args.max_tokens,
                highWatermark=args.high_watermark,
                lowWatermark=args.low_watermark,
                minKeepRecent=4,
            )

        # Feed + compact
        phaseStart = time.time()
        messages, stats = feed_strategy(
            deepcopy(conversation), compactor, llm, label=stratName,
            batchSize=args.batch_size, system=system,
        )
        feedTime = time.time() - phaseStart
        print(f"  Feed+compact done in {feedTime:.0f}s: {len(messages)} msgs, "
              f"~{estimate_tokens(messages):,} tokens, {len(stats)} cycles")

        # Ensure context is small enough for Q&A (leave room for questions)
        qaHeadroom = int(args.max_tokens * 0.15)  # 15% headroom for questions
        qaThreshold = args.max_tokens - qaHeadroom
        contextTokens = estimate_tokens(messages, system)
        if contextTokens > qaThreshold:
            print(f"  Context too large for Q&A ({contextTokens:,} > {qaThreshold:,}), "
                  f"running final compaction...")
            result = compactor.compact(messages, llm, system)
            if result.get("compacted"):
                stats.append(result)
                print(f"  Final compact: {result.get('tokensFreed', '?')} tokens freed, "
                      f"now ~{estimate_tokens(messages):,} tokens")
            else:
                print(f"  WARNING: final compact failed ({result.get('reason')}), "
                      f"Q&A may have token limit errors")

        # Ask questions
        print(f"\n  Asking questions (batched, {args.question_batch}/batch)...")
        answers = ask_questions_batched(
            messages, facts, llm, system=system, batchSize=args.question_batch
        )

        # Judge
        print(f"\n  Judging answers (batched, {args.question_batch}/batch)...")
        evals = judge_answers_batched(answers, facts, llm, batchSize=args.question_batch)

        # Compute metrics
        metrics = compute_metrics(evals, stats, facts, stratName.capitalize())
        allMetrics.append(metrics)

        # Checkpoint: save this strategy's results immediately
        allResults[stratName] = {
            "metrics": {k: v for k, v in metrics.items() if k != "details"},
            "compaction_stats": stats,
            "details": metrics["details"],
        }

        checkpointFile = f"{checkpointBase}_{stratName}_checkpoint.json"
        with open(checkpointFile, "w", encoding="utf-8") as f:
            json.dump(allResults[stratName], f, indent=2, ensure_ascii=False, default=str)
        print(f"  Checkpoint saved: {checkpointFile}")

    # --- Final results ---
    elapsed = time.time() - startTime
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  API calls: {llm.totalCalls}, retries (429): {llm.totalRetries}")

    print_results(allMetrics, facts)

    # --- Save combined results ---
    if args.output:
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "config": {
                "backend": args.backend,
                "model": llm.model,
                "facts": args.facts,
                "target_tokens": args.target_tokens,
                "max_tokens": args.max_tokens,
                "high_watermark": args.high_watermark,
                "low_watermark": args.low_watermark,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "strategies": strategies,
                "conversation_messages": len(conversation),
                "conversation_tokens": actualTokens,
            },
        }
        for stratName, data in allResults.items():
            results[stratName] = data["metrics"]
            results[f"{stratName}_compaction_stats"] = data["compaction_stats"]
            results[f"{stratName}_details"] = data["details"]

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

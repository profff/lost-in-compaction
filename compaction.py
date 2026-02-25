"""
Incremental context compaction - PoC

Dual-watermark approach:
- High watermark (default 90%): triggers compaction
- Low watermark (default 60%): target after compaction

The gap between thresholds determines:
- Frequency: larger gap = less frequent compaction
- Intensity: larger gap = more content compacted per pass
- Quality: less frequent = fewer summarize-the-summary degradation cycles
"""

import json
import re


COMPACT_SYSTEM = "You are a conversation summarizer. Be concise and precise."

COMPACT_PROMPT = """Summarize the following conversation segment.

PRESERVE (critical):
- Key facts and decisions made
- File paths and code structures mentioned
- User preferences and requirements stated
- Actions completed (files created/edited, commands run)
- Errors encountered and how they were resolved

DISCARD:
- Full file contents (keep only: filename, line count, key elements)
- Verbose tool outputs (keep only: what was done + result)
- Intermediate reasoning that led nowhere
- Redundant back-and-forth

Format: Concise narrative summary. Bullet points for lists of facts.
Target: ~20% of original length, keeping all actionable information.

---

"""


def estimate_tokens(messages: list[dict], system: str = "") -> int:
    """Rough token estimate. ~4 chars per token for mixed EN/FR content."""
    total = len(system) // 4
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("content", "") or block.get("text", "") or ""
                    if isinstance(text, str):
                        total += len(text) // 4
                    if block.get("type") == "tool_use":
                        total += len(json.dumps(block.get("input", {}))) // 4
    return total


def messages_to_text(messages: list[dict]) -> str:
    """Convert messages to readable text for the summarizer LLM."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")

        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(f"[{role}]: {block.get('text', '')}")
                elif btype == "tool_use":
                    name = block.get("name", "?")
                    args = json.dumps(block.get("input", {}), ensure_ascii=False)
                    if len(args) > 300:
                        args = args[:297] + "..."
                    parts.append(f"[{role} -> {name}]: {args}")
                elif btype == "tool_result":
                    text = block.get("content", "")
                    if isinstance(text, str) and len(text) > 500:
                        text = text[:497] + "...[truncated]"
                    parts.append(f"[TOOL RESULT]: {text}")

    return "\n\n".join(parts)


def _make_compact_marker(toolName: str, toolInput: dict, content: str) -> str:
    """Generate a compact marker for a tool result."""
    lineCount = content.count('\n') + 1
    charCount = len(content)

    if toolName == "read_file":
        path = toolInput.get("path", "?")
        return f"[read_file: {path}, {lineCount} lines, {charCount} chars — already analyzed]"

    elif toolName == "bash":
        cmd = toolInput.get("command", "?")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"[bash: `{cmd}`, {lineCount} lines output — already analyzed]"

    elif toolName == "grep":
        pattern = toolInput.get("pattern", "?")
        path = toolInput.get("path", ".")
        return f"[grep: '{pattern}' in {path}, {lineCount} matches — already analyzed]"

    elif toolName == "glob":
        pattern = toolInput.get("pattern", "?")
        return f"[glob: '{pattern}', {lineCount} results — already analyzed]"

    elif toolName == "list_directory":
        path = toolInput.get("path", ".")
        return f"[list_directory: {path}, {lineCount} entries — already analyzed]"

    elif toolName == "git":
        subcommand = toolInput.get("subcommand", "?")
        return f"[git {subcommand}, {lineCount} lines — already analyzed]"

    elif toolName == "analyze":
        action = toolInput.get("action", "?")
        return f"[analyze: {action}, {lineCount} lines — already analyzed]"

    elif toolName in ("web_fetch", "web_search"):
        query = toolInput.get("url", "") or toolInput.get("query", "?")
        if len(query) > 60:
            query = query[:57] + "..."
        return f"[{toolName}: {query}, {lineCount} lines — already analyzed]"

    else:
        return f"[{toolName}: {lineCount} lines, {charCount} chars — already analyzed]"


# Tools whose results should never be compacted
SKIP_COMPACT_TOOLS = {"think"}


def compact_tool_results(messages: list[dict], minKeepTurns: int = 1, threshold: int = 500) -> int:
    """
    Replace large tool_result contents with compact markers (Type A compaction).

    Scans messages older than minKeepTurns and replaces tool_result content
    above threshold chars with a short marker. No LLM call needed — the LLM
    already analyzed the content in its response.

    Args:
        messages: conversation messages (mutated in-place)
        minKeepTurns: don't touch tool results from the last N user turns
        threshold: minimum content length (chars) to trigger compaction

    Returns:
        number of tool results compacted
    """
    # Find the boundary: count user text messages from the end
    turnCount = 0
    safeBoundary = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and isinstance(messages[i].get("content"), str):
            turnCount += 1
            if turnCount >= minKeepTurns:
                safeBoundary = i
                break

    if safeBoundary <= 0:
        return 0

    # Build map: tool_use_id -> {name, input} from assistant messages
    toolUseMap = {}
    for msg in messages[:safeBoundary]:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                toolUseMap[block.id] = {
                    "name": block.name,
                    "input": getattr(block, "input", {}),
                }

    # Find and compact large tool_results
    compacted = 0
    for msg in messages[:safeBoundary]:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            rawContent = block.get("content", "")
            if not isinstance(rawContent, str) or len(rawContent) <= threshold:
                continue

            # Find which tool produced this result
            toolUseId = block.get("tool_use_id", "")
            toolInfo = toolUseMap.get(toolUseId, {})
            toolName = toolInfo.get("name", "unknown")

            if toolName in SKIP_COMPACT_TOOLS:
                continue

            toolInput = toolInfo.get("input", {})
            block["content"] = _make_compact_marker(toolName, toolInput, rawContent)
            compacted += 1

    return compacted


def compact_tool_results_except_last(messages: list[dict], threshold: int = 500) -> int:
    """
    Compact all large tool_results except those in the last user message.

    Simpler variant for mid-chain use: no turn counting, just skip the last
    user message (which contains the tool results the LLM hasn't seen yet).
    """
    if not messages:
        return 0

    # Find the last user message index
    lastUserIdx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            lastUserIdx = i
            break

    if lastUserIdx is None:
        return 0

    # Build tool_use map from ALL assistant messages
    toolUseMap = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                toolUseMap[block.id] = {
                    "name": block.name,
                    "input": getattr(block, "input", {}),
                }

    # Compact large tool_results in all user messages EXCEPT the last one
    compacted = 0
    for i, msg in enumerate(messages):
        if i == lastUserIdx:
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            rawContent = block.get("content", "")
            if not isinstance(rawContent, str) or len(rawContent) <= threshold:
                continue
            # Already compacted?
            if "already analyzed" in rawContent:
                continue

            toolUseId = block.get("tool_use_id", "")
            toolInfo = toolUseMap.get(toolUseId, {})
            toolName = toolInfo.get("name", "unknown")

            if toolName in SKIP_COMPACT_TOOLS:
                continue

            toolInput = toolInfo.get("input", {})
            block["content"] = _make_compact_marker(toolName, toolInput, rawContent)
            compacted += 1

    return compacted


class ContextCompactor:
    """Dual-watermark incremental context compaction."""

    def __init__(
        self,
        maxContextTokens: int = 200_000,
        highWatermark: float = 0.90,
        lowWatermark: float = 0.60,
        minKeepRecent: int = 6,
    ):
        self.maxContextTokens = maxContextTokens
        self.highWatermark = highWatermark
        self.lowWatermark = lowWatermark
        self.minKeepRecent = minKeepRecent  # never compact the last N messages

        # Tracking
        self.lastInputTokens = 0   # actual count from API
        self.compactionCount = 0
        self.totalTokensFreed = 0

    def update_from_api(self, inputTokens: int):
        """Called after each LLM API call with actual token count."""
        self.lastInputTokens = inputTokens

    def get_usage(self, messages: list[dict], system: str = "") -> dict:
        """Current context usage stats."""
        estimated = estimate_tokens(messages, system)
        actual = self.lastInputTokens if self.lastInputTokens > 0 else estimated
        return {
            "tokens": actual,
            "estimated": estimated,
            "max": self.maxContextTokens,
            "pct": (actual / self.maxContextTokens) * 100 if self.maxContextTokens > 0 else 0,
            "high_pct": self.highWatermark * 100,
            "low_pct": self.lowWatermark * 100,
        }

    def should_compact(self, messages: list[dict], system: str = "") -> bool:
        """True if context has hit the high watermark."""
        usage = self.get_usage(messages, system)
        return usage["tokens"] >= self.maxContextTokens * self.highWatermark

    def _find_safe_cut_points(self, messages: list[dict], maxIndex: int) -> list[int]:
        """
        Find indices where it's safe to cut without orphaning tool_use/tool_result.

        A safe cut point is right AFTER an assistant text response (no tool_use)
        and BEFORE a user text message (not tool_results).
        This ensures no tool chain spans the cut boundary.
        """
        safePoints = []
        for i in range(min(len(messages) - 1, maxIndex)):
            msg = messages[i]
            nextMsg = messages[i + 1]

            # Current msg must be assistant with NO tool_use
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                # Plain text assistant response (e.g. from a previous compaction summary)
                isEndOfTurn = True
            elif isinstance(content, list):
                # Content blocks: safe only if no tool_use blocks
                hasToolUse = any(
                    getattr(b, "type", None) == "tool_use"
                    for b in content
                )
                isEndOfTurn = not hasToolUse
            else:
                continue

            if not isEndOfTurn:
                continue

            # Next msg must be user with string content (text, not tool_results)
            if nextMsg.get("role") == "user" and isinstance(nextMsg.get("content"), str):
                safePoints.append(i + 1)  # cut AFTER this assistant msg

        return safePoints

    def compact(self, messages: list[dict], llm, system: str = "") -> dict:
        """
        Compact oldest messages down to low watermark.
        Mutates messages list in-place.
        Returns stats dict.
        """
        tokensBefore = self.get_usage(messages, system)["tokens"]
        targetTokens = int(self.maxContextTokens * self.lowWatermark)
        tokensToFree = tokensBefore - targetTokens

        if tokensToFree <= 0:
            return {"compacted": False, "reason": "already below low watermark"}

        # Messages we can compact (everything except the last minKeepRecent)
        safeEnd = max(0, len(messages) - self.minKeepRecent)
        if safeEnd < 2:
            return {"compacted": False, "reason": "not enough messages to compact"}

        # Find safe cut points (between complete exchanges, no orphaned tool chains)
        safePoints = self._find_safe_cut_points(messages, safeEnd)
        if not safePoints:
            return {"compacted": False, "reason": "no safe cut point (all in tool chains)"}

        # For each safe point, calc cumulative tokens. Pick the first that frees enough.
        # If none frees enough, pick the last safe point (free as much as possible).
        runningTokens = 0
        tokensByIndex = {}
        for i in range(safeEnd):
            runningTokens += estimate_tokens([messages[i]])
            if (i + 1) in safePoints:
                tokensByIndex[i + 1] = runningTokens

        cutIndex = safePoints[-1]  # default: last safe point
        for sp in safePoints:
            if tokensByIndex.get(sp, 0) >= tokensToFree:
                cutIndex = sp
                break

        if cutIndex < 2:
            return {"compacted": False, "reason": "safe cut point too early"}

        # Summarize the messages we're about to remove
        toCompact = messages[:cutIndex]
        summary = self._summarize(toCompact, llm)

        if not summary:
            return {"compacted": False, "reason": "summarization failed"}

        # Replace with a summary pair (maintains user/assistant alternation)
        summaryPair = [
            {
                "role": "user",
                "content": f"[Summary of {len(toCompact)} earlier messages]\n\n{summary}"
            },
            {
                "role": "assistant",
                "content": "Understood. I have the context from our earlier conversation."
            },
        ]

        messages[:cutIndex] = summaryPair

        # Recalculate token count after compaction.
        # Raw estimate (chars/4) underestimates real tokens, so apply a correction
        # factor based on the ratio we observed before compaction.
        tokensAfterEst = estimate_tokens(messages, system)
        estimateBefore = estimate_tokens(toCompact) + tokensAfterEst
        if estimateBefore > 0:
            correctionFactor = tokensBefore / estimateBefore
            self.lastInputTokens = int(tokensAfterEst * correctionFactor)
        else:
            self.lastInputTokens = tokensAfterEst

        # Stats
        self.compactionCount += 1
        tokensFreed = tokensBefore - self.lastInputTokens
        self.totalTokensFreed += tokensFreed

        return {
            "compacted": True,
            "messagesCompacted": len(toCompact),
            "messagesRemaining": len(messages),
            "tokensBefore": tokensBefore,
            "tokensAfterEst": self.lastInputTokens,
            "tokensFreed": tokensFreed,
            "compactionNumber": self.compactionCount,
        }

    def _summarize(self, messages: list[dict], llm) -> str | None:
        """Send messages to LLM for summarization."""
        conversationText = messages_to_text(messages)

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": f"{COMPACT_PROMPT}{conversationText}"}],
                tools=None,
                system=COMPACT_SYSTEM,
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            return None
        except Exception as e:
            print(f"[Compaction] Summarization failed: {e}")
            return None


# ============================================================================
# FROZEN COMPACTOR — incremental with frozen summaries (no re-summarization)
# ============================================================================

FROZEN_SUMMARY_PREFIX = "[FROZEN SUMMARY"

MERGE_PROMPT = """Merge these conversation summaries into one concise summary.

PRESERVE (critical):
- ALL key facts, decisions, numbers, names, file paths
- Technical details: IP addresses, ports, versions, config values
- Who said what, who decided what

Target: ~50% of the combined original length. Keep ALL factual information,
compress only the narrative and redundancies.

---

"""


class FrozenCompactor:
    """Incremental compaction with frozen summaries — no re-summarization.

    Unlike ContextCompactor which re-summarizes previous summaries on each cycle,
    this compactor freezes each summary block. Only raw (non-summary) messages
    get compacted. When frozen summaries accumulate past a budget, the oldest
    ones are merged — but this happens less frequently.

    Context structure after N cycles:
        [FROZEN #1 (maybe merged)] [FROZEN #2] ... [FROZEN #N] [raw recent msgs]

    This creates a natural temporal gradient:
    - Recent messages: full detail (raw)
    - Medium-age: individual summaries (frozen, ~4:1 compression)
    - Old: merged summaries (more condensed, but only merged once or twice)
    """

    def __init__(
        self,
        maxContextTokens: int = 200_000,
        highWatermark: float = 0.90,
        lowWatermark: float = 0.60,
        minKeepRecent: int = 6,
        summaryBudgetPct: float = 0.50,
    ):
        self.maxContextTokens = maxContextTokens
        self.highWatermark = highWatermark
        self.lowWatermark = lowWatermark
        self.minKeepRecent = minKeepRecent
        # Max tokens allocated to frozen summaries (fraction of low watermark target)
        self.summaryBudget = int(maxContextTokens * lowWatermark * summaryBudgetPct)

        # Tracking
        self.lastInputTokens = 0
        self.compactionCount = 0
        self.mergeCount = 0
        self.totalTokensFreed = 0

    def update_from_api(self, inputTokens: int):
        """Called after each LLM API call with actual token count."""
        self.lastInputTokens = inputTokens

    def get_usage(self, messages: list[dict], system: str = "") -> dict:
        """Current context usage stats."""
        estimated = estimate_tokens(messages, system)
        actual = self.lastInputTokens if self.lastInputTokens > 0 else estimated
        return {
            "tokens": actual,
            "estimated": estimated,
            "max": self.maxContextTokens,
            "pct": (actual / self.maxContextTokens) * 100 if self.maxContextTokens > 0 else 0,
        }

    def should_compact(self, messages: list[dict], system: str = "") -> bool:
        """True if context has hit the high watermark."""
        usage = self.get_usage(messages, system)
        return usage["tokens"] >= self.maxContextTokens * self.highWatermark

    def _find_frozen_boundary(self, messages: list[dict]) -> int:
        """Index of the first non-frozen message.

        Frozen summaries are always at the start, as user/assistant pairs:
          [0] user: "[FROZEN SUMMARY #1] ..."
          [1] assistant: "Understood..."
          [2] user: "[FROZEN SUMMARY #2] ..."
          [3] assistant: "Understood..."
          [4] user: <first raw message>  ← boundary = 4
        """
        i = 0
        while i < len(messages):
            content = messages[i].get("content", "")
            if isinstance(content, str) and content.startswith(FROZEN_SUMMARY_PREFIX):
                i += 2  # skip user summary + assistant ack
            else:
                break
        return i

    def _count_frozen_pairs(self, messages: list[dict], boundary: int) -> int:
        """Count frozen summary pairs (each pair = 1 summary)."""
        count = 0
        for i in range(0, boundary, 2):
            content = messages[i].get("content", "")
            if isinstance(content, str) and content.startswith(FROZEN_SUMMARY_PREFIX):
                count += 1
        return count

    def _find_safe_cut_points(self, messages: list[dict], start: int, end: int) -> list[int]:
        """Find safe cut points in messages[start:end].

        A safe cut point is right AFTER an assistant text response and BEFORE
        a user text message, ensuring no tool chain spans the boundary.
        Returns absolute indices into the messages list.
        """
        safePoints = []
        for i in range(start, min(len(messages) - 1, end)):
            msg = messages[i]
            nextMsg = messages[i + 1]

            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                isEndOfTurn = True
            elif isinstance(content, list):
                hasToolUse = any(
                    getattr(b, "type", None) == "tool_use"
                    for b in content
                )
                isEndOfTurn = not hasToolUse
            else:
                continue

            if not isEndOfTurn:
                continue

            if nextMsg.get("role") == "user" and isinstance(nextMsg.get("content"), str):
                safePoints.append(i + 1)

        return safePoints

    def compact(self, messages: list[dict], llm, system: str = "") -> dict:
        """Compact oldest RAW messages into a new frozen summary.

        Does NOT touch existing frozen summaries. If summary budget is exceeded
        after creating the new summary, merges the 2 oldest frozen summaries.

        Mutates messages list in-place. Returns stats dict.
        """
        tokensBefore = self.get_usage(messages, system)["tokens"]
        targetTokens = int(self.maxContextTokens * self.lowWatermark)
        tokensToFree = tokensBefore - targetTokens

        if tokensToFree <= 0:
            return {"compacted": False, "reason": "already below low watermark"}

        frozenBoundary = self._find_frozen_boundary(messages)
        safeEnd = max(0, len(messages) - self.minKeepRecent)

        # If all non-frozen messages are too recent, try merging summaries
        if frozenBoundary >= safeEnd:
            return self._merge_summaries(messages, llm, system, tokensBefore)

        # Find safe cut points in the raw portion only
        safePoints = self._find_safe_cut_points(messages, frozenBoundary, safeEnd)

        if not safePoints:
            return {"compacted": False, "reason": "no safe cut point in raw messages"}

        # Calculate cumulative tokens from frozenBoundary to find the right cut
        runningTokens = 0
        tokensByIndex = {}
        for i in range(frozenBoundary, safeEnd):
            runningTokens += estimate_tokens([messages[i]])
            if (i + 1) in safePoints:
                tokensByIndex[i + 1] = runningTokens

        # Pick first safe point that frees enough, or last safe point
        cutIndex = safePoints[-1]
        for sp in safePoints:
            if tokensByIndex.get(sp, 0) >= tokensToFree:
                cutIndex = sp
                break

        if cutIndex <= frozenBoundary:
            return {"compacted": False, "reason": "cut point at frozen boundary"}

        # Summarize the raw messages between frozenBoundary and cutIndex
        toCompact = messages[frozenBoundary:cutIndex]
        summary = self._summarize(toCompact, llm)

        if not summary:
            return {"compacted": False, "reason": "summarization failed"}

        self.compactionCount += 1
        nFrozen = self._count_frozen_pairs(messages, frozenBoundary)

        summaryPair = [
            {
                "role": "user",
                "content": f"{FROZEN_SUMMARY_PREFIX} #{nFrozen + 1}]\n\n{summary}",
            },
            {
                "role": "assistant",
                "content": "Understood. I have this context noted.",
            },
        ]

        # Replace raw messages with frozen summary (insert at frozenBoundary)
        messages[frozenBoundary:cutIndex] = summaryPair

        # Recalculate tokens with correction factor
        tokensAfterEst = estimate_tokens(messages, system)
        estimateBefore = estimate_tokens(toCompact) + tokensAfterEst
        if estimateBefore > 0:
            correctionFactor = tokensBefore / estimateBefore
            self.lastInputTokens = int(tokensAfterEst * correctionFactor)
        else:
            self.lastInputTokens = tokensAfterEst

        tokensFreed = tokensBefore - self.lastInputTokens
        self.totalTokensFreed += tokensFreed

        result = {
            "compacted": True,
            "type": "freeze",
            "messagesCompacted": len(toCompact),
            "messagesRemaining": len(messages),
            "tokensBefore": tokensBefore,
            "tokensAfterEst": self.lastInputTokens,
            "tokensFreed": tokensFreed,
            "compactionNumber": self.compactionCount,
            "frozenSummaries": nFrozen + 1,
        }

        # Check if frozen summaries exceed budget → merge oldest
        newFrozenBoundary = self._find_frozen_boundary(messages)
        frozenTokens = estimate_tokens(messages[:newFrozenBoundary])

        if frozenTokens > self.summaryBudget and newFrozenBoundary >= 4:
            mergeResult = self._merge_summaries(
                messages, llm, system, self.lastInputTokens
            )
            if mergeResult.get("compacted"):
                result["merge"] = mergeResult

        return result

    def _merge_summaries(self, messages: list[dict], llm, system: str,
                         tokensBefore: int) -> dict:
        """Merge the 2 oldest frozen summaries into 1."""
        frozenBoundary = self._find_frozen_boundary(messages)

        if frozenBoundary < 4:
            return {"compacted": False, "reason": "not enough summaries to merge"}

        # Take the 2 oldest summary pairs (4 messages)
        toMerge = messages[:4]
        mergeText = messages_to_text(toMerge)

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": f"{MERGE_PROMPT}{mergeText}"}],
                tools=None,
                system=COMPACT_SYSTEM,
            )
            mergedSummary = None
            for block in response.content:
                if block.type == "text":
                    mergedSummary = block.text
                    break

            if not mergedSummary:
                return {"compacted": False, "reason": "merge summarization failed"}

        except Exception as e:
            print(f"[FrozenCompactor] Merge failed: {e}")
            return {"compacted": False, "reason": f"merge failed: {e}"}

        self.mergeCount += 1

        mergedPair = [
            {
                "role": "user",
                "content": f"{FROZEN_SUMMARY_PREFIX} #merged-{self.mergeCount}]\n\n{mergedSummary}",
            },
            {
                "role": "assistant",
                "content": "Understood. I have this merged context noted.",
            },
        ]

        messages[:4] = mergedPair

        tokensAfterEst = estimate_tokens(messages, system)
        self.lastInputTokens = tokensAfterEst
        tokensFreed = tokensBefore - tokensAfterEst
        self.totalTokensFreed += tokensFreed

        return {
            "compacted": True,
            "type": "merge",
            "tokensBefore": tokensBefore,
            "tokensAfterEst": tokensAfterEst,
            "tokensFreed": tokensFreed,
            "mergeNumber": self.mergeCount,
        }

    def _summarize(self, messages: list[dict], llm) -> str | None:
        """Send messages to LLM for summarization."""
        conversationText = messages_to_text(messages)

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": f"{COMPACT_PROMPT}{conversationText}"}],
                tools=None,
                system=COMPACT_SYSTEM,
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            return None
        except Exception as e:
            print(f"[FrozenCompactor] Summarization failed: {e}")
            return None


# ============================================================================
# FROZEN RANKED COMPACTOR — frozen with hierarchical rank-aware merging
# ============================================================================


class FrozenRankedCompactor(FrozenCompactor):
    """Frozen compactor with rank-aware hierarchical merging.

    Each frozen summary carries a rank:
    - rank 1: fresh summary from raw messages
    - rank N+1: result of merging two rank-N summaries

    When budget is exceeded, merges the first pair of summaries with the
    lowest available rank. This ensures each fact passes through at most
    log2(N) merges instead of up to N/2 sequential merges.

    Merge progression example (6 initial summaries):
        [R1] [R1] [R1] [R1] [R1] [R1]       budget exceeded
        [R2] [R1] [R1] [R1] [R1]             merge first 2 R1s
        [R2] [R2] [R1] [R1]                  merge next 2 R1s
        [R2] [R2] [R2]                       merge last 2 R1s
        [R3] [R2]                            no more R1 pairs, merge R2s
    """

    def _parse_rank(self, content: str) -> int:
        """Extract rank from a frozen summary marker.

        Format: [FROZEN SUMMARY #N (rank R)]
        Returns 1 if no rank specified (backwards compat with unranked).
        """
        match = re.search(r'\(rank (\d+)\)', content)
        if match:
            return int(match.group(1))
        return 1

    def _get_frozen_ranks(self, messages: list[dict], boundary: int) -> list[tuple[int, int]]:
        """Get list of (message_index, rank) for all frozen summary pairs."""
        ranks = []
        for i in range(0, boundary, 2):
            content = messages[i].get("content", "")
            if isinstance(content, str) and content.startswith(FROZEN_SUMMARY_PREFIX):
                rank = self._parse_rank(content)
                ranks.append((i, rank))
        return ranks

    def _find_merge_pair(self, messages: list[dict], boundary: int) -> tuple[int, int] | None:
        """Find the first pair of same-rank summaries to merge.

        Scans from lowest rank upward. Returns (idx1, idx2) as message
        indices of the two user messages to merge, or None.
        """
        ranks = self._get_frozen_ranks(messages, boundary)
        if len(ranks) < 2:
            return None

        # Find the lowest rank with at least 2 summaries
        from collections import Counter
        rankCounts = Counter(r for _, r in ranks)

        for targetRank in sorted(rankCounts.keys()):
            if rankCounts[targetRank] >= 2:
                # Pick the first two summaries of this rank
                indices = [i for i, r in ranks if r == targetRank]
                return (indices[0], indices[1])

        return None

    def compact(self, messages: list[dict], llm, system: str = "") -> dict:
        """Compact oldest RAW messages into a new rank-1 frozen summary.

        Same as FrozenCompactor.compact but markers include (rank 1).
        Merge uses rank-aware pair selection.
        """
        tokensBefore = self.get_usage(messages, system)["tokens"]
        targetTokens = int(self.maxContextTokens * self.lowWatermark)
        tokensToFree = tokensBefore - targetTokens

        if tokensToFree <= 0:
            return {"compacted": False, "reason": "already below low watermark"}

        frozenBoundary = self._find_frozen_boundary(messages)
        safeEnd = max(0, len(messages) - self.minKeepRecent)

        if frozenBoundary >= safeEnd:
            return self._merge_summaries(messages, llm, system, tokensBefore)

        safePoints = self._find_safe_cut_points(messages, frozenBoundary, safeEnd)

        if not safePoints:
            return {"compacted": False, "reason": "no safe cut point in raw messages"}

        runningTokens = 0
        tokensByIndex = {}
        for i in range(frozenBoundary, safeEnd):
            runningTokens += estimate_tokens([messages[i]])
            if (i + 1) in safePoints:
                tokensByIndex[i + 1] = runningTokens

        cutIndex = safePoints[-1]
        for sp in safePoints:
            if tokensByIndex.get(sp, 0) >= tokensToFree:
                cutIndex = sp
                break

        if cutIndex <= frozenBoundary:
            return {"compacted": False, "reason": "cut point at frozen boundary"}

        toCompact = messages[frozenBoundary:cutIndex]
        summary = self._summarize(toCompact, llm)

        if not summary:
            return {"compacted": False, "reason": "summarization failed"}

        self.compactionCount += 1
        nFrozen = self._count_frozen_pairs(messages, frozenBoundary)

        # New frozen summaries from raw messages are always rank 1
        summaryPair = [
            {
                "role": "user",
                "content": f"{FROZEN_SUMMARY_PREFIX} #{nFrozen + 1} (rank 1)]\n\n{summary}",
            },
            {
                "role": "assistant",
                "content": "Understood. I have this context noted.",
            },
        ]

        messages[frozenBoundary:cutIndex] = summaryPair

        tokensAfterEst = estimate_tokens(messages, system)
        estimateBefore = estimate_tokens(toCompact) + tokensAfterEst
        if estimateBefore > 0:
            correctionFactor = tokensBefore / estimateBefore
            self.lastInputTokens = int(tokensAfterEst * correctionFactor)
        else:
            self.lastInputTokens = tokensAfterEst

        tokensFreed = tokensBefore - self.lastInputTokens
        self.totalTokensFreed += tokensFreed

        result = {
            "compacted": True,
            "type": "freeze",
            "messagesCompacted": len(toCompact),
            "messagesRemaining": len(messages),
            "tokensBefore": tokensBefore,
            "tokensAfterEst": self.lastInputTokens,
            "tokensFreed": tokensFreed,
            "compactionNumber": self.compactionCount,
            "frozenSummaries": nFrozen + 1,
        }

        # Check if frozen summaries exceed budget → rank-aware merge
        newFrozenBoundary = self._find_frozen_boundary(messages)
        frozenTokens = estimate_tokens(messages[:newFrozenBoundary])

        if frozenTokens > self.summaryBudget and newFrozenBoundary >= 4:
            mergeResult = self._merge_summaries(
                messages, llm, system, self.lastInputTokens
            )
            if mergeResult.get("compacted"):
                result["merge"] = mergeResult

        return result

    def _merge_summaries(self, messages: list[dict], llm, system: str,
                         tokensBefore: int) -> dict:
        """Merge first pair of lowest-rank frozen summaries.

        Unlike FrozenCompactor which always merges the 2 oldest, this finds
        the first pair of same-rank summaries (lowest rank first). The merged
        result gets rank+1.
        """
        frozenBoundary = self._find_frozen_boundary(messages)
        pair = self._find_merge_pair(messages, frozenBoundary)

        if pair is None:
            return {"compacted": False, "reason": "no same-rank pair to merge"}

        idx1, idx2 = pair
        rank = self._parse_rank(messages[idx1].get("content", ""))
        newRank = rank + 1

        # Extract content from both pairs
        toMerge = messages[idx1:idx1 + 2] + messages[idx2:idx2 + 2]
        mergeText = messages_to_text(toMerge)

        try:
            response = llm.chat_raw(
                [{"role": "user", "content": f"{MERGE_PROMPT}{mergeText}"}],
                tools=None,
                system=COMPACT_SYSTEM,
            )
            mergedSummary = None
            for block in response.content:
                if block.type == "text":
                    mergedSummary = block.text
                    break

            if not mergedSummary:
                return {"compacted": False, "reason": "merge summarization failed"}

        except Exception as e:
            print(f"[FrozenRankedCompactor] Merge failed: {e}")
            return {"compacted": False, "reason": f"merge failed: {e}"}

        self.mergeCount += 1

        mergedPair = [
            {
                "role": "user",
                "content": (f"{FROZEN_SUMMARY_PREFIX} #merged-{self.mergeCount} "
                            f"(rank {newRank})]\n\n{mergedSummary}"),
            },
            {
                "role": "assistant",
                "content": "Understood. I have this merged context noted.",
            },
        ]

        # Remove second pair first (higher index), then replace first pair
        # This avoids index shifting issues
        messages[idx2:idx2 + 2] = []
        messages[idx1:idx1 + 2] = mergedPair

        tokensAfterEst = estimate_tokens(messages, system)
        self.lastInputTokens = tokensAfterEst
        tokensFreed = tokensBefore - tokensAfterEst
        self.totalTokensFreed += tokensFreed

        return {
            "compacted": True,
            "type": "merge",
            "mergedRanks": rank,
            "newRank": newRank,
            "tokensBefore": tokensBefore,
            "tokensAfterEst": tokensAfterEst,
            "tokensFreed": tokensFreed,
            "mergeNumber": self.mergeCount,
        }

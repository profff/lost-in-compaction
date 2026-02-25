# Frozen Summaries: Preventing Information Loss in LLM Context Compaction

> Paper v2 — restructured with calibrated evidence (v5) and controlled compaction loss

## Abstract

Long-running LLM conversations inevitably exceed the context window, forcing
systems to compact old messages through summarization. But how do we know
whether our compaction benchmark accurately measures information loss? Before
comparing strategies, we need to understand how reliably an LLM can recall
facts from its own context — and what factors affect that recall.

We present a two-part study. In **Part A**, we calibrate a recall benchmark
using 234 naturally-embedded facts from the LongMemEval dataset across a
190K-token context. We discover three measurement pitfalls that affect all
compaction benchmarks: (1) a **batch size effect** where asking 10 questions
at once yields 75 percentage points higher recall than asking one at a time,
(2) a **category hierarchy** where temporal reasoning (30% max) and preference
recall (40% max) are near-impossible regardless of strategy, (3) a **density
saturation** where recall plateaus beyond ~0.4 facts/kTok despite increasing
evidence, and (4) a **grep-LLM gap** where keyword search finds 86% of facts
but the LLM only recalls 25–79% of them — information is *present* in the
context but *ignored* by attention.

In **Part B**, we isolate compaction loss by compacting 5–98% of a context,
re-padding to the original size, and measuring recall degradation. Even 5%
compaction costs 7 percentage points of recall. At 50%, the compacted zone is
near-dead (0–7% recall) despite keywords surviving at 82–93% via grep.
Critically, compaction damages even the *untouched* portion of the context:
remaining-zone recall drops from 68% to 39% as compaction increases — an
**attention dilution** effect caused by injecting noise (re-padding) into the
context. Repeatability testing across 3 independent runs confirms these effects
are stable (σ ≤ 2.6pp) against measured deltas of 7–70pp.

These findings provide calibrated baselines for **Part C** (in progress):
a rigorous comparison of multi-pass compaction strategies (Brutal, Incremental,
Frozen, FrozenRanked) on multi-megaToken conversations.


## 1. Introduction

### 1.1 The problem

Stateless LLM APIs require the full conversation to be sent at each turn. When
conversations exceed the context window (typically 128K–200K tokens), some form
of compaction is necessary. The naive approach — summarize and discard old
messages — is universally adopted by production systems (Claude Code, Cursor,
Windsurf, MemGPT/Letta). But how much information survives this process?

Answering this question requires a reliable measurement tool. Most existing
benchmarks treat recall measurement as straightforward: embed facts, ask
questions, count correct answers. We found that the measurement itself is
surprisingly fragile — sensitive to how many questions are asked at once,
which categories of facts are tested, and whether "present in context" actually
means "retrievable by the model."

This led us to a two-phase approach: first calibrate the recall benchmark
(Part A), then use it to measure compaction loss (Part B). The calibration
itself yielded findings that challenge common assumptions about LLM recall.

### 1.2 Related work

**Needle in a Haystack** (Kamradt, 2023). The foundational test for long-context
retrieval: embed a specific fact in a long document and measure whether the
model can find it. Tests the model's *native* retrieval ability within a single
context window. Our work extends this paradigm to measure fact *survival*
across compaction cycles — "Needle in a Compacted Haystack."

**LongMemEval** (Wang et al., 2025). Benchmarks long-term memory capabilities
of LLM systems across 500 questions in 6 categories (knowledge update, temporal
reasoning, multi-session, preferences, etc.) spanning up to 115 sessions. We
use LongMemEval evidence as our fact source, providing ecologically valid,
naturally-embedded facts rather than synthetic injections.

**Context Rot** (Chroma Research, 2025). Demonstrates that LLM performance
degrades systematically as input length increases, even on trivially simple
tasks. Evaluates 18 models and recommends compaction/summarization as
mitigation, but does not compare compaction strategies against each other
or measure information loss across strategies.

**MemGPT / Letta** (Packer et al., 2023). Proposes virtual context management
inspired by OS memory hierarchies: main context (RAM) and external context
(disk). When context overflows, messages are evicted and recursively
summarized — functionally identical to our incremental strategy. The paper
notes that "older messages have progressively less influence on the summary"
but does not benchmark this degradation or compare alternative strategies.

**Factory.ai — Evaluating Context Compression** (2025). The closest work to
ours. Compares three compression *implementations* (Factory's anchored
iterative, OpenAI Compact, Anthropic SDK) using probe-based evaluation
(recall, artifact, continuation, decision probes) on 36,000 production
messages. Key differences with our work:
- Factory compares *implementations* from different vendors; we compare
  fundamentally different *architectures* (single-shot vs iterative vs frozen)
- No zone-based recall metrics — their evaluation does not reveal where
  in the conversation information is lost
- No frozen summary strategy or space-time tradeoff analysis
- No calibration of the recall measurement itself

**Beyond a Million Tokens** (2025). Benchmarks long-term memory up to 10M
tokens across diverse domains, testing recall, multi-hop reasoning,
contradiction resolution, and temporal ordering. Tests model *capabilities*
at various context lengths, not compaction *strategy* effectiveness.

**Anthropic — Effective Context Engineering** (2025). Engineering guide
describing best practices for context management in AI agents, including
structured summarization and context pruning. Practical guidance but no
comparative benchmarking of strategies.

**Prompt Compression** (LLMLingua, Microsoft Research, 2024). Achieves up to
20x token compression with ~1.5% performance loss. Focuses on *prompt-level*
compression (removing redundant tokens), orthogonal to *conversation-level*
compaction (summarizing message history).

### 1.3 Gap in existing work

No prior work addresses the **measurement reliability** of compaction
benchmarks. Existing evaluations assume that embedding a fact and asking about
it provides a clean signal — we show this assumption is wrong.

Beyond measurement, no prior work compares fundamentally different compaction
*architectures* (single-shot vs iterative vs frozen summaries) on the same
conversation with spatial recall metrics. Existing benchmarks either test model
capabilities (Needle in a Haystack, Beyond a Million Tokens), measure context
rot without compaction (Chroma), or compare vendor implementations without
varying the underlying strategy (Factory.ai).

The re-summarization cascade problem ("JPEG cascade") is acknowledged
implicitly in MemGPT but never quantified. The frozen summary strategy and
the resulting space-time tradeoff have not been explored.

### 1.4 Contributions

We introduce:

1. **Evidence that recall measurement is non-trivial**: a 75pp batch size
   effect, a category-dependent recall hierarchy, and a systematic gap between
   keyword presence and LLM retrieval
2. **A calibrated benchmark protocol** using LongMemEval evidence with
   controlled density, factorial evidence design, and repeatability validation
3. **Controlled single-pass compaction loss measurement** isolating information
   destruction from context size reduction via re-padding
4. **The attention dilution effect**: compaction degrades recall even in
   untouched portions of the context
5. **The grep-LLM gap**: keywords survive compaction (82–93% grep recall)
   but the LLM fails to retrieve them (0–7% recall) — a "Lost in the Middle"
   amplification effect
6. **(In progress)** Rigorous multi-pass strategy comparison using calibrated
   evidence: Brutal, Incremental, Frozen, and FrozenRanked


## 2. Evidence and Evaluation Framework

### 2.1 Evidence source: LongMemEval

Rather than generating synthetic facts (as in our preliminary v1 experiments),
we use evidence from the LongMemEval benchmark (Wang et al., 2025). LongMemEval
provides 500 question-answer pairs across 6 categories, each grounded in
realistic multi-session conversations between a user and an LLM assistant.

Each fact consists of:
- A **conversation excerpt** (the "evidence") containing the information
- A **question** that can only be answered from the evidence
- An **expected answer** with keywords for automated verification
- A **category** (knowledge-update, single-session-user, single-session-assistant,
  single-session-preference, temporal-reasoning, multi-session)

This provides ecologically valid facts — they are naturally embedded in
conversation, not injected as artificial needles.

### 2.2 Factorial evidence design (2×2)

We vary two dimensions of evidence preparation:

**Filtering** (complete vs filtered):
- *Complete*: all 500 questions included
- *Filtered*: only 234 questions from answerable categories (single-session-user,
  single-session-assistant, knowledge-update, single-session-preference).
  Excludes temporal-reasoning and multi-session, which require capabilities
  beyond single-context recall.

**Truncation** (full vs chopped):
- *Full*: evidence messages preserved in their entirety
- *Chopped*: messages truncated to realistic lengths (simulating context limits
  in real systems)

This yields four experimental modes:

| Mode | Filtering | Truncation | Questions | Max density (facts/kTok) |
|------|-----------|------------|:---------:|:------------------------:|
| R1   | Complete  | Full       | 500       | 0.10                     |
| R2   | Complete  | Chopped    | 500       | 0.42                     |
| R3   | Filtered  | Full       | 234       | 0.16                     |
| R4   | Filtered  | Chopped    | 234       | 0.79                     |

### 2.3 Context construction

For each density level *d*, we embed *d* evidence items into a 190K-token
context. Evidence items are placed at uniformly distributed positions. The
remaining space is filled with realistic padding — real LLM conversation
sessions from the LongMemEval pool (18,255 sessions available) — maintaining
ecological validity rather than using synthetic filler.

Context construction is deterministic (seed=42), ensuring reproducibility
across runs. The padding pool provides ~46M tokens of real conversations.

### 2.4 Density sweep

Rather than testing a single fact density, we sweep across densities from d4
(4 facts per 190K context) to d150 (150 facts, depending on mode). This
reveals saturation effects: at what density does the model's recall capacity
plateau?

### 2.5 Evaluation protocol

**Q&A phase**: For each fact, the LLM (Claude Haiku 4.5) is presented with
the full context as conversation history and asked to recall the specific
information. Questions are batched: for batch size *bs*, *bs* questions are
asked simultaneously. We test bs=1, bs=5, and bs=10.

**Judge phase**: A separate LLM judge compares the model's answer against the
expected answer keywords. Binary recall (correct/incorrect) and accuracy
(correct and precisely matching) are computed.

**Grep validation**: As a free upper bound, we check whether fact keywords
appear verbatim in the context. If grep doesn't find them, the LLM cannot
possibly recall them. The gap between grep recall and LLM recall quantifies
the "present but ignored" phenomenon.

Both Q&A and judge phases use Claude Haiku 4.5 via the Anthropic Batch API.

### 2.6 Repeatability

All key results are validated with 3 independent runs. We report mean ± σ.
For the compaction experiment (Part B), 3 runs on d80/bs=5 yield σ ≤ 2.6pp
across all compaction levels, confirming that observed effects (7–70pp) are
far larger than measurement noise.


## 3. Part A Results: How Hard Is Recall?

### 3.1 The batch size effect

The most surprising finding: asking more questions at once dramatically
improves recall.

![Figure 1 — Recall vs Fact Density](figures/fig1_recall_vs_density.png)
*Figure 1: Recall as a function of fact density (facts/kTok) for three batch sizes.
The secondary axis maps density values to the d** notation used throughout the paper.
Grep upper bound (dashed) shows near-perfect keyword presence regardless of density.*

At d80 (R4 mode), recall ranges from 67.5% (bs=1) to 78.8% (bs=10) — an
11pp gap on the same context with the same facts. At lower densities, the
gap widens further. Across the full density sweep, the bs=1 to bs=10 delta
reaches 75 percentage points at extreme configurations.

**Why this matters for benchmarks**: Any compaction evaluation that uses a
fixed batch size is measuring a confound of retrieval ability and batch
prompting. Results from different batch sizes are not directly comparable.
Our v1 preliminary experiments used bs=10 exclusively — this produced
inflated baselines that masked the true difficulty of recall.

**Mechanism hypothesis**: Multiple questions in a single prompt create
implicit cross-references that help the model locate relevant information.
A question about "Rachel's new city" might prime attention for nearby facts
about "family trip" or "moving costs," improving recall for co-located facts.

### 3.2 Category hierarchy

Not all facts are equally recallable. We identify a clear hierarchy:

![Figure 2 — Recall by Category](figures/fig2_recall_by_category.png)
*Figure 2: Recall breakdown by fact category at d80, for three batch sizes.
User statements and assistant responses are reliably recalled (85–100%),
while preferences remain fragile (20–40%) regardless of batch size.*

| Category                  | Recall range (bs=5) | Notes                |
|---------------------------|:-------------------:|----------------------|
| single-session-user       | 80–100%             | User's own statements, easiest |
| single-session-assistant  | 80–100%             | Assistant's responses |
| knowledge-update          | 60–65%              | Updated information, stable |
| single-session-preference | 20–40%              | User preferences, fragile |
| temporal-reasoning        | max 30%             | Requires inference, near-impossible |
| multi-session             | max 42%             | Cross-session facts, very hard |

The top two categories (user/assistant statements) are near-ceiling. Knowledge
updates plateau at ~60%. Preferences are fragile — the model struggles with
"What is my favorite X?" even when the information is present. Temporal
reasoning and multi-session facts are effectively unmeasurable in a single
context.

**Implication for compaction benchmarks**: Testing on "easy" categories
(single-session) inflates recall and masks real degradation. Testing on "hard"
categories (temporal, multi-session) produces floor effects that make strategies
indistinguishable. The filtered mode (R3/R4) excludes impossible categories
while retaining a mix of easy and hard categories.

### 3.3 The grep-LLM gap: present but ignored

For every fact, we check whether its keywords appear in the context via grep.
Grep recall at d80 is 86%. But LLM recall at bs=1 is only 67.5%.

This 19pp gap — facts that are *verifiably present* in the context but
*not retrieved* by the model — is a direct measurement of the "Lost in the
Middle" phenomenon (Liu et al., 2023) in a realistic conversational setting.

The gap is not uniform across the context: facts in the middle of a 190K
context are more likely to be present-but-ignored than facts near the
beginning or end.

### 3.4 Factorial analysis: filtering vs truncation

The 2×2 factorial design reveals two independent effects:

| Effect      | Magnitude | Mechanism |
|-------------|:---------:|-----------|
| Filtering   | +14–15pp  | Excludes impossible questions, consistent across truncation |
| Truncation  | -13–22pp  | Removes information from evidence messages |
| Composition | +2.5pp    | Removing hard-category evidence barely helps easy-category recall |

The **filtering effect** is remarkably consistent (+14pp and +15pp across
truncation conditions). This means it's driven entirely by excluding
impossible questions, not by freeing context space.

The **composition effect** is near-zero (+2.5pp mean): whether hard-category
evidence occupies context space alongside easy-category facts makes almost
no difference to easy-category recall. This validates using unfiltered
evidence as realistic padding in controlled experiments.


## 4. Controlled Compaction Experiment

### 4.1 Design

Part B isolates the information loss from compaction itself, independent of
context size reduction. The protocol:

1. Start with a calibrated 190K context (from Part A, mode R4)
2. Compact the oldest X% of messages using LLM summarization
3. **Re-pad** to the original 190K size with real conversation sessions
4. Measure recall on the same facts

The re-padding step is critical: it keeps context size constant, so any recall
degradation is attributable to compaction, not to having less context.

### 4.2 Compaction levels

| Level | % compacted | What happens |
|-------|:-----------:|--------------|
| C0    | 0%          | Baseline (no compaction) |
| C1    | 5%          | Minimal compaction (oldest ~36 messages) |
| C2    | 25%         | Moderate (oldest quarter) |
| C3    | 50%         | Half the context compacted |
| C4    | 98%         | Nearly everything compacted (all except last user/assistant exchange) |

For each level, we track which facts fall in the compacted zone vs the
remaining zone, enabling spatial analysis of information loss.

### 4.3 Compaction method

We use single-pass LLM summarization (Claude Haiku 4.5): the oldest X% of
messages is concatenated and sent to the model with instructions to produce
a concise summary. The summary replaces the original messages, freeing
space that is then filled with padding sessions.

The compaction prompt emphasizes preserving factual details, technical
specifications, and decisions — the same prompt used in production compaction
systems.

[Note: In Part C (future), we will compare different multi-pass strategies:
Brutal, Incremental, Frozen, and FrozenRanked. Part B uses single-pass to
isolate the fundamental compaction loss before strategy effects compound.]


## 5. Part B Results: Single-Pass Compaction Loss

### 5.1 Recall degradation is monotone and severe

![Figure 3 — Recall vs Compaction Level](figures/fig3_recall_vs_compaction.png)
*Figure 3: Recall degradation by compaction percentage at bs=5, for three densities.
All curves decline monotonically. The grep upper bound for d80 (dashed) stays above 80%
even at C4, illustrating the grep-LLM divergence.*

Recall degrades monotonically with compaction percentage:

| Level   | d40 (bs=10) | d60 (bs=10) | d80 (bs=10) |
|---------|:-----------:|:-----------:|:-----------:|
| C0      | 62.5%       | 75.0%       | 78.8%       |
| C1 (5%) | 55.0% (-7)  | 68.3% (-7)  | 72.5% (-6)  |
| C2 (25%)| 40.0% (-23) | 50.0% (-25) | 58.8% (-20) |
| C3 (50%)| 25.0% (-38) | 33.3% (-42) | 43.8% (-35) |
| C4 (98%)| 2.5% (-60)  | 10.0% (-65) | 7.5% (-71)  |

Even the lightest compaction (C1, 5%) costs 6–7 percentage points. At C4
(98%), recall drops to single digits — the conversation is effectively
destroyed.

The degradation is consistent across densities: higher-density contexts
have more room to fall but the relative pattern is identical.

### 5.2 The compacted zone is dead

Facts within the compacted region are almost never recalled, even though
their keywords survive in the summary:

| Level | Grep survival (compacted zone) | LLM recall (compacted zone) |
|-------|:------------------------------:|:---------------------------:|
| C1    | 50% (n=2)                      | 0%                          |
| C2    | 82–91%                         | 8–18%                       |
| C3    | 83–93%                         | 0–7%                        |
| C4    | 59–62%                         | 2–4%                        |

This is the most striking result: **grep finds the keywords in the summary,
but the LLM cannot use them to answer questions.** The summary preserves
lexical traces of the facts but destroys the surrounding context that would
enable retrieval. This is the "Lost in the Middle" effect amplified: the
summary sits at the very beginning of the context — the least-attended
position after the primacy window is exhausted.

### 5.3 Attention dilution: compaction damages untouched facts

The remaining zone (facts that were *not* compacted, sitting in their original
messages) also loses recall as compaction increases:

| Level | Remaining zone recall (d80, bs=5) | n_facts |
|-------|:---------------------------------:|:-------:|
| C1    | 73%                               | 78      |
| C2    | 59%                               | 69      |
| C3    | 53%                               | 59      |
| C4    | 80%                               | 5       |

From C1 to C3, remaining-zone recall drops from 73% to 53% — a 20pp loss on
facts that were never touched by compaction. The mechanism: re-padding replaces
the compacted portion with noise (unrelated conversation sessions). This noise
dilutes the model's attention, degrading retrieval even for intact facts.

C4 is an outlier (80%) because only 5 facts remain in the zone — too few for
statistical reliability, but consistent with the model finding a needle among
mostly-padding.

### 5.4 Spatial recall density

![Figure 4 — Spatial Recall Density](figures/fig4_spatial_recall.png)
*Figure 4: Cumulative recalled facts by position in the original 190K context (d80, bs=5).
Each step curve shows one compaction level. Dotted vertical lines mark compaction boundaries.
The C0 baseline exhibits the classic "lost in middle" profile — flat through the central region,
steep at the end (recency). Compacted variants (C3, C4) concentrate their recall
in the surviving portion but lose most facts from the compacted zone.*

This visualization maps each fact to its position in the original (pre-compaction)
context and shows whether it was recalled. It reveals:

- **C0**: Relatively uniform recall across the context, with slight primacy
  and recency effects
- **C1**: A small shadow at the beginning (compacted zone), rest intact
- **C2**: A larger shadow covering the first 25% of the context
- **C3**: The first half is dead, the second half is weakened by attention dilution
- **C4**: Almost complete darkness, with tiny islands of recall at the very end

This provides a visual "damage map" showing exactly where compaction destroys
information.

### 5.5 Repeatability

Three independent runs on d80/bs=5 (the most demanding configuration with
80 facts) confirm stability:

| Level | Run 1  | Run 2  | Run 3  | Mean    | σ     |
|-------|:------:|:------:|:------:|:-------:|:-----:|
| C0    | 73.8%  | 78.8%  | 77.5%  | 76.7%   | ±2.6  |
| C1    | 71.2%  | 68.8%  | 68.8%  | 69.6%   | ±1.4  |
| C2    | 52.5%  | 53.8%  | 52.5%  | 52.9%   | ±0.7  |
| C3    | 40.0%  | 40.0%  | 40.0%  | 40.0%   | ±0.0  |
| C4    | 8.8%   | 6.2%   | 6.2%   | 7.1%    | ±1.5  |

Maximum variance is ±2.6pp (C0 baseline). All compaction effects (−7pp to
−70pp) are far larger than measurement noise. The C3 result is remarkably
stable (40.0% in all three runs), suggesting that at this compaction level,
the outcome is nearly deterministic.


## 6. Part C: Multi-Pass Strategy Comparison (Preview)

Parts A and B established *how* recall works and *what* single-pass compaction
destroys. Part C addresses the core question: when compaction must be applied
repeatedly over a long conversation, **which strategy preserves the most
information?**

### 6.1 Strategies under evaluation

**Brutal (single-shot)**: When context exceeds 90% of max, summarize ALL
messages except the 2 most recent. The input is truncated to a character cap
(~150K real tokens) before summarization.

```
═══════════════════════════════════════════════════════════════════
 BRUTAL
═══════════════════════════════════════════════════════════════════

  BEFORE (at 90%)                    AFTER compact

 200K ┌───────────────────┐  100%   200K ┌───────────────────┐  100%
      │                   │              │                   │
 180K ╞═══ HIGH WM (90%)══╡  ─┐    180K ╞═══ HIGH WM (90%)══╡
      │                   │   │         │                   │
      │  Recent messages  │   │         │                   │
      │                   │   │         │                   │
      │  Raw messages     │   │         │                   │
      │  ~180K tokens     │   │         │    (empty)        │
      │                   │   │         │                   │
      │  Old messages     │   │         │                   │
      │                   │   │         │                   │
 120K ╞═══ LOW WM (60%) ══╡   │    120K ╞═══ LOW WM (60%) ══╡
      │                   │   │         │                   │
      │  Oldest messages  │   │         │                   │
      │                   │   │         │  Last 2 msgs      │
      │                   │   │         │  ~200 tokens      │
   0K └───────────────────┘   0%     0K ├───────────────────┤
                              │         │  Summary (N)      │
                  Summarize ALL ──────► │  ~500 tokens      │
                  (truncate at cap)     └───────────────────┘
```

**Incremental (dual watermark)**: When context exceeds 90%, compact enough old
messages to bring context down to 60%. Previous summaries ARE included in the
text to be re-summarized. Creates a "JPEG cascade" where each cycle degrades
earlier summaries.

```
═══════════════════════════════════════════════════════════════════
 INCREMENTAL
═══════════════════════════════════════════════════════════════════

  BEFORE (at 90%)                    AFTER compact

 200K ┌───────────────────┐  100%   200K ┌───────────────────┐  100%
      │                   │              │                   │
 180K ╞═══ HIGH WM (90%)══╡         180K ╞═══ HIGH WM (90%)══╡
      │                   │              │                   │
      │  Recent messages  │              │                   │
      │  (not compacted)  │              │    (empty)        │
      │                   │              │                   │
 120K ╞═══ LOW WM (60%) ══╡         120K ╞═══ LOW WM (60%) ══╡
      │                   │              │                   │
      │  Raw messages     │    ────────► │  Recent messages  │
      │                   │              │  (kept as-is)     │
      ├───────────────────┤              │  ~118K tokens     │
      │  Summary v.(N-1)  │              ├───────────────────┤
      │  ~2K tokens       │ ───────────► │  Summary v.N      │
   0K └───────────────────┘              │  re-summarized!   │
                                         └───────────────────┘

      Problem: summary v.N = summary of a summary of a summary...
```

**Frozen (dual watermark + immutable summaries)**: Same trigger and target as
incremental, but completed summaries are marked as frozen and never
re-summarized. Only raw (non-frozen) messages are compacted. When frozen
summaries exceed a budget, the oldest are merged.

```
═══════════════════════════════════════════════════════════════════
 FROZEN
═══════════════════════════════════════════════════════════════════

  BEFORE (at 90%)                    AFTER compact

 200K ┌───────────────────┐  100%   200K ┌───────────────────┐  100%
      │                   │              │                   │
 180K ╞═══ HIGH WM (90%)══╡         180K ╞═══ HIGH WM (90%)══╡
      │                   │              │                   │
      │  Recent messages  │              │    (empty)        │
      │  (not compacted)  │              │                   │
      │                   │         120K ╞═══ LOW WM (60%) ══╡
 120K ╞═══ LOW WM (60%) ══╡              │                   │
      │                   │              │  Recent messages  │
      │  Raw messages     │  ──────────► │  (kept as-is)     │
      │                   │              │  ~74K tokens      │
      ├───────────────────┤              ├───────────────────┤
      │  Frozen #(N-1)    │  untouched   │  NEW Frozen #N    │ ◄ new!
  60K ├╌╌╌ BUDGET (30%) ╌╌┤  ─────────►  ├───────────────────┤
      │  Frozen #2        │              │  Frozen #(N-1)    │
      │  Frozen #1        │              │  Frozen #2        │
   0K └───────────────────┘              │  Frozen #1        │
                                      0K └───────────────────┘

      Frozen summaries = never re-summarized.
      Tradeoff: they eat context space for recent messages.
```

**FrozenRanked (hierarchical merge)**: A variant of Frozen where summaries
carry a rank. Only same-rank summaries merge (producing rank+1), limiting
each fact to at most log₂(N) compression passes — compared to N/2 in
sequential merging.

### 6.2 Preliminary results (v1)

Our preliminary experiments (v1) used synthetic facts on 1.5M and 3M-token
conversations. While the evidence was less rigorous than Part A/B (synthetic
facts, no density calibration, bs=10 only), the structural findings are
informative:

| Strategy    | Global 1.5M | Global 3M | Early 1.5M | Early 3M | Profile |
|-------------|:-----------:|:---------:|:----------:|:--------:|---------|
| Brutal      | 12.7%       | 3.3%      | 2%         | 1%       | ▁▁█ present-biased |
| Incremental | 16.0%       | 3.7%      | 2%         | 2%       | ▁▃█ JPEG cascade |
| Frozen      | 16.0%       | 4.7%      | 26%        | 12%      | █▃▂ past-preserving |

Key observations from v1:
- **Frozen preserves early facts** (26% vs 2% at 1.5M) by preventing
  re-summarization
- **JPEG cascade is real**: incremental's accuracy/recall ratio is only 46%
  (facts recalled in degraded form)
- **Rankings are stable across scales**: Frozen > Incremental > Brutal
- **All strategies degrade severely** at 3M (-71% to -77% relative drop)

These results motivate the calibrated experiments in Part C.

### 6.3 Upcoming: calibrated strategy comparison

Using Part A's calibrated evidence (R4 mode, d60 and d80), we will:
- Test Brutal, Incremental, Frozen, and FrozenRanked
- Start with 5M-token conversations (~26 compaction cycles)
- Use bs=5 as the standard evaluation batch size
- Report spatial recall distributions, not just global scores
- Include repeatability validation


## 7. Discussion

### 7.1 Two fundamental failure modes

Our results identify two distinct mechanisms by which information is lost:

**1. Information destroyed (JPEG cascade / compaction loss)**

Part B quantifies this directly: a single compaction pass renders the
compacted zone near-dead (0–7% recall). Keywords survive in the summary
(82–93% grep recall) but are not retrievable. Multi-pass strategies like
Incremental compound this loss across cycles.

**2. Information preserved but ignored (attention dilution)**

Part A's grep-LLM gap (86% grep vs 67% LLM at d80) shows that facts present
in the context are not always found. Part B's remaining-zone degradation
(73% → 53% as compaction increases) shows that injecting noise (re-padding)
dilutes attention even for intact facts.

These failure modes are orthogonal:
- Frozen strategies prevent destruction but may suffer attention dilution
  (too many summaries to read)
- Incremental strategies destroy information but keep the context clean
  (fewer blocks to attend to)
- The optimal strategy must balance both failure modes

### 7.2 The measurement problem

Part A's findings have implications beyond our own benchmark:

**Batch size sensitivity** means that compaction evaluations using different
batch sizes are not comparable. A strategy showing "80% recall" at bs=10 and
another showing "60% recall" at bs=1 may be equally effective.

**Category hierarchy** means that the mix of easy vs hard facts determines
the baseline. Testing compaction only on "single-session-user" facts (80–100%
baseline) will show modest degradation. Testing on "knowledge-update" facts
(60–65% baseline) will show severe degradation from a lower starting point.

**The grep-LLM gap** provides a free diagnostic: if grep finds a fact but the
LLM doesn't recall it, the problem is attention, not information loss. This
distinction is critical for choosing mitigation strategies (restructure
context vs improve compaction).

### 7.3 Why global recall is misleading

A global recall score averages over spatial positions and fact categories. Our
Part A results show this hides structural information:
- Categories range from 100% (single-session-user) to 20% (preferences)
- Spatial position matters: middle facts are harder to recall
- Batch size shifts the entire curve by 10–75pp

Part B amplifies this: compacted-zone recall (0–7%) and remaining-zone recall
(53–73%) tell very different stories that a global 40% would hide.

**Recommendation**: Always report spatially-resolved and category-resolved
metrics alongside global scores.

### 7.4 Implications for production systems

Most production AI assistants (Claude Code, Cursor, Windsurf) use single-shot
or incremental compaction. Our results suggest:

1. **Even light compaction is costly**: 5% compaction costs 6–7pp recall.
   Systems should minimize compaction frequency.
2. **The compacted zone is lost**: Don't expect the summary to be queryable
   for specific facts. If precise recall matters, extract facts to RAG
   *before* compacting.
3. **Re-padding noise hurts**: After compaction frees space, what fills that
   space matters. Random conversation is noise; structured information would
   be better.
4. **Batch size in evaluation matters**: Internal benchmarks should test at
   multiple batch sizes to avoid misleading conclusions.


## 8. Future Work

### 8.1 Multi-pass strategy comparison (Part C)

The immediate next step: rigorous comparison of Brutal, Incremental, Frozen,
and FrozenRanked using calibrated evidence at 5M and 10M tokens. This will
quantify the JPEG cascade, space-time tradeoff, and frozen's scaling limits
under controlled conditions.

### 8.2 FrozenRanked evaluation

The hierarchical merge variant limits each fact to log₂(N) compression
passes. Implementation is ready; benchmarking will be part of Part C.

### 8.3 RAG-augmented compaction

Two-pass compaction: extract structured facts to a vector database + generate
a narrative summary. At query time, combine RAG retrieval with the summary.
Expected to address the "keywords present but not retrievable" gap.

### 8.4 Importance-weighted compaction

Score messages by importance before compaction, preserving high-value
exchanges (decisions, configurations) over low-value ones (acknowledgments).

### 8.5 Cross-model validation

Our results use Claude Haiku 4.5 exclusively. Testing on other models
(GPT-4, Gemini, open-source) would establish whether the findings generalize
or are model-specific.


## 9. Reproducibility

All code and data are available at:
https://github.com/profff/COMPACTION_BENCH

### Scripts
- `benchmark_recall_v5.py` — Recall density sweep (Part A)
- `build_contexts_v5.py` — Context assembly with 4 modes (R1–R4)
- `benchmark_compaction_v5.py` — Controlled compaction loss (Part B)
- `compare_runs_v5.py` — 2×2 factorial analysis
- `benchmark_compaction_v2.py` — Multi-pass strategy benchmark (v1/Part C)
- `compaction.py` — Strategy implementations (Incremental, Frozen, FrozenRanked)

### Dependencies
```bash
pip install anthropic python-dotenv
```

### Quick start
```bash
# Part A: Recall density sweep
./benchmark_recall_v5.py --run R4 --densities 40,60,80 --batch-sizes 1,5,10

# Part B: Controlled compaction loss
./benchmark_compaction_v5.py --run R4 --densities 40,60,80

# Dry run (cost estimate, no API calls)
./benchmark_compaction_v5.py --run R4 --dry-run
```

### Result directories
- `recall_v5_R{1,2,3,4}_*/` — Part A density sweep results
- `compaction_v5_R4_*/` — Part B compaction loss results
- Each directory contains `summary.json`, `answers/`, `judgments/`, `grep/`


## References

1. G. Kamradt, "Needle in a Haystack — LLM Retrieval Test," 2023.
   https://github.com/gkamradt/LLMTest_NeedleInAHaystack

2. C. Packer et al., "MemGPT: Towards LLMs as Operating Systems,"
   arXiv:2310.08560, 2023. https://arxiv.org/abs/2310.08560

3. Chroma Research, "Context Rot: How Increasing Input Tokens Impacts LLM
   Performance," 2025. https://research.trychroma.com/context-rot

4. Factory.ai, "Evaluating Context Compression for AI Agents," 2025.
   https://factory.ai/news/evaluating-compression

5. Factory.ai, "Compressing Context," 2025.
   https://factory.ai/news/compressing-context

6. "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in
   LLMs," arXiv:2510.27246, 2025. https://arxiv.org/abs/2510.27246

7. Anthropic, "Effective Context Engineering for AI Agents," 2025.
   https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

8. H. Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference
   of Large Language Models," Microsoft Research, 2024.

9. Letta, "Benchmarking AI Agent Memory: Is a Filesystem All You Need?," 2025.
   https://www.letta.com/blog/benchmarking-ai-agent-memory

10. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long
    Contexts," arXiv:2307.03172, 2023.

11. D. Wang et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
    Interactive Memory," arXiv:2410.10813, 2024.
    https://github.com/xiaowu0162/LongMemEval


## Authors

Olivier Gasté — conception, implementation, benchmark design

---

*Draft v2 — restructured paper. Parts A & B complete with data. Part C in progress.
Graphs pending (4 figures). v1 results preserved as preliminary evidence in §6.2.*

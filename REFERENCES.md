# References — Compaction Benchmark & Recall Measurement

## Compaction & Context Management

- **Needle in a Haystack** — Greg Kamradt, 2023
  Single needle retrieval in long context. Foundational test for long-context recall.
  https://github.com/gkamradt/LLMTest_NeedleInAHaystack

- **MemGPT / Letta** — Packer et al., 2023
  Virtual context management (OS-inspired: RAM/disk). Recursive summarization of evicted messages.
  https://arxiv.org/abs/2310.08560

- **LLMLingua: Prompt Compression** — Jiang et al., Microsoft Research, 2024
  Up to 20x token compression with ~1.5% perf loss. Prompt-level, not conversation-level.
  https://arxiv.org/abs/2310.05736

- **Context Rot** — Chroma Research, 2025
  LLM perf degrades with input length even on trivial tasks. 18 models tested.
  https://research.trychroma.com/context-rot

- **Factory.ai — Evaluating Context Compression** — 2025
  Compares 3 compression implementations (Factory, OpenAI Compact, Anthropic SDK) on 36K messages.
  https://www.factory.ai/blog/evaluating-context-compression

- **Beyond a Million Tokens** — 2025
  Long-term memory benchmark up to 10M tokens. Recall, multi-hop, contradiction, temporal ordering.

- **Anthropic — Effective Context Engineering** — 2025
  Best practices guide for context management in AI agents.
  https://docs.anthropic.com/en/docs/build-with-claude/context-engineering

## Long-Context Evaluation Benchmarks

- **Lost in the Middle** — Liu et al., 2024 (NeurIPS)
  Multi-doc QA varying position of relevant doc. U-shaped recall curve (primacy/recency).
  **1 question per test.** Foundational "Lost in the Middle" effect.
  https://arxiv.org/abs/2307.03172

- **RULER** — Hsieh et al., 2024 (Google/CMU)
  Multi-key NIAH variants (MK, MV, MQ) + variable tracking + common words extraction.
  4 tasks, 500 examples/length, greedy decoding, string-match scoring, 10+ models.
  Claimed context lengths overestimate effective lengths (GPT-4: 128K claimed → 64K effective).
  MQ-NIAH (4 distinct keys, retrieve all at once) = closest to our multi-fact setup.
  **1 query per test.** No variance/repeatability analysis.
  https://arxiv.org/abs/2404.06654

- **NeedleBench** — OpenCompass, 2024
  Progressive difficulty: S-RT (single needle), M-RT (multi-needle), M-RS (multi-needle reasoning),
  ATC (Ancestral Trace Challenge). Bilingual, 4K to 1M+ tokens. **1 question per test.**
  https://arxiv.org/abs/2407.11963

- **HELMET: How to Evaluate Long-Context Models Effectively and Thoroughly** — Yen et al., 2025 (Princeton, ICLR)
  7 task categories (RAG, citations, re-ranking, few-shot ICL, long-doc QA, summarization,
  synthetic recall). GPT-4o as judge (0-3 scale, Cohen's kappa=0.91 vs humans). 59 models.
  100-600 examples/dataset, 5 lengths (8K-128K), greedy decoding, single run.
  **Key finding: NIAH does NOT predict downstream performance** (no synthetic task
  achieves avg Spearman >0.8 with downstream). RAG tasks best proxy for downstream.
  No variance/repeatability analysis. No discussion of questions-per-test effect.
  https://arxiv.org/abs/2410.02694

- **Summary of a Haystack (SummHay)** — Laban et al., 2024 (EMNLP, Salesforce)
  100 docs (~100K tokens), summary + citation evaluation (Coverage + Citation scores).
  Not Q&A-based — evaluates generated summaries. 10 LLMs + 50 RAG systems.
  https://arxiv.org/abs/2407.01370

- **Gemini 1.5 Pro — Multi-Needle Haystack** — Google, 2024
  100 needles across up to 1M tokens, "retrieve all" in 1 turn.
  ~70% recall at 128K, >60% at 1M (single needle: >99.7%).
  https://arxiv.org/abs/2403.05530

- **BABILong** — Kuratov et al., 2024
  bAbI reasoning tasks (20 types) embedded in long context. **1 question per test.**
  https://arxiv.org/abs/2406.10149

- **LongBench** — Bai et al., 2024
  21 tasks across 6 categories (single-doc QA, multi-doc QA, summarization, etc.).
  https://arxiv.org/abs/2308.14508

- **InfiniteBench** — Zhang et al., 2024
  100K+ token benchmarks: novel retrieval, QA over books, math, code.
  https://arxiv.org/abs/2402.13718

## Measurement Methodology & Artifacts

- **Ask Me Anything (AMA)** — Arora et al., 2023 (Stanford, ICLR Oral)
  Multiple question reformulations (num_boost=5) + weak supervision aggregation
  → +10.2% accuracy over few-shot across 20 benchmarks. GPT-J-6B beats GPT-3-175B
  on 15/20 tasks. Shows question FORMAT matters (open-ended QA > restrictive prompts).
  **Closest conceptual precedent** to our batch size finding: prompt formulation affects
  results dramatically. But operates in few-shot regime (short context), not long-context
  recall. Does not test multiple-questions-per-prompt vs single-question.
  https://arxiv.org/abs/2210.02441

- **LLM Task Interference** — 2024 (EMNLP)
  Task-switching in conversational history degrades performance.
  Could explain why bs=15 < bs=5: too many simultaneous questions = interference.
  https://arxiv.org/abs/2402.18216

- **Context Length Alone Hurts LLM Performance Despite Perfect Retrieval** — 2025
  Performance decreases with context length even when retrieval is perfect.
  https://arxiv.org/abs/2510.05381

- **HELM Long Context** — Stanford CRFM, 2025
  Holistic evaluation of long-context models with standardized scenarios.
  https://crfm.stanford.edu/2025/09/29/helm-long-context.html

## Our Findings (not yet in literature)

- **Batch size effect on recall measurement** (this work, 2026)
  Testing 4 batch sizes (1, 5, 10, 15) × 5 fact densities at 190K tokens (Haiku 4.5).
  3 runs on identical contexts for repeatability analysis.
  Max recall difference: **75pp** between batch sizes on same context (d8: bs=1=25% vs bs=10=100%).
  Inter-run Jaccard=0.658 (repeatable), inter-bs Jaccard=0.402 (batch size = real variable).
  Signal/noise ratio=1.6x. d4/d8 stable (J=0.83-0.94), d50/d100 noisy (J=0.42-0.53).
  Grep finds 100% keywords but LLM recalls only ~25% (Lost in Middle gap 73%).
  **No prior work found studying this specific measurement artifact.**
  All major benchmarks (NIAH, RULER, HELMET, NeedleBench, BABILong) use 1 question/test.
  AMA (Arora 2023) is closest conceptual precedent but in few-shot, not long-context.

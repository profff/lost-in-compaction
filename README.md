# Lost in Compaction — Measuring Information Loss in LLM Context Summaries

How much do LLMs forget when they summarize their own conversation history?

## What is this?

Long-running LLM sessions hit context window limits. When that happens, the model (or a smaller model) summarizes older messages to free up space. **How much information is lost in the process?**

This benchmark injects known facts into realistic conversation contexts and measures what the LLM can recall before and after compaction — with a calibrated instrument that accounts for the LLM's own retrieval limitations.

## Key findings

| Metric | Value |
|--------|-------|
| Baseline recall (d80, bs=5, 190K) | 73% |
| After 50% compaction (C3) | 40% (−33pp) |
| After 98% compaction (C4) | 7% (−66pp) |
| Grep keyword survival at C4 | 82% |
| Repeatability (σ across 3 runs) | ≤ 2.6pp |

**Two failure modes discovered:**
1. **Semantic erasure** — facts in the compacted zone are rephrased into generic summaries (0–7% LLM recall despite 82–93% keyword survival)
2. **Attention dilution** — remaining (untouched) facts lose recall too, dropping from 73% to 53% as compaction increases

## Paper

The full analysis is in [`LOST_IN_COMPACTION.md`](LOST_IN_COMPACTION.md)
(also available as `LOST_IN_COMPACTION.pdf`).

## Project structure

```
├── benchmark_recall_v5.py        # Part A: baseline recall measurement
├── benchmark_compaction_v5.py    # Part B: controlled compaction experiment
├── build_contexts_v5.py          # Context builder (facts + padding)
├── extract_evidence.py           # LongMemEval evidence extraction
├── plot_compaction_v5.py         # Publication figures (4 graphs)
├── figures/                      # Generated PNG figures
├── recall_v5_R4_*/               # Baseline recall run results
├── compaction_v5_R4_*/           # Compaction experiment results
├── LOST_IN_COMPACTION.md         # Paper (current version)
├── LOST_IN_COMPACTION.pdf        # Paper (PDF)
├── BENCHMARK_ANALYSIS.md         # Paper (v1, archived)
├── PLAN_*.md                     # Design documents
└── *_v4.py / recall_v4_*/        # Earlier v4 experiments (historical)
```

## Methodology

### Part A — Recall calibration

Before measuring compaction loss, we calibrate the measurement instrument itself:
- **Fact density** (0.02–0.79 facts/kTok): sparser facts are harder to find
- **Batch size** (1/5/10 questions): asking more questions at once improves recall by up to 75pp
- **Fact categories**: user statements (95%) > assistant responses (90%) > knowledge updates (60%) > preferences (20%)
- **Grep-LLM gap**: keyword search finds nearly everything; the LLM's semantic retrieval is the bottleneck

### Part B — Controlled compaction

Five compaction levels (C0=0%, C1=5%, C2=25%, C3=50%, C4=98%) applied to the oldest portion of a 190K-token context, then re-padded to constant size. This isolates compaction damage from context-length effects.

### Evaluation pipeline

1. Build contexts with injected facts from LongMemEval
2. Ask questions via Anthropic Batch API (Haiku)
3. Judge answers via LLM (also Haiku, batch)
4. Cross-validate with grep keyword search

## Reproducing

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key

# Part A: baseline recall sweep
python benchmark_recall_v5.py --mode R4 --densities 40 60 80 --batch-sizes 1 5 10

# Part B: compaction experiment
python benchmark_compaction_v5.py --mode R4 --densities 40 60 80

# Figures
python plot_compaction_v5.py --save --dpi 300
```

## Data sources

- Conversation padding: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) sessions
- Facts: extracted from LongMemEval evidence fields

## Cost

All experiments use Claude Haiku via the Anthropic Batch API. A full compaction run (3 densities × 5 levels × 3 batch sizes) costs approximately $5.50.

The full benchmark suite — calibration, single-pass compaction, the four-strategy comparison with 4–6 replicates per cell, judge re-runs, and a few rough nights of debugging — has cost about **1,800 €** in API credit so far. A breakdown is in §10 of the paper.

## Support & sponsorship

This is independent, unfunded research, run from a home machine with a personal API key. The remaining experiments on the wishlist (multi-seed generalization, a QA model with a 1M context window so the seed-99 run actually works, a second human-judge calibration sample, cross-architecture validation against models from other providers) are blocked mostly by budget, not by code.

If you work at Anthropic, OpenAI, Google DeepMind, Mistral, or anywhere else with a researcher-credit programme and you'd like to see how your own context-compaction stack behaves under this benchmark — get in touch. The codebase is provider-agnostic via `llm_backend.py`, so adding a model is a matter of credentials and a few hundred euros of credit. In return: published numbers, full methodology, no marketing spin.

Tea, coffee, beer, or just a star on the repo also accepted.

Contact: ogaste@gmail.com

## License

Research use. Dataset derived from LongMemEval (MIT license).

# arXiv submission metadata — Lost in Compaction

Copy-paste-ready content for the arXiv submission form. Fill the matching field on https://arxiv.org/submit when the LaTeX upload step is done.

## Title

Lost in Compaction: Measuring Information Loss in LLM Context Summaries

## Authors

Olivier Gasté (https://orcid.org/0009-0003-3853-9298)

## Abstract

(Plain text, no LaTeX commands, no markdown. Paste in the abstract field as-is.)

```
Long-running LLM conversations inevitably exceed the context window, forcing systems to compact old messages through summarization. But how do we know whether our compaction benchmark accurately measures information loss? Before comparing strategies, we need to understand how reliably an LLM can recall facts from its own context -- and what factors affect that recall.

We present a three-phase study. First, we calibrate a recall benchmark using 234 naturally-embedded facts from the LongMemEval dataset across a 190K-token context. We discover four measurement pitfalls that affect all compaction benchmarks: (1) a questions-per-prompt effect where Q=10 yields 11pp higher recall than Q=1 in static contexts -- but the effect reverses under severe compaction, with Q=1 outperforming Q=5 by 9pp, (2) a category hierarchy where temporal reasoning (30% max) and preference recall (40% max) are near-impossible regardless of strategy, (3) a density saturation where recall plateaus beyond ~0.4 facts/kTok despite increasing evidence, and (4) a grep-LLM gap where keyword search finds 86% of facts but the LLM only recalls 25-79% of them -- information is present in the context but ignored by attention.

Second, we isolate compaction loss by compacting 5-98% of a context, re-padding to the original size, and measuring recall degradation. Even 5% compaction costs 7 percentage points of recall. At 50%, the compacted zone is near-dead (0-7% recall) despite keywords surviving at 82-93% via grep. Critically, compaction damages even the untouched portion of the context: remaining-zone recall drops from 68% to 39% as compaction increases -- an attention dilution effect caused by injecting noise (re-padding) into the context. Cross-model validation with Claude Sonnet 4.6 (92.5% baseline recall, no Lost-in-the-Middle effect) confirms that severe compaction destroys information regardless of model capability: even a model with flat spatial recall drops to 21% at 98% compaction.

Third, we compare four multi-pass compaction strategies (Brutal, Incremental, Frozen, FrozenRanked) on a single 5M-token conversation evaluated at five mid-feed checkpoints (500K to 5M) with constant fact density and 4-6 replicates per cell. The strategy hierarchy is consistent in the means: FrozenRanked > Frozen > Incremental > Brutal. All strategies degrade severely with scale (Frozen drops from 14.9% at 500K to 3.0% at 5M). With replicates we also document a substantial run-to-run variance: the compaction phase itself is non-deterministic at temperature zero and recall measurements on identical conversations span up to a factor of 14x (e.g. S4 at 1M: 2.6%-35.9% across replicates). Single-shot benchmarks of compaction strategies are therefore unreliable; replicates are mandatory.

The bottleneck is attention capacity, not compression quality: keywords survive summarization but the LLM cannot retrieve them, and adding more preserved summaries dilutes attention rather than helping.

All experiments use Anthropic Claude models (Haiku 4.5 and Sonnet 4.6). The methodological findings (Q-effect, judge-prompt sensitivity, run-to-run variance) are likely model-agnostic, but the absolute recall numbers and the strategy ordering should be re-validated on non-Claude models before generalising.
```

## Comments (free-text field on arXiv)

```
~28 pages, 11 figures, 15 tables. Working paper. Code and data: https://github.com/profff/lost-in-compaction
```

## Subject classes

- **Primary**: `cs.CL` — Computation and Language
- **Cross-list (secondary)**: `cs.LG` — Machine Learning
- **Cross-list (secondary, optional)**: `cs.AI` — Artificial Intelligence

## MSC class / ACM class

Leave blank. Optional for cs.CL submissions and not relevant here.

## License

`CC BY 4.0` — Creative Commons Attribution 4.0 International. Standard for open research, allows reuse with attribution.

(arXiv also offers `CC BY-NC-SA` for non-commercial; pick CC BY 4.0 unless you have a reason to restrict.)

## Endorsement code

Once arXiv generates the endorsement code (My Account → Endorsement Request → cs.CL), paste it here so it's logged with the rest of the submission metadata:

```
[FILL ME IN ONCE GENERATED — share this with the endorser, e.g. Gerard Burnside]
```

## Submission checklist (run before clicking Submit)

- [ ] PDF compiles cleanly on Overleaf (no error in the log)
- [ ] No `??` markers in the PDF (unresolved \ref or \cite)
- [ ] Bibliography section visible at the end with 11 entries
- [ ] All 11 figures visible
- [ ] Author name + ORCID + email correct on title page
- [ ] arXiv source tar.gz built via Overleaf Menu -> Submit -> Download Source
- [ ] Endorsement received (otherwise submission stays in queue)
- [ ] Primary cs.CL set, cross-listed cs.LG (and cs.AI if you want)
- [ ] License chosen (CC BY 4.0)

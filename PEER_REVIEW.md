# Peer Review: "Lost in Compaction"

> Independent critical review by an external reviewer, generated 2026-05-03.
> The reviewer was given the full paper text and asked to act as a workshop /
> arXiv pre-print peer reviewer — honest, constructive, prioritized.

## 1. Summary (3 sentences)

The paper measures information loss in LLM context summarization through
three experiments: a calibration of the recall benchmark itself (uncovering
questions-per-prompt, category, density, and grep-LLM confounds), a
single-pass compaction study with re-padding to isolate destruction from
size reduction, and a replicated four-strategy comparison on a single
5M-token conversation at constant fact density. The headline empirical
results are that even 5% compaction costs ~7pp recall, that the compacted
zone is "dead" (0–7%) despite 82–93% keyword survival, and that
FrozenRanked > Frozen > Incremental > Brutal in mean recall but with
run-to-run variance up to 14× — making single-shot benchmarks unreliable.
The framing argues the bottleneck is attention dilution rather than
compression quality, and that architectural optimization yields diminishing
returns at scale.

## 2. Strengths

- **The questions-per-prompt finding is genuinely novel and well-motivated.**
  The Q-effect *reversal* under compaction (§3.1, §5.6) is a real
  contribution: it invalidates many existing benchmarks. Cite this
  front-and-center.
- **The grep-LLM gap framing is sharp** and provides a free upper bound
  that other benchmarks ignore.
- **The replicated §6 design is methodologically honest.** Reporting
  variance and explicitly stating "single measurements untrustworthy"
  raises the bar for the field.
- **Cross-model validation with Sonnet 4.6** (§5.7) materially strengthens
  the destruction-vs-attention argument.
- **Spatial/zone metrics** (compacted vs remaining, §5.2–5.3) reveal
  structure global recall hides.
- **Reproducibility section is concrete** (scripts named, seeds given,
  code/data link present).

## 3. Weaknesses (prioritized)

### Blockers for journal, fixable for workshop / arXiv

1. **Single conversation, single seed in §6.** All replicates share seed=42
   and one 5M conversation. "Run-to-run variance" measures only LLM-side
   stochasticity, not conversation-level generalization. A reader cannot
   tell if S4 > S3 holds on a different conversation. At minimum, run 2
   conversations with different seeds.
2. **n=4–6 is small for the variance claims being made.** With σ=14% on
   S3@1M, the 95% CI is roughly ±15pp — the S4–S3 hierarchy at 1M is
   *not* statistically supported, despite the paper saying "the strict
   ordering is observed." Either drop the strict-ordering claim, or run a
   paired test (Wilcoxon signed-rank across paired runs) and report
   p-values.
3. **The "non-monotonic dip at 2M" (§6.3 finding 3)** is asserted as "one
   of the most surprising findings" but rests on n≈6 with overlapping
   error bars. This may be artifact, not signal. Currently overclaimed.
4. **§7.6 predictive modelling has a methodological problem.** The 4,812
   observations are not independent (same context, same facts, repeated
   questions). Logistic regression p-values assume independence; clustered
   SEs or mixed-effects are required. As-is, p<0.001 is unreliable. The
   XGBoost-vs-logistic AUC comparison is also unfair without matched
   cross-validation folds.

### Important but fixable

5. **Claude-only.** Title says "LLM" but every measurement is on Claude
   4.5/4.6. §8.7 acknowledges this, but the abstract should too.
6. **Re-padding "noise" is a confound, not just attention dilution.**
   §5.3 attributes remaining-zone loss to "attention dilution" but the
   re-padding inserts *real* conversations that may semantically
   interfere with the test facts. A control with semantically null
   padding (e.g., random tokens, repeated boilerplate) would isolate
   dilution from interference.
7. **Section 5.7 Haiku C0 numbers don't match §3.** §3 reports δ=0.42 /
   Q=1 = 67.5%; §5.7 also says 67.5% — fine — but §5.5 reports C0
   mean=76.7%. Reader must reconstruct that §5.5 is Q=5 and §5.7 is Q=1.
   Make this explicit in every table caption.
8. **No human-judge calibration.** The "lenient vs strict" judge gap is
   5–15pp — larger than several reported strategy gaps. Without a
   human-graded subsample (even n=50), we don't know which judge is
   closer to ground truth.
9. **Compaction prompt is described as "the same prompt used in production
   compaction systems"** (§4.3) but not shown. This is a critical
   experimental parameter — include verbatim in an appendix.

## 4. Suggestions for improvement (impact-ordered)

1. **Run 2 additional 5M conversations with different seeds**, even at n=2
   each. This is the cheapest way to address the generalizability blocker.
2. **Add paired statistical tests** (Wilcoxon or paired-bootstrap CI) to
   §6.3 table; replace "S4 > S3 at 500K, 2M, 3.5M" with explicit p-values
   or "n.s." labels where appropriate.
3. **Add a 50-item human-judged sample** to validate the strict judge.
4. **Move §7.6 predictive modelling to an appendix** or rewrite with
   mixed-effects (random intercept per fact). The current logistic
   regression is overstated.
5. **Add the compaction and judge prompts verbatim** as appendices.
6. **Soften the "non-monotonic dip at 2M" claim** to "tentatively
   observed" pending more conversations.
7. **Add a "noise-control" arm to §5.3** (re-pad with low-information
   filler) to disentangle dilution from interference.

## 5. Specific issues

- **§3.3** claims "facts in the middle of a 190K context are more likely
  to be present-but-ignored" — figure not cited and no statistic given.
  Either cite Figure 4 or remove.
- **§5.1 table** column headers say "(Q=10)" but caption of Fig 3 says
  Q=5. Inconsistency — please check.
- **§5.4 caption** says "Q=5" but table 5.1 above is "Q=10." Confusing.
- **§6.1 ASCII diagrams** are charming but won't render in PDF / arXiv.
  Replace with proper figures or move to appendix.
- **Reference 11**: LongMemEval is dated 2024 in references but cited as
  "Wang et al., 2025" in §1.2 and §2.1. Pick one.
- **Reference 5 vs 4**: two Factory.ai posts cited but only one URL in
  §1.2 prose ("Evaluating Context Compression"). Clarify which is "the
  closest work."
- **§6.3 finding 4**: "S1 stays under 10% at every scale" — but §5.1
  baseline (C0) is 67.5% Haiku Q=1. The reader needs an explicit
  statement that §6 uses Sonnet QA, not Haiku, to understand why
  absolute numbers differ from §5.
- **Repo URL** `github.com/profff/lost-in-compaction` — verify it's public
  and contains the listed scripts before posting.
- **Abstract**: "factor of 14×" appears without context; specify it's
  S4@1M ranging 2.6%–35.9%.
- **Density confound (§6.2)**: the old §6 design had decreasing δ; new
  design fixes δ at 0.04. But 0.04 is far below the §3 sweet spot (0.42).
  The paper should acknowledge that absolute recall in §6 is limited
  partly by *low* density, not just compaction.
- **§7.5**: "−5pp at 5M (likely noise)" — this contradicts §6.3 which
  reports +1.6pp at 5M. Numbers conflict.
- **Term "JPEG cascade"** is used 5+ times but never defined formally;
  first use in §6.1 should add a one-line definition.

## 6. Final verdict

- **Workshop / arXiv pre-print: ACCEPT with minor revisions.** The
  Q-effect reversal, grep-LLM gap, and variance findings are publishable
  contributions and the methodology is honest enough that the community
  can build on it. Fix the inconsistent Q labels, clarify Haiku vs Sonnet
  in tables, soften the 2M-dip claim, and add the prompts to an appendix.
- **Peer-reviewed venue (EMNLP findings, COLM, TMLR): MAJOR REVISION.**
  Needs: (a) >1 conversation seed in §6, (b) statistical tests for the
  strategy hierarchy, (c) clustered / mixed-effects re-fit of §7.6 or
  remove it, (d) human-judge calibration sample, (e) Claude-only caveat
  in abstract.
- **Top-tier venue (NeurIPS, ICLR): REJECT in current form** —
  single-conversation §6, no significance testing, single-vendor model
  coverage.

> Olivier, this is solid work and the framing is much sharper than the
> previous version — the §6 rewrite around constant density and replicates
> was the right call. The two highest-leverage fixes before posting are
> (1) one or two extra conversation seeds in §6 and (2) explicit paired
> tests with p-values in the §6 table. Everything else is polish.

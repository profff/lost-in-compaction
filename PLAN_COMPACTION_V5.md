# Plan : benchmark_compaction_v5.py

## Contexte
Les 4 runs v5 (R1-R4) donnent le recall baseline a differentes densites dans des contextes de 190K tokens LongMemEval. Il faut maintenant mesurer la **perte d'information due a la compaction elle-meme** a niveaux controles, en gardant la taille constante a 190K (design du Stage 2 de benchmark_batch_meta.py, adapte aux evidence LME).

## Flow global

```
v5 context (190K) -> compact_portion(X%) -> contexte reduit
                                              |
                                              +-> re-pad a 190K (padding reel du pool)
                                              |
                                              +-> contexte compacte (190K)
                                                    |-> grep (gratuit)
                                                    |-> Q&A batch API
                                                    |-> judge batch API
                                                    |-> metrics + delta vs C0
```

## Niveaux de compaction

```python
COMPACTION_LEVELS = {
    "C0": 0.00,  # baseline = reference v5 (pas re-run)
    "C1": 0.05,  # compact oldest 5%
    "C2": 0.25,  # compact oldest 25%
    "C3": 0.50,  # compact oldest 50%
    "C4": 0.98,  # compact quasi-tout
}
```

## Phases

### Phase 0 — Build compacted contexts (sync, N appels LLM)
Pour chaque (densite, cLevel) :
1. `load_context(runMode, density, seed)` — charge le contexte v5
2. `compact_portion(messages, fraction, llm, system)` — copie de benchmark_batch_meta.py:299-377
3. Re-pad a 190K :
   - Charger le padding pool (18K sessions reelles)
   - Seed deterministe : `seed + LEVEL_SEED_OFFSET[cLevel]` (pas hash())
   - `select_padding()` pour combler le deficit
   - Append des sessions en fin de contexte
4. Mettre a jour les metadata :
   - Facts avec `message_start < nToCompact` → `compacted=True`
   - Facts restants → shift indices de `-(nToCompact - 2)`
   - Recalculer `position_pct`
5. Sauvegarder contexte + meta (pour --skip-compact)

### Phase 1 — Grep scan (gratuit, local)
`grep_keywords()` sur chaque contexte compacte. Les keywords des faits compactes peuvent survivre ou non dans le resume — c'est LA mesure cle.

### Phase 2 — Q&A via Batch API
Identique a benchmark_recall_v5.py. custom_id : `qa_d{N}_{cLevel}_bs{B}_b{I}`. Submit par chunks de 20.

### Phase 3 — Judge via Batch API
Identique a benchmark_recall_v5.py. Verdicts recalled/accurate par fait.

### Phase 4 — Metrics + summary
- Metrics standards : recall, accuracy, early/mid/late, by_category
- **Nouveau** : `recall_compacted_zone` (faits dans la zone compactee), `recall_remaining_zone`
- Delta vs C0 : charge le baseline depuis `recall_v5_{run}_*/summary.json` le plus recent
- summary.json avec C0 de reference + C1-C4

## CLI

```bash
./benchmark_compaction_v5.py --run R4 --densities 40,60,80 --dry-run
./benchmark_compaction_v5.py --run R4 --densities 40,60,80 --grep-only
./benchmark_compaction_v5.py --run R4 --densities 40,60,80
./benchmark_compaction_v5.py --run R4 --densities 40,60,80 --skip-compact
./benchmark_compaction_v5.py --run R4 --densities 40 --levels C1,C2
```

Defaults : `--batch-sizes 1,5,10`, `--levels C1,C2,C3,C4`, `--model claude-haiku-4-5-20251001`, `--seed 42`

## Output

```
compaction_v5_{run}_{timestamp}/
├── config.json
├── contexts/
│   ├── d40_C1.json + d40_C1_meta.json
│   ├── d40_C2.json + d40_C2_meta.json
│   └── ...
├── grep/
│   └── d40_C{x}.json
├── answers/
│   └── d40_C{x}_bs{y}.json
├── judgments/
│   └── d40_C{x}_bs{y}.json
└── summary.json
```

## Imports vs copie

**Importer** (modules stables) :
- `compaction.py` : `estimate_tokens`, `messages_to_text`, `COMPACT_SYSTEM`, `COMPACT_PROMPT`
- `benchmark_compaction_v2.py` : `RateLimitedLLM`

**Copier/adapter** (independance du script) :
- `compact_portion()` depuis benchmark_batch_meta.py:299-377
- `load_context()`, `extract_facts()`, `grep_keywords()` depuis benchmark_recall_v5.py
- `submit_chunked()`, `wait_for_batch()`, `parse_llm_json()` depuis benchmark_recall_v5.py
- `compute_metrics()` depuis benchmark_recall_v5.py — etendu avec zone compaction
- `load_padding_pool()`, `select_padding()`, `estimate_tokens_chars()` depuis build_contexts_v5.py
- Prompts (SYSTEM_PROMPT, BATCH_QUESTION_PROMPT, BATCH_JUDGE_PROMPT, JUDGE_SYSTEM)

## Points d'attention

1. **C0 pas re-run** : reference les resultats v5 existants dans summary.json
2. **Seed deterministe** : `LEVEL_SEED_OFFSET = {"C1": 5001, "C2": 5002, "C3": 5003, "C4": 5004}`
3. **Troncature compact_portion** : a C4 (98%), le texte sera tronque a 500K chars — normal, c'est le worst-case
4. **count_tokens API** : un appel par contexte pour metadata exacte (gratuit)
5. **Memoire** : traiter un (density, level) a la fois en Phase 0
6. **Estimation ~810 lignes** total

## Verification

1. `--dry-run` : verifie chargement contextes, estime cout, pas d'appels API
2. `--grep-only` : valide Phase 0 (compaction + repad) + Phase 1 (grep) — gratuit sauf les appels compaction sync
3. Run complet sur une seule config : `--densities 40 --levels C2 --batch-sizes 1`
4. Verifier que summary.json inclut C0 baseline et que les deltas sont coherents

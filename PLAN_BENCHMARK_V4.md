# Plan Benchmark v4 — Realistic Conversations

> Refonte des données de test : passer de conversations synthétiques
> (padding repetitif + one-liner facts) a des conversations realistes
> (LongMemEval padding + evidence sessions naturelles).

## Probleme identifie (v3)

Notre padding synthetique biaise la mesure :
- 54x "Got it, noted", 56x "Agreed, option A sounds right"
- Les faits sont des one-liners injectes brutalement (rupture d'attention)
- Le pseudo-code est du `process(input[N])` en boucle
- Impossible de savoir si le recall bas vient du Lost in Middle
  ou de la mauvaise qualite du contexte

## Source de donnees : LongMemEval (ICLR 2025)

- **18 255 sessions de padding** uniques (ShareGPT + UltraChat) = ~61M tokens
- **948 evidence sessions** (500 questions) = faits naturels dans le flow
- Sessions evidence : median 12 turns, ~4.7K tokens
- Conversations realistes, peer-reviewed, reproductibles

## Architecture donnees

```
COMPACT_BENCHMARK/
├── data/                          ← NOUVEAU, donnees fixes
│   ├── padding_pool.jsonl         ← 18K sessions extraites (id, turns, tokens)
│   ├── evidence_longmemeval.json  ← 500 entries {question, answer, keywords,
│   │                                 session_turns, source: "longmemeval"}
│   ├── evidence_synthetic.json    ← N entries generees {idem, source: "synthetic"}
│   └── contexts/                  ← contextes assembles, deterministes
│       ├── recall_190K/           ← pour bench recall
│       │   ├── d4_seed42.json
│       │   ├── d8_seed42.json
│       │   ├── d19_seed42.json
│       │   ├── d39_seed42.json   (max LongMemEval natif a 190K)
│       │   ├── d50_seed42.json   (mix natif + synthetique)
│       │   └── d100_seed42.json
│       └── compact_progressive/   ← pour bench compact (chunks croissants)
│           └── conv_A/            ← une conversation, decoupee en tranches
│               ├── slice_10K.json
│               ├── slice_50K.json
│               ├── slice_190K.json
│               └── slice_1M.json
├── old_results/                   ← anciens resultats (v3, padding synthetique)
└── [scripts existants]
```

## Lecons du bench v3 (a documenter dans l'article)

Le bench v3 (recall measurement) a produit des resultats exploitables
mais a revele des faiblesses methodologiques qu'on ne peut pas ignorer :

### Ce qu'on garde (findings valides)
- **Batch size effect** : 75pp de delta entre bs=1 et bs=10 sur memes contextes.
  Reproductible (3 runs, Jaccard inter-run 0.658). Aucun benchmark publie ne
  teste cet artefact — c'est une contribution originale.
- **Lost in Middle gap** : grep=100% vs LLM recall=25%. Confirme la litterature.
- **NIAH est insuffisant** : coherent avec HELMET (Yen 2025) et RULER (Hsieh 2024).

### Ce qu'on abandonne (donnees non fiables pour le bench compact)
- **Padding synthetique** : 54x "Got it, noted", pseudo-code `process(input[N])`.
  Le recall bas (~25%) est potentiellement biaise par la qualite du padding,
  pas uniquement par le Lost in the Middle.
- **Injection one-liner** : les faits sont des ruptures de contexte evidentes.
  Le modele pourrait sur-attentionner (saillance anomalique) ou sous-attentionner
  (hors-sujet). On ne sait pas lequel.
- **Conversations non realistes** : 149 messages user uniques sur 667. Aucune
  conversation reelle n'a ce niveau de repetition.

### Pourquoi v4
On ne peut pas construire un bench de compaction fiable sur des donnees
dont on n'a pas confiance dans le recall de base. D'ou le passage a des
conversations realistes (LongMemEval) avant de relancer les tests de compact.

Les anciens resultats sont archives dans `old_results/` pour reference.

---

## Etape 1 : Extraction padding pool

**Script : `extract_padding.py`**

- Input : `longmemeval_data/longmemeval_s_cleaned.json`
- Output : `data/padding_pool.jsonl` (1 session par ligne)
- Chaque ligne : `{id, source, turns: [{role, content}], chars, est_tokens}`
- Deduplique par session ID
- Stats : 18K sessions, ~61M tokens

## Etape 2 : Extraction evidence LongMemEval

**Script : `extract_evidence.py`**

- Input : `longmemeval_data/longmemeval_oracle.json`
- Output : `data/evidence_longmemeval.json`
- Chaque entry :
  ```json
  {
    "fact_id": "LM_0001",
    "question": "What was the first issue...",
    "answer": "GPS system not functioning correctly",
    "keywords": ["GPS", "not functioning", "correctly"],
    "evidence_turns": [{"role": "user", "content": "..."}],
    "has_answer_indices": [0, 4],
    "source": "longmemeval",
    "original_question_id": "gpt4_2655b836",
    "est_tokens": 4764
  }
  ```
- Extraction keywords : depuis la reponse attendue (noms propres, nombres, termes specifiques)
- 500 entries, ~4.4M tokens d'evidence

## Etape 3 : Generation evidence synthetique

**Script : `generate_evidence.py`**

- Generer des mini-conversations (6-12 turns) ou un fait emerge naturellement
- Prompt LLM : "Write a realistic multi-turn conversation where the user
  naturally mentions that [FACT]. The fact should emerge in context, not as
  a random one-liner."
- Chaque evidence : meme format que LongMemEval, tag `source: "synthetic"`
- Fact IDs : `SY_0001`, `SY_0002`, ...
- Generer ~200 facts (assez pour d100 + marge)
- Varier les sujets : infra, code, config, decisions, people, schedules
- Budget : ~$5-10 (200 appels courts)

## Etape 4 : Assemblage contextes

**Script : `build_contexts.py`**

- Input : padding pool + evidence (LM + SY) + config (densite, target tokens, seed)
- Algorithme :
  1. Selectionner N evidence sessions selon la densite voulue
  2. Tirer des padding sessions aleatoirement (seed fixe) pour remplir
  3. Intercaler evidence a positions controlees (uniform spacing)
  4. Calibrer a 190K tokens exact via `count_tokens` API
  5. Sauvegarder contexte + metadata (positions des faits, sources)
- Output : `data/contexts/recall_190K/d{N}_seed{S}.json`
- Deterministe : meme seed = meme contexte, reproductible

Pour le bench compact :
- Assembler UNE longue conversation (~10M tokens)
- Decouper en tranches progressives (10K, 50K, 100K, 190K, 500K, 1M)
- Les faits sont distribues dans la conversation complete
- Chaque tranche contient les faits de son prefix

## Etape 5 : Adapter le bench recall

**Modifier `benchmark_batch_meta.py` (ou nouveau script)**

- Charger les contextes pre-assembles au lieu de les generer
- Charger les facts (questions + keywords) depuis le fichier evidence
- Le reste est identique : Q&A batch + juge batch + metrics
- Nouveau dans l'analyse : recall par source (longmemeval vs synthetic)

## Etape 6 : Adapter le bench compact

- Utiliser les tranches progressives comme input
- A chaque seuil : compacter + Q&A batch + mesurer recall
- Comparer 3 strategies (brutal, incremental, frozen) sur memes donnees
- Les faits etant fixes, on peut tracer leur survie a travers les cycles

## Capacites et limites

| Cible | Padding | Max facts (LM) | Max facts (LM+SY) |
|-------|:-------:|:--------:|:---------:|
| 190K  | OK      | ~39      | ~100+     |
| 1M    | OK      | ~200     | ~400+     |
| 10M   | OK      | 500      | 500+      |
| 100M  | **NON** (61M dispo) | - | - |

Pour 100M : completer le pool avec ShareGPT/UltraChat bruts (hors LongMemEval).
Pas prioritaire — 10M suffit largement pour le bench compact.

## Priorite d'execution

1. **Extract padding** (etape 1) — 5 min, pas d'API
2. **Extract evidence LM** (etape 2) — 5 min, pas d'API
3. **Build contextes recall 190K** (etape 4, densites basses) — minutes, 5 appels count_tokens
4. **Lancer bench recall** sur contextes LM natifs (d4, d8, d19) — batch API, ~$10
5. **Comparer** avec anciens resultats (meme protocole, meilleur padding)
6. **Generate evidence SY** (etape 3) — si besoin de densites hautes, ~$5-10
7. **Build contextes compact** (etape 4 progressive) — plus tard
8. **Bench compact** (etape 6) — plus tard

## Tracabilite

Chaque fait porte :
- `fact_id` : LM_XXXX (LongMemEval) ou SY_XXXX (synthetique)
- `source` : "longmemeval" | "synthetic"
- `position` : index dans le contexte assemble (% du total)
- `est_tokens` : taille de l'evidence session

L'analyse peut ventiler les resultats par source pour detecter
un biais eventuel des faits synthetiques.

## Cout estime

| Etape | Cout |
|-------|------|
| Extraction (1-2) | $0 |
| Generation SY (3) | ~$5-10 |
| Assemblage (4) | ~$0.50 (count_tokens) |
| Bench recall 190K (5) | ~$19/run (batch API) |
| Bench compact (6) | ~$5 compact sync + $19 Q&A batch |
| **Total v4** | **~$50-70** |

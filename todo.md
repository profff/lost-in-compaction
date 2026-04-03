# TODO — Phase D : Strategy Comparison at Constant Density

## Contexte

Les runs §6 originaux utilisaient 80 faits fixes dans des conversations de 500K à 10M tokens.
La densité δ décroissait de 0.16 à 0.008 → confound compaction vs dilution.

**Nouveau design** : faits distribués uniformément, évaluation à des checkpoints mid-feed.
Modèle QA : Sonnet 4.6 via gateway `claude -p` (gratuit, abo Max).

## Infra

- [x] `llm_backend.py` : backend `wrapper` (WrapperBackend, parallèle, ThreadPoolExecutor)
- [x] `benchmark_compaction_v2.py` : `WrapperLLM` (chat_raw compatible)
- [x] `benchmark_iterative_v6.py` : refactoré pour backend abstraction + `--checkpoints`
- [x] `openai-gateway/server.py` v0.2.0 : proxy `claude -p` sur port 8082
- [x] Conversation 5M construite : `data/conversations/v6_R4/d200_5M_seed42.json` (200 faits, δ=0.04)

## Phase D — Runs

### Run test (en cours, 2026-04-03)
- **Conv** : 5M, 200 faits R4, δ=0.04
- **Stratégie** : S3 seul, checkpoint @2.5M
- **Backend** : wrapper (gateway claude -p), 2 workers
- **Objectif** : valider le pipeline checkpoint + gateway

### Run complet Phase D
- **Conv** : 5M, 200 faits R4, δ=0.04
- **Stratégies** : S1, S2, S3, S4
- **Checkpoints** : 500K, 1M, 2M, 3M, 5M
- **Modèle QA** : Sonnet via gateway
- **Modèle judge** : Sonnet via gateway (ou Haiku batch si budget)
- **Estimation** : ~4-5h overnight avec gateway

### Extension 10M (si résultats intéressants)
- Construire conv 10M (besoin de plus de faits → R2 500 faits ou synthétiques)
- Checkpoints : 500K, 1M, 2M, 5M, 7M, 10M
- Objectif : observer les merges FrozenRanked (rank 2+)

---

## Papier

- [ ] Clarifier modèles utilisés (Haiku §3-5, Sonnet §6/Phase D)
- [ ] Réduire §6 aux résultats préliminaires + introduction Phase D
- [ ] Retirer §7 (modèle prédictif) → reconstruire après Phase D
- [ ] Intégrer résultats Phase D dans §6
- [ ] Mettre à jour §8 Discussion
- [ ] Phase D ≠ Future Work, c'est §6 amélioré
- [ ] Simplifier multi-model → garder §5.7 Sonnet, retirer Qwen
- [ ] Push GitHub : `git@github.com:profff/COMPACTION_BENCH.git`

## Piste : S5 Retrieval Memory
- Au lieu de compacter, embed la conversation chunk par chunk + vector search
- Inspiré discussion avec chercheur Bell Labs (tool-augmented retrieval)
- Ref : MemGPT/Letta, ReadAgent, MemWalker
- À implémenter après Phase D comme comparaison

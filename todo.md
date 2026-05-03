# TODO — Phase D : Strategy Comparison at Constant Density

## État au 2026-04-30 ~3h du matin

**4 runs Phase D effectués** :
- **run 2136** (Sonnet judge, gateway) : S1+S2+S3+S4, 4 checkpoints, sans 5M final
- **run 2249** (Haiku judge, gateway) : S3+S4 seulement, 5 checkpoints incluant 5M
- **run 2333** (Haiku judge, gateway) : S1+S2 seulement, partiel (rate limit nocturne)
- **run 29_2249** (Haiku judge, batch API) : **DATASET COMPLET** S1-S4, 5 checkpoints
  - Coût : $140 / $250 disponibles → reste ~$110

**Tous rejudgés en mode STRICT** (BATCH_JUDGE_PROMPT_STRICT) — judge Haiku était laxiste
sur "recalled" (marquait true pour des "I don't recall + topic mentioned").

**Stats consolidées strict** : voir `figures/fig_phase_d_strict_meanstd.png`
- S4 mean : 33% → 24% → 15% → 14% → 7% (n=3 replicates)
- Hiérarchie S4 > S3 > S2 ≈ S1 confirmée
- Variance importante (std 5-15pp) mais hiérarchie maintenue

**Décisions à prendre** :
- 2e run complet pour confirmer (~$140) ?
- OU rerun QA-only pour isoler variance compaction vs QA (~$80) ?
- OU on attaque le papier avec ce qu'on a ?

**Investigations 0% suspects** (faites) :
- ✅ S1@2M (run 2333) : était JUDGE_FAIL, rejudgé via `rejudge_only.py` → **3.8%**
- ✅ S4@3.5M (run 2249) : était bug summary, vrai recall **16.4%** (recomputed)
- ✅ Audit complet via `phase_d_consolidated.json`

**Trous restants** (à reeval depuis snapshots quand quota Max dispo) :
- S2@3.5M (run 2136 ou 2333) — QA_FAIL
- S3@3.5M (run 2249) — QA_FAIL
- S1+S2 @ 3.5M et 5M (run 2333) — QA_FAIL ou missing

Estimé ~50-100 appels via `reeval_checkpoints.py --only-failed`.

## Commandes utiles

### Rerun les QA fails (depuis snapshots, plus cher mais nécessaire)
```bash
# Run 2333 (S1+S2) — checkpoints + final
python reeval_checkpoints.py iterative_v6_R4_20260428_2333 \
    --strategies S1,S2 --checkpoints 2001K,3501K,5M \
    --backend wrapper --workers 1 \
    --judge-backend wrapper --judge-model claude-haiku-4-5-20251001 \
    --only-failed

# Run 2249 (S3+S4) — juste S3@3.5M
python reeval_checkpoints.py iterative_v6_R4_20260427_2249 \
    --strategies S3 --checkpoints 3501K \
    --backend wrapper --workers 1 \
    --judge-backend wrapper --judge-model claude-haiku-4-5-20251001 \
    --only-failed

# Run 2136 (S1-S4) — S2@3.5M, S3@2M
python reeval_checkpoints.py iterative_v6_R4_20260416_2136 \
    --strategies S2,S3 --checkpoints 2001K,3501K \
    --backend wrapper --workers 1 \
    --judge-backend wrapper --judge-model claude-haiku-4-5-20251001 \
    --only-failed
```

### Rejudge depuis answers (bcp moins cher)
```bash
# Si les answers sont OK mais judge a fail
python rejudge_only.py iterative_v6_R4_*/checkpoint_XXX \
    --strategies S1 --judge-model claude-haiku-4-5-20251001
```

### Recompute summaries (gratuit, depuis disque)
```bash
python recompute_summaries.py --all
```

### Consolider tout
```bash
python phase_d_summary.py
# génère phase_d_consolidated.json
```

## Procédure recommandée pour les runs futurs

1. ~~Désactiver hooks globaux~~ : **plus nécessaire** depuis 2026-04-29.
   Le gateway utilise `--setting-sources project,local` qui skip les hooks user
   automatiquement. Tes hooks ctxguard/mood/nectime restent actifs pour tes
   sessions Claude Code normales.
2. **Lancer gateway** : `python D:/dev/AI_Bridge/CLAUDE_FRIENDS/openai-gateway/server.py`
3. **Workers=1**, `--judge-model claude-haiku-4-5-20251001` pour économiser quota
4. Se rappeler : rate limit Max 5h ~50-100M tokens d'input. 4 strats × 5 cp × 138K input = ~28M tokens d'input QA → potentiellement OK en 1 fenêtre, mais juste.

## Fichiers de travail

- **`phase_d_consolidated.json`** : tous les résultats Phase D consolidés
- **`figures/fig_phase_d_preview.png`** : courbes S1-S4 vs taille de conv
- **`IDEAS_FUTURE_WORK.md`** : pistes Future Work (Frozen-lossless, retrieval, etc.)
- **`reeval_checkpoints.py`** : reeval depuis snapshots disque
- **`rejudge_only.py`** : rejudge depuis answers existantes
- **`recompute_summaries.py`** : recompute summaries depuis verdicts existants
- **`phase_d_summary.py`** : génère `phase_d_consolidated.json`

## Papier

- [ ] Compléter les trous Phase D (~50-100 appels)
- [ ] Mettre à jour §6 avec les nouvelles données Phase D
- [ ] Réécrire §8 Discussion avec les findings :
  - Hiérarchie S4 > S3 > S2 > S1
  - Convergence S3 ≈ S4 à grande échelle (5M)
  - L'avantage S4 se concentre en début de course (rank 2+ jamais déclenché)
  - Argument 1M context renforce le besoin de la compaction (cf IDEAS_FUTURE_WORK.md §7)
- [ ] Retirer ou simplifier §7 (modèle prédictif) — données changent
- [ ] Push GitHub

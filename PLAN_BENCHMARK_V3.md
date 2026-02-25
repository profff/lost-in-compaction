# Plan Benchmark Compaction v3

> Plan complet pour la prochaine itération du benchmark.
> Objectif : méthodologie plus rigoureuse, données exploitables, courbe de dégradation.

## Contexte

Benchmark v2 (fait) a produit des résultats à 1.5M et 3M tokens avec 3 stratégies.
Limites identifiées :
- Métriques uniquement en fin de conversation (pas de courbe de dégradation)
- Pas d'archivage des données brutes (réponses, verdicts juge, contexte)
- Questions par batch de 10 (impact sur le recall non vérifié)
- API synchrone avec rate limiting (coûteux en temps)
- Bug Q&A découvert (contexte trop large pour les questions) — fixé mais
  montre l'importance d'archiver les données pour détecter ce genre d'erreurs

## Améliorations planifiées

### 1. Archivage complet des données (priorité haute)

Sauvegarder avec chaque run :
- **Faits générés** : texte, question, keywords, position (index message)
- **Contexte compacté final** : liste complète des messages
- **Réponses du LLM** : réponse brute à chaque question
- **Verdicts du juge** : réponse brute du juge (pas juste recalled/accurate)
- **Metadata** : seed, preset, timestamps, coûts API

Format : un gros JSON ou un dossier par run avec fichiers séparés.

### 2. Keyword scan proxy (priorité haute, coût = 0)

À chaque cycle de compaction, scanner les keywords de chaque fait déjà injecté
dans le texte du contexte compacté. Résultat : `keyword_present: bool` par fait.

- **Gratuit** : aucun appel API, juste du string matching
- **Upper bound** : si le keyword n'est plus dans le contexte, recall = 0 garanti
- **Disponible à chaque cycle** : courbe de survie sans aucun coût
- **Mesure "Lost in the Middle"** : gap entre keyword_present et LLM_recalled

### 3. Méta-expérience : impact du batch size (priorité haute, faire EN PREMIER)

Hypothèse : poser 10 questions d'un coup pourrait noyer le modèle et réduire
le recall (surtout mid-conversation). Test :

- Config légère : 500K tokens, 50 faits, 1 stratégie (frozen)
- Batch sizes : 1, 5, 10, 20 questions par appel
- Comparer recall × batch size
- Coût estimé : ~$5-10 (4 configs × 50 questions × contexte)

Si batch size n'a pas d'impact → garder 10 (10× moins cher que individuel).
Si impact significatif → passer à 1 question/appel avec batch API.

### 4. Anthropic Batch API (priorité moyenne)

Utiliser l'API batch (50% discount, pas de rate limiting) pour les phases
évaluation (questions) et jugement :

- Doc : https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- Console : https://platform.claude.com/workspaces/default/batches
- Submit batch → poll → retrieve results
- Parfait pour la phase Q&A qui est embarrassingly parallel
- Phase compaction reste synchrone (séquentielle par nature)

### 5. Courbe de dégradation — évaluation aux snapshots intermédiaires

Évaluer le recall à intervalles réguliers pendant le feed+compact :
- **Keyword scan** : à chaque cycle (gratuit) → courbe "survie dans le texte"
- **LLM evaluation** : tous les 5 cycles (~10 points) → courbe "recall réel"
- Seulement sur les faits déjà injectés à ce stade

Deux courbes par stratégie :
1. `keyword_survival(t)` — upper bound, gratuit, chaque cycle
2. `llm_recall(t)` — recall réel, batch API, tous les 5 cycles

Le gap entre les deux = effet "Lost in the Middle" quantifié.

Coût estimé (batch API, 10 points × 3 stratégies) : ~$25-50

### 6. Snapshots contexte

Sauvegarder l'état complet du contexte à chaque cycle de compaction :
- Nombre de messages, tokens estimés
- Nombre de frozen summaries (si applicable)
- Liste des faits dont les keywords sont encore présents
- Optionnel : dump complet des messages (lourd mais utile pour debug)

### 7. Métrique Facts/KTok

Ajouter aux résultats : `facts_recalled / conversation_KTokens`
- Normalise entre différentes longueurs de conversation
- Permet de tracer la densité d'information retenue vs conversation traitée
- Exemples actuels :
  - 1.5M Frozen : 24/1500 = 0.016 facts/KTok
  - 3M Frozen : 14/3000 = 0.0047 facts/KTok

## Améliorations du papier

### 8. Section fact generation : retirer le nombre fixe

Expliquer la méthodologie (densité 1/10K, catégories, keywords) sans
hardcoder "150 facts". Le nombre dépend de la taille de la conversation.

### 9. Restructuration compaction strategies

Un chapitre par stratégie avec :
- Schéma du contexte (ASCII art existant)
- Algorithme en diagramme mermaid (était là avant, a été retiré)
- Description

## Ordre d'exécution recommandé

```
1. Meta-test batch size (#3)          ← valide la méthodo, pas cher
2. Keyword scan proxy (#2)            ← gratuit, implement d'abord
3. Archivage complet (#1)             ← instrument le bench
4. Snapshots contexte (#6)            ← dépend de #1
5. Batch API (#4)                     ← réduit les coûts de 50%
6. Courbe de dégradation (#5)         ← le gros morceau, dépend de #2,#4,#6
7. Facts/KTok (#7)                    ← trivial, faire en passant
8. Papier (#8, #9)                    ← en parallèle du reste
```

## Estimation coûts du prochain benchmark complet

Avec batch API (50% discount) et évaluation aux snapshots :

| Phase | Appels | Input tokens | Coût batch |
|-------|--------|-------------|------------|
| Compaction (48 cycles × 3 strats) | ~144 | ~25M | ~$10 |
| Keyword scan (48 cycles × 3) | 0 | 0 | $0 |
| Q&A snapshots (10 points × 3 strats × 30 batches) | ~900 | ~144M | ~$29 |
| Jugement (même volume) | ~900 | ~30M | ~$6 |
| **Total** | ~1944 | ~199M | **~$45** |

Plus le méta-test batch size : ~$5-10.

## Stratégies à tester

- Brutal (baseline)
- Incremental (existant)
- Frozen (existant, meilleur actuel)
- Frozen Ranked (implémenté, pas encore benchmarké)

## Fichiers clés

- `benchmark_compaction_v2.py` — script de bench (à modifier)
- `francine/compaction.py` — implémentations (Incremental, Frozen, FrozenRanked)
- `BENCHMARK_ANALYSIS.md` — papier (v4 actuelle)
- `STRATEGIES_IDEAS.md` — idées de stratégies futures
- `plot_recall_distribution.py` — distribution fine (15 bins)
- `plot_context_composition.py` — composition du contexte final

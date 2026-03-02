# TODO — Phase D : Nested Benchmark à densité constante

## Contexte

Les runs §6 utilisent 80 faits fixes dans des conversations de 500K à 10M tokens.
La densité δ décroît de 0.16 à 0.008 facts/kTok — on confond compaction et dilution.

Le nested design résout ça :
- Conversations emboîtées (1M = 500K existante + 500K neuve)
- Cohortes de faits (G1, G2, ...) suivies à travers toutes les tailles
- Mode R2 (500 faits LongMemEval) au lieu de R4 (234)

Budget déjà dépensé : ~$450

---

## Plomberie (avant tout run)

- [ ] Modifier `build_conversation_v6.py` pour conversations nested + mode R2
- [ ] Modifier `benchmark_iterative_v6.py` pour eval par cohorte + checkpoints intermédiaires
- [ ] Construire les conversations nested
- [ ] Investiguer LoCoMo comme source de faits supplémentaires (pendant que les runs tournent)

---

## Runs planifiés

### Run 1 — Proof of concept (~$26)
- **Objectif** : Valider le design nested avant d'investir plus
- **Taille** : 1M tokens (= conv 500K existante + 500K neuve)
- **Densité** : δ = 0.16 facts/kTok (80 faits par segment de 500K)
- **Faits** : 160 total (G1=80 @0-500K, G2=80 @500K-1M)
- **Stratégies** : S1, S2, S3, S4
- **Évaluation** : recall G1@500K, recall G1@1M, recall G2@1M
- **Résultat attendu** : on voit si les faits G1 "vieillissent" entre 500K et 1M

### Run 2 — Série complète δ=0.16 (~$83 S3+S4, ~$157 all 4)
- **Objectif** : Matrice de vieillissement des faits à densité constante
- **Taille** : 500K → 1M → 1.5M → 2M → 2.5M → 3M (6 checkpoints)
- **Densité** : δ = 0.16 facts/kTok constant
- **Faits** : 500 total (R2 mode), ~80 par segment de 500K, 6 cohortes
- **Stratégies** : S3+S4 minimum, idéalement les 4
- **Évaluation** : à chaque checkpoint, recall de TOUTES les cohortes précédentes
- **Résultat attendu** : courbe de vieillissement par cohorte, comparable à calibration §3

### Run 3 — Chasse aux merges 6M (~$70)
- **Objectif** : Observer les merges FrozenRanked (jamais vus à 10M actuel)
- **Taille** : 6M tokens
- **Densité** : δ = 0.08 facts/kTok (40 faits par segment 500K)
- **Faits** : ~480 (R2 mode)
- **Stratégies** : S3 + S4 seulement
- **Résultat attendu** : ~108 cycles → summaries devraient dépasser budget → merges S4

### Run 4 — Chasse aux merges 12M (~$125)
- **Objectif** : Observer le merge regime de FrozenRanked à grande échelle
- **Taille** : 12M tokens
- **Densité** : δ = 0.04 facts/kTok (20 faits par segment 500K)
- **Faits** : ~480 (R2 mode)
- **Stratégies** : S3 + S4 seulement
- **Résultat attendu** : ~216 cycles → merges massifs → S4 devrait enfin se différencier de S3

---

## Décisions à prendre

- Run 2 : 4 stratégies ($157) ou S3+S4 seulement ($83) ?
- Run 3+4 : lancer si Run 1 est concluant, ou directement ?
- LoCoMo : si les 500 faits R2 suffisent pas, faut-il synthétiser ou importer ?

---

## Autres TODO (papier)

- [ ] Intégrer résultats nested dans §6 quand disponibles
- [ ] Mettre à jour modèle prédictif §7 avec nouvelles données
- [ ] GitHub push : `git@github.com:profff/COMPACTION_BENCH.git`

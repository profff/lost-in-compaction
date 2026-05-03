# Future Work — Beyond Compaction

Idées de mémoire LLM alternatives à la compaction, esquissées pendant la session
de finalisation Phase D (2026-04-28).

À développer après publication de "Lost in Compaction".

## Contexte

Notre papier établit que **la compaction perd de l'information**, et que l'enjeu
augmente avec la taille du contexte (Opus 1M = compactions plus rares mais plus
catastrophiques quand elles arrivent).

Les idées ci-dessous visent à conserver l'information **sans perte** tout en
gardant le contexte navigable par le LLM.

---

## 1. Frozen-Lossless (variante S3 enrichie)

**Principe** : chaque summary frozen garde un pointer vers le verbatim original.

```json
{
  "id": "frozen_42",
  "summary": "...",            // narrative court (S3 classique)
  "link": "frozen_42_full.txt" // verbatim accessible via tool
}
```

**Tool** : `expand(frozen_id)` charge le verbatim dans le contexte (swap avec
un chunk pas pertinent) ou à côté.

**Avantages** :
- Réutilise le mécanisme S3 mesuré dans le papier
- Le LLM décide quand expander (économe par défaut)
- Pas de perte sur les questions qui demandent de la précision

**Variante mesurable** : `S3-Lossless` à comparer à `S3` sur les mêmes faits.
Hypothèse : recall > S3 pour questions précises, équivalent pour questions
générales.

---

## 2. Continuous / On-the-fly Frozen

**Principe** : compacter en streaming au lieu d'attendre un seuil.
À chaque turn, le n-ième turn précédent passe automatiquement en frozen.

**Avantages** :
- Pas de "saut" catastrophique de compaction (notre principal problème à 1M)
- Chaque summary est mieux focalisé (peu de matière)
- Pas de phase transitoire saturée

**Trade-offs** :
- +1 appel LLM par turn (latence interactive)
- Volume tokens total ↑ (mais inputs petits)
- Chain longue : 5000 turns = 5000 entrées dans le contexte

**Lien existant** : c'est essentiellement l'**anchored iterative** de
Factory.ai (cité §1.2). Mesure manquante : impact de la fréquence de
compaction sur le recall.

---

## 3. Graph-augmented context (pseudo-Graphiti dans le contexte)

**Principe** : enrichir chaque frozen avec des liens vers d'autres frozen
pour permettre la nav par chaîne causale.

```json
{
  "id": "frozen_43",
  "tags": ["resultats_mesures", "bug_technique"],
  "actors": ["claude", "olivier"],
  "time": "2026-04-26 18:00",
  "oneliner": "crash après N requests, ne pas oublier de désactiver les tools",
  "link": "frozen_43_full.txt",
  "frozen_link": "frozen_42"  // référence à la décision/contexte précédent
}
```

**Avec Opus 1M** : 50 sessions × ~200 tokens = 10K tokens d'index = 1% du
contexte. Multi-hop natif sans tool call (le LLM suit les liens dans son
attention).

**Inspiration** : Graphiti / Zep (graphe de connaissances temporel), mais
simplifié — graphe directement *dans* le contexte au lieu d'un service séparé.

**Variante mesurable** : `S5_Graph` vs `S3_Frozen`. Hypothèse : meilleur recall
sur questions multi-hop ("le bug → la décision qui en a découlé → le résultat").

---

## 4. Tags fonctionnels vs topiques

**Insight** : pour un agent **mono-projet**, les tags topiques saturent
("tout c'est compaction benchmark"). Les tags **fonctionnels** restent
discriminants : `[décision]`, `[résultat]`, `[bug]`, `[idée]`, `[blocage]`.

**Pour multi-projet** (cas Francine / agent par contexte client) : les tags
topiques sont aussi importants.

**Implications design** : prévoir les deux dimensions de tags + une dimension
temporelle/acteur.

---

## 5. Keywords-only avec petit LLM en parallèle

**Principe** : ne pas summarizer du tout. Juste extraire 5-10 keywords +
oneliner de 10-15 tokens. Verbatim en mémoire externe.

**Architecture** :
- Agent principal (Opus/Sonnet) répond à l'utilisateur
- Petit LLM (Haiku-class ou local) extrait keywords/tags du turn précédent
  en parallèle, avec juste les keywords précédents en contexte (cohérence
  nomenclature)
- Quasi gratuit en latence côté user

**Format index** : ~20-30 tokens par session vs 50-100 pour summary classique.

**Limitation** : keywords seuls peu navigables, le LLM principal va systématique
faire `expand()`. D'où l'intérêt du **oneliner + keywords** combiné (compromis).

---

## 6. Trade-off frequency × granularity

Notre Phase D actuelle teste **un seul point** sur cette courbe (high
watermark fixe à 90%, compaction par blocs de ~57K tokens).

À explorer dans un papier de suivi :
- **Fréquence** : compacter tous les N tokens (fixe) vs à un seuil (dynamique)
- **Granularité** : compacter K tokens à la fois (1K → 100K)

Extrêmes :
- (très_fréquent, très_petit) = continuous frozen → chain longue, pas de saut
- (rare, très_gros) = compaction batch (notre étude) → peu de chain, perte
  massive

Hypothèse : sweet spot quelque part au milieu, dépendant de la taille de la
fenêtre contextuelle.

---

## 7. Implications du Opus 1M sur le papier

Avec contexte 1M, la compaction se déclenche **moins souvent** mais chaque
passe est **plus catastrophique** :

| Window | Compactions / 5M conv | Tokens compactés / passe |
|--------|-----------------------|--------------------------|
| 200K   | ~50                   | ~60K (30% du contexte)   |
| 1M     | ~5                    | ~700K (70% du contexte)  |

**Coûts de la compaction massive (700K en une passe)** :
- Attention dilution sur l'input du compactor
- Irréversibilité presque totale (passe rare et énorme)
- Qualité du résumé qui chute mécaniquement avec la taille de l'input

**Coûts structurels du contexte 1M lui-même** (indépendants de la compaction) :
- KV cache massif → GPU/RAM lourde pour servir le contexte actif
- Réduction des poids (quantization récente, papier Google sur les MoE et la
  compression KV) atténue mais n'annule pas
- C'est une raison structurelle de vouloir compacter, même sans le problème
  de qualité

**Conclusion pour la Discussion §8** : l'argument du 1M *renforce* le besoin
de comprendre la compaction. Plus la fenêtre est grande :
1. Plus le coût de la garder pleine est élevé (KV cache)
2. Plus chaque compaction "rare" est catastrophique en perte
3. Donc plus il faut concevoir des stratégies qui évitent les sauts massifs
   (continuous frozen, lossless avec expand, etc.)

---

## Plateformes pour POC

Pour tester ces designs sans coût API :
- **Gateway `claude -p` (port 8082)** : accès Opus 1M via abo Max, compatible
  OpenAI SDK. Limites : rate-limit 5h dur, single-user.
- **Sonnet 200K via gateway** : suffisant pour valider les concepts (1M utile
  pour validation à grande échelle seulement).
- **Local LLMs longs contextes** : Qwen 32B (128K), Llama 3.1 (128K) sur
  matériel adéquat.

Le projet **Francine** (`D:/dev/AI_Bridge/CLAUDE_FRIENDS/francine`) est le
candidat naturel pour implémenter ces patterns.

---

## Références

- MemGPT/Letta — Packer 2023, archival memory + recall search
- Zep / Graphiti — Rasmussen 2025, temporal knowledge graph
- ReadAgent — Lee 2024, gist + interactive lookup
- Factory.ai anchored iterative — compaction en streaming (cité §1.2)
- Mem0 — selective retrieval, fact preservation

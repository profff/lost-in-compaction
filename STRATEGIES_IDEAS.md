# Stratégies de compaction — idées à tester

## Stratégies benchmarkées (v1)

| # | Nom | Résultat 1.5M/200K | Verdict |
|---|-----|---------------------|---------|
| 1 | **Brutal** | 12.7% recall, 0% mid | Pire partout |
| 2 | **Incremental** | 16% recall, JPEG cascade | Bon en late, mauvais en early |
| 3 | **Frozen** | 16% recall, 90% F000-F009 | Bon en early, mauvais en late |

## Stratégies candidates (v2)

### S4. Frozen V2 — merge hiérarchique par rang (idée Olivier)

**Principe** : Chaque résumé gelé porte un "rang" = nombre de fois qu'il a été
mergé. Rang 1 = frais. Quand le budget est dépassé, on merge toujours les 2
résumés du plus bas rang disponible (en paire). Quand il n'y a plus de paires
de rang N, on passe au rang N+1.

```
Initial:    [R1] [R1] [R1] [R1] [R1] [R1] [R1] [R1]
Merge 1:    [R2]      [R1] [R1] [R1] [R1] [R1] [R1]
Merge 2:    [R2] [R2]      [R1] [R1] [R1] [R1] [R1]
Merge 3:    [R2] [R2] [R2]      [R1] [R1] [R1] [R1]
Merge 4:    [R2] [R2] [R2] [R2]           [R1] [R1]  (orphelin R1 si impair)
Plus de R1: [R3]      [R2] [R2] [R1]
```

**Propriété** : max log₂(N) merges par fait au lieu de N/2 séquentiel.
On merge toujours des résumés de même "qualité".

**Hypothèse** : Meilleur recall early que frozen actuel sur longues conv (quand
les merges se déclenchent).

**Complexité implem** : Faible — ajouter un champ `rank` au marqueur frozen,
modifier `_merge_summaries()` pour chercher la paire de plus bas rang.

**Variante** : pondérer le budget par rang (un R2 "coûte" plus cher qu'un R1
dans le budget, pour forcer les merges de R1 en priorité).

---

### S5. Résumé structuré unique (mini-CLAUDE.md)

**Principe** : Au lieu de N résumés narratifs empilés, maintenir UN SEUL
document structuré avec des sections thématiques :

```markdown
## Decisions
- [cycle 3] Choisi Redis pour le cache (pas Memcached)
- [cycle 7] Architecture microservices confirmée

## Config & Infra
- Server kronos-000: 16.140.63.58, SSH 2222
- DB: PostgreSQL 15.4, port 5432

## Files modified
- src/auth.py: refactored login flow (cycle 5)
- src/cache.py: created (cycle 3)

## Current task
- Debugging timeout on /api/users endpoint
```

**Avantage** : Résout le problème "Lost in the Middle" — un seul document
structuré est plus facile à scanner que 23 résumés narratifs. Le modèle peut
aller directement à la section pertinente.

**Risque** : Le LLM doit METTRE À JOUR le document existant (pas le réécrire
from scratch). C'est un prompt plus complexe : "voici le document actuel +
les nouveaux messages, produis le document mis à jour". Si le LLM réécrit
tout, on perd les anciennes entrées → retour au JPEG cascade.

**Complexité implem** : Moyenne — nouveau prompt de compaction (update vs
rewrite), parsing/validation du format structuré.

**Variante "append-only"** : Les sections ne sont jamais réécrites, juste
augmentées. Quand une section dépasse un seuil, on la résume (mais section
par section, pas tout le document).

---

### S6. Hybrid frozen + structuré

**Principe** : Deux zones de mémoire compactée :
1. **Facts store** (structuré, clé-valeur) — jamais résumé, append-only,
   extracté automatiquement pendant la compaction
2. **Narrative summary** (classique) — résumé incrémental du contexte narratif
   (quoi on faisait, pourquoi, où on en est)

```
[FACTS — structured, append-only]
server.kronos-000.ip = 16.140.63.58
server.kronos-000.ssh_port = 2222
decision.cache = Redis (cycle 3)
...

[NARRATIVE SUMMARY — incremental, peut être re-résumé]
We started by setting up the infrastructure, then moved to...

[Raw recent messages]
```

**Avantage** : Les faits précis survivent indéfiniment (append-only). Le
narratif peut être compressé agressivement sans perdre les détails factuels.

**Risque** : L'extraction de faits dépend du LLM — il peut rater des faits
importants ou extraire du bruit. Nécessite un bon prompt d'extraction.

**Complexité implem** : Haute — deux passes de compaction (extraction faits +
résumé narratif), gestion du fact store, déduplication.

**Lien RAG** : C'est essentiellement un RAG intégré au contexte. Pourrait
être combiné avec un vrai RAG (ChromaDB) pour offloader les faits hors
contexte.

---

### S7. Sliding window avec "fossiles"

**Principe** : Comme brutal, mais au lieu de tout résumer, on extrait les N
faits les plus "importants" avant de jeter le reste. Ces faits deviennent des
"fossiles" — des one-liners indestructibles en tête de contexte.

```
[FOSSILS — one-liners, never summarized]
- kronos-000: 16.140.63.58:2222
- Cache: Redis (decision cycle 3)
- Bug: timeout /api/users (investigating)

[Recent raw messages — sliding window]
```

**Avantage** : Très compact (fossiles = quelques tokens chacun), pas de
problème Lost in the Middle (liste courte et scannable).

**Risque** : L'importance est subjective — quels faits garder ? Le LLM doit
scorer l'importance, ce qui ajoute un appel ou de la complexité au prompt.

**Complexité implem** : Moyenne — extraction de faits + scoring importance +
gestion de la liste de fossiles.

---

### S8. Frozen avec TTL (time-to-live)

**Principe** : Chaque résumé gelé a un TTL basé sur son âge. Les résumés les
plus anciens sont progressivement "évaporés" (supprimés) plutôt que mergés.
On accepte la perte totale des faits très anciens en échange de plus d'espace
pour le présent.

**Hypothèse** : Dans un contexte de coding assistant, les faits d'il y a 2h
sont rarement utiles. Mieux vaut les sacrifier pour avoir plus de contexte
récent.

**Variante** : TTL variable par catégorie — les décisions ont un long TTL,
le debugging a un court TTL.

---

## Matrice de comparaison

| Stratégie | Early recall | Late recall | Complexité | Lost in Middle? |
|-----------|:-----------:|:----------:|:----------:|:---------------:|
| S4. Frozen V2 (rang) | ★★★★★ | ★★ | Faible | Oui (même pb) |
| S5. Structuré unique | ★★★★ | ★★★★ | Moyenne | **Non** |
| S6. Hybrid fact+narr | ★★★★★ | ★★★★ | Haute | Partiel |
| S7. Fossiles | ★★★ | ★★★★★ | Moyenne | **Non** |
| S8. Frozen TTL | ★★ | ★★★★★ | Faible | Oui |

## Priorité suggérée

1. **S4 (Frozen V2 rang)** — rapide à implémenter, teste l'hypothèse merge
2. **S5 (Structuré)** — attaque le problème Lost in the Middle
3. **S7 (Fossiles)** — variante intéressante si S5 est trop complexe

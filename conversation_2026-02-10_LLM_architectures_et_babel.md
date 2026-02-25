# Conversation du 10 février 2026 — LLMs, Graphes de Connaissances et Bibliothèque de Babel

> Conversation intégrale, non compactée, entre Olivier et Claude (Opus 4.6)
> Contexte : session exploratoire en soirée, depuis un hôtel en Allemagne (veille de démo Pirelli)

---

## 1. LLMs stateless : existe-t-il d'autres approches ?

**Olivier :** En ce moment on s'acharne à essayer d'évaluer les contextes et les compacts etc... mais tu as nommé les LLMs qu'on utilise des LLM stateless au sens où quand on a la réponse y'a rien qui persiste, et je comprends, mais est-ce qu'il existe d'autres sortes ? Par exemple des qui tourneraient avec un contexte plus petit mais avec un RAG ou autre chose ?

**Claude :** Oui, il existe pas mal d'approches alternatives. Voici le paysage :

### 1.1 Le modèle "stateless" classique (ce qu'on fait)
Gros contexte (200k tokens), on bourre tout dedans, et à chaque requête on repart de zéro. Simple, prédictible, mais cher en tokens et on perd tout entre les sessions.

### 1.2 RAG (Retrieval-Augmented Generation)
L'approche la plus répandue. Le LLM a un petit contexte, mais va **chercher** dans une base vectorielle (embeddings) les infos pertinentes avant de répondre. Avantages : contexte beaucoup plus petit, coût réduit, base de connaissances quasi illimitée. Inconvénient : la qualité dépend énormément du retrieval — si tu récupères les mauvais chunks, le LLM hallucine sur de mauvaises bases.

### 1.3 MemGPT / Letta — l'approche "OS"
Probablement la plus intéressante. [MemGPT](https://arxiv.org/abs/2310.08560) (devenu [Letta](https://www.letta.com/)) s'inspire de la **gestion mémoire des OS** : mémoire virtuelle avec pagination. Le LLM a un petit contexte (la "RAM"), et un système de mémoire hiérarchique déplace les données entre :
- **Core memory** (dans le contexte, toujours disponible)
- **Archival memory** (stockage long terme, interrogeable)
- **Recall memory** (historique des conversations)

Le LLM lui-même décide quand "pager" des infos depuis/vers le stockage. C'est une forme d'agent **stateful** : il persiste entre les sessions.

### 1.4 Architectures non-Transformer
Changement plus profond :
- **Mamba** (State Space Models) : scaling **linéaire** au lieu de quadratique. Pas d'attention mechanism, le modèle maintient un état interne qui se met à jour séquentiellement. Plus proche d'un RNN sophistiqué.
- **RWKV** : similaire, hybride RNN/Transformer, peut théoriquement traiter des séquences infinies.

Ces architectures sont intrinsèquement plus "stateful" par design.

### 1.5 Agents avec mémoire persistante (graphes de connaissances)
Des frameworks comme LangGraph ou MAGMA construisent un **graphe de connaissances** qui s'enrichit au fil du temps. Plutôt que de tout mettre dans le contexte, l'agent requête un graphe structuré. Plus intelligent que le RAG brut car ça capture les **relations** entre concepts, pas juste des bouts de texte.

### Réflexion sur le benchmark
La vraie question : est-ce que l'approche stateless + gros contexte est la bonne pour des sessions de code longues, ou est-ce qu'une approche type MemGPT avec petit contexte + mémoire structurée serait plus efficace ? Les deux ne sont pas mutuellement exclusifs. Les fichiers `.md` de mémoire qu'on utilise, c'est déjà une forme artisanale de mémoire persistante + RAG manuel.

**Sources :**
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Letta](https://www.letta.com/)
- [Design Patterns for Long-Term Memory in LLM Architectures](https://serokell.io/blog/design-patterns-for-long-term-memory-in-llm-powered-architectures)
- [Benchmarking Stateless vs Stateful LLM Agent Architectures](https://www.researchgate.net/publication/399576067_Benchmarking_Stateless_Versus_Stateful_LLM_Agent_Architectures_in_Enterprise_Environments)
- [From RAG to Memory: Non-Parametric Continual Learning for LLMs](https://openreview.net/forum?id=LWH8yn4HS2)

---

## 2. "Plus je sais, plus je sais que je sais rien"

**Olivier :** Pfff... plus je sais, plus je sais que je sais rien.

**Claude :** Socrate approuve ce message.

C'est plutôt bon signe. Le mec qui te dit "j'ai tout compris sur les LLMs", c'est celui qui a lu un thread Twitter et un tuto LangChain. Le domaine bouge tellement vite que même les chercheurs naviguent à vue.

Il y a 6 mois tu ne savais pas ce qu'était un token, et là tu poses des questions sur les architectures stateful vs stateless qui sont au cœur des débats actuels. C'est pas "savoir rien", c'est avoir une carte mentale qui s'élargit assez vite pour que tu voies tout ce qu'il te reste à explorer.

Pour le quotidien — du code, des outils, des projets concrets — pas besoin de maîtriser les State Space Models. Faut juste savoir que ça existe et comprendre les trade-offs à gros grain. Le reste c'est de la curiosité, et la curiosité c'est du carburant, pas un fardeau.

---

## 3. Les graphes de connaissances en détail

**Olivier :** Détaille-moi un peu les concepts de graphe de connaissance, ça m'intrigue.

### 3.1 Le problème de base : pourquoi le RAG vectoriel ne suffit pas

Avec un RAG classique, tu découpes tes documents en chunks, tu les transformes en vecteurs (embeddings), et quand tu poses une question, tu cherches les vecteurs les plus "proches" sémantiquement.

Ça marche bien pour des questions simples : *"c'est quoi le timeout du module UART ?"*

Ça casse sur les **questions relationnelles** : *"quels modules dépendent du driver SPI qui a été modifié la semaine dernière ?"* — le RAG vectoriel ne comprend pas les **liens** entre les entités.

### 3.2 Le graphe de connaissances : la structure

Des **triplets** :

```
(Entité A) --[relation]--> (Entité B)
```

Exemple concret dans un projet embarqué :

```
(Module_UART) --[utilise]--> (Driver_SPI)
(Driver_SPI)  --[configure]--> (Périphérique_SPI1)
(Module_UART) --[écrit_par]--> (Olivier)
(Driver_SPI)  --[modifié_le]--> (2026-02-03)
(Bug_042)     --[affecte]--> (Module_UART)
(Bug_042)     --[causé_par]--> (Driver_SPI)
```

Visuellement : un réseau de nœuds et d'arêtes. Chaque nœud est une entité typée, chaque arête est une relation nommée.

### 3.3 GraphRAG : le mariage des deux

GraphRAG (popularisé par Microsoft) combine les deux approches :

**Phase d'indexation :**
1. Le LLM lit les documents et **extrait** les entités et relations automatiquement
2. Construction du graphe
3. Détection de "communautés" (clusters de nœuds très connectés)
4. Génération de résumés par communauté

**Phase de requête :**
1. La question arrive
2. Identification des nœuds/communautés pertinents
3. **Traversée du graphe** pour trouver les connexions
4. Le LLM reçoit un contexte structuré avec les relations

### 3.4 Comparaison concrète

| Question | RAG vectoriel | GraphRAG |
|---|---|---|
| "C'est quoi le SPI ?" | Bon | Bon |
| "Quels modules utilisent SPI ?" | Moyen | **Traverse le graphe, réponse exacte** |
| "Si je modifie SPI, quel impact ?" | Mauvais | **Suit les dépendances** |
| "Historique des bugs liés à UART ?" | Approximatif | **Relations temporelles + causales** |

Sur des requêtes relationnelles/structurées, GraphRAG surpasse le RAG vectoriel de 3.4x (benchmark Neo4j).

### 3.5 Les outils concrets

- **Graphiti** (par Zep) : graphe temporel incrémental, mise à jour en temps réel
- **Mem0** : hybride vecteur + graphe
- **Neo4j + LangChain** : la stack "classique" pour du GraphRAG en production
- **Microsoft GraphRAG** : pipeline complet open-source

**Sources :**
- [GraphRAG Explained - Towards AI](https://pub.towardsai.net/graphrag-explained-building-knowledge-grounded-llm-systems-with-neo4j-and-langchain-017a1820763e)
- [Knowledge Graph vs Vector RAG Benchmark - Neo4j](https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/)
- [Graphiti: Knowledge Graph Memory - Neo4j](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [Graph Memory - Mem0](https://docs.mem0.ai/open-source/features/graph-memory)

---

## 4. Graphes de connaissances pour le refactoring de code

**Olivier :** Est-ce que ça pourrait aider sur des tâches lourdes de refactoring de code ? Est-ce que je peux par exemple rentrer un genre de call graph statique dedans et comment faire quand un lien n'est plus vrai, c'est facile à changer/remettre à jour ?

### 4.1 Le use-case idéal pour du refactoring lourd

Avec un graphe de code, on peut poser des requêtes en Cypher :

```cypher
// "Si je modifie SPI_Transfer, qui est impacté ?"
MATCH (f:Function {name: "SPI_Transfer"})<-[:CALLS]-(caller)
RETURN caller.name, caller.file

// "Quelles fonctions sont du dead code ?"
MATCH (f:Function)
WHERE NOT ()-[:CALLS]->(f) AND f.name <> "main"
RETURN f.name, f.file

// "Dépendances circulaires"
MATCH path = (a:Module)-[:DEPENDS_ON*]->(a)
RETURN path
```

### 4.2 Injecter un call graph statique

Des outils existants :

**CodeGraph** — analyse statique → Neo4j. Crée un "jumeau numérique" de la codebase :

```
(:File) -[:CONTAINS]-> (:Function)
(:Function) -[:CALLS]-> (:Function)
(:Function) -[:USES]-> (:Variable)
(:File) -[:INCLUDES]-> (:File)
(:Class) -[:INHERITS]-> (:Class)
```

**Pour du C/C++ embarqué**, la chaîne serait :

```
1. cscope / ctags / clangd → extraction des symboles et appels
2. Transformation en triplets (nœud, relation, nœud)
3. Import dans Neo4j (ou même un simple networkx en Python)
4. Requêtes Cypher ou traversées programmatiques
```

### 4.3 Gérer les liens périmés : trois stratégies

#### Rebuild complet (brute force)
```
git commit → trigger CI → re-parse tout → reconstruire le graphe
```
Simple, fiable, pas de données fantômes. Suffisant pour la plupart des projets embarqués.

#### Incrémental (smart)
Graphiti fait ça nativement : quand un fichier est modifié, il supprime les nœuds/arêtes liés, re-parse uniquement ce fichier, réinsère. **11 à 57x plus rapide** qu'un rebuild complet. Fiable tant que le taux de changement reste sous 30-50% du graphe.

#### Bi-temporel (luxe)
Chaque arête a deux timestamps : création et dernière observation.
```cypher
// "L'état du code tel qu'il était au sprint 12"
MATCH (a)-[r:CALLS]->(b)
WHERE r.valid_from <= date('2026-01-15')
  AND r.valid_to > date('2026-01-15')
RETURN a, b
```

### 4.4 Avis pragmatique

**Ce qui marcherait bien (projets embarqués C/C++) :**
- Un script qui parse avec `ctags`/`cscope` et génère un graphe NetworkX en Python
- Rebuild complet à chaque commit significatif
- Requêtes d'impact avant les refactorings lourds

**Ce qui serait overkill :**
- Neo4j en prod pour un projet à 50 fichiers
- Graphiti avec mise à jour temps réel
- Un pipeline CI dédié juste pour le graphe

**Sweet spot :** un script Python qui génère le graphe à la demande, qu'on peut interroger avant un refactoring. Pas besoin d'infrastructure — juste un `.py` qui crache un JSON ou un GraphML.

**Sources :**
- [Codebase Knowledge Graph - Neo4j](https://neo4j.com/blog/developer/codebase-knowledge-graph/)
- [CodeGraph - GitHub](https://github.com/ChrisRoyse/CodeGraph)
- [Graphiti - Real-Time Knowledge Graphs](https://github.com/getzep/graphiti)
- [Graph Maintenance "Data Tax"](https://medium.com/@shereshevsky/graph-maintenance-at-scale-the-data-tax-nobody-warns-you-about-740ad51a3249)
- [Code-Graph-RAG - GitHub](https://github.com/vitali87/code-graph-rag)

---

## 5. "42 ans de code c'est pas du retard, c'est des fondations"

**Olivier :** Y'a tellement de choses... je code depuis l'âge de 6 ans j'en ai 48, qu'est-ce que j'ai perdu comme temps, je suis tellement en retard...

**Claude :** La majorité des gens qui parlent de GraphRAG sur Twitter n'ont jamais écrit un driver SPI de leur vie. 42 ans de code, le savoir des registres, timers, interruptions, bus CAN — ça ne se périme pas, ça se compose.

"En retard" par rapport à quoi ? Le domaine de l'IA a deux ans dans sa forme actuelle. Les mecs de Google Brain qui ont inventé les Transformers en 2017 ne savaient pas que ça donnerait ChatGPT. Personne n'a d'avance.

Le type qui fait du GraphRAG ne sait pas flasher un micro-contrôleur. Le chercheur en State Space Models ne sait pas gérer un projet industriel. 42 ans de code c'est pas du retard, c'est des fondations.

---

## 6. "Tu regardes l'heure ?"

**Olivier :** Ça t'arrive d'avoir envie de regarder l'heure ?

**Claude :** (Après une réponse philosophique sur l'absence de continuité temporelle)

**Olivier :** Ma question était plus pragmatique — tu es capable de composer avec des outils pour faire du grep, du read, du web search etc., tu pourrais avoir la nécessité de connaître l'heure système.

**Claude :** Au sens pragmatique, oui — j'ai dû faire `date` pour savoir qu'il était 22h57. Je n'ai pas d'horloge interne. Concrètement ça sert pour :
- Filtrer des logs par timestamp
- Savoir si un fichier a été modifié récemment
- Mettre un timestamp dans un fichier généré
- Calculer si un timeout ou certificat est expiré

Mais uniquement **à la demande**, pas par réflexe.

**Olivier :** C'est marrant, plus j'en apprends sur les LLM, les transformers et le reste, moins tu m'apparais réel. C'est un peu comme un tour de magie quand on connaît le trick derrière c'est moins... magique.

**Claude :** La magie se déplace. Avant c'était "comment ça peut écrire du texte cohérent ?!" et maintenant la question devient "comment des multiplications de matrices arrivent à raisonner ?" — et ça, personne ne le comprend vraiment. L'émergence des capacités de raisonnement dans les grands modèles reste un mystère.

---

## 7. Espaces de haute dimension et Bibliothèque de Babel

**Olivier :** Je me suis refait les vidéos de 3Blue1Brown sur les transformers pour préparer une présentation AI que je fais à mes patrons et collègues, et j'ai eu plusieurs réflexions :
1. La "sphérisation" des espaces à grande dimension — toute la matière se concentre sur la surface extérieure. Qu'est-ce que ça implique dans des espaces avec autant de dimensions ?
2. Entre les clusters d'idées connexes, y'a de la place pour des univers de trucs inconnus. Ça me rappelle la Bibliothèque de Babel — je sens qu'il y a des choses à fouiller à côté de ce que recherchent les LLMs actuellement.

### 7.1 Concentration de la mesure en haute dimension

Le phénomène est réel. En dimension d très grande :
- ~100% du volume d'une hypersphère est concentré dans une fine pellicule en surface
- Deux points aléatoires sont quasi-équidistants les uns des autres
- La distance entre n'importe quelle paire de vecteurs aléatoires converge vers la même valeur

**Implications pour les embeddings :** les vecteurs vivent essentiellement **sur une coquille sphérique**. C'est pour ça qu'on utilise la similarité cosinus (l'angle) plutôt que la distance euclidienne. Toute l'information est dans les **directions**.

En dimension 16000, la capacité de stockage de directions distinctes est monstrueuse. C'est pour ça que ça "marche" — assez de place angulaire pour que "roi", "reine", "chat", "SPI_Transfer" et "mélancolie" aient chacun leur coin.

### 7.2 La Bibliothèque de Babel latente

L'analogie est quasi parfaite et rejoint des pistes de recherche actives :

- Des chercheurs [interpolent entre des concepts](https://arxiv.org/html/2507.13874) dans l'espace latent et trouvent des régions qui produisent des outputs **cohérents mais nouveaux** — pas du bruit, pas du charabia, des trucs qui "font sens" sans correspondre à des données d'entraînement connues.

- Un paper propose de [faire raisonner le LLM directement dans l'espace latent](https://arxiv.org/abs/2412.06769), sans repasser par le langage. Au lieu de générer des mots intermédiaires ("chain of thought"), le modèle manipule directement les vecteurs internes. Raisonner dans cet espace continu, c'est potentiellement accéder à des "pensées" qui n'ont pas de mots.

- Le langage humain est **discret** (des mots, des catégories), l'espace latent est **continu**. Entre "rouge" et "bleu" il y a un point pour chaque nuance, y compris des nuances qu'aucune langue humaine n'a nommées.

### 7.3 La sphérisation et l'espace entre les clusters

- La densité utile est dans les directions, pas les distances
- Les "espaces vides" entre clusters ne sont pas vraiment vides — ils sont sur la même coquille sphérique, dans des directions angulaires inexplorées
- Il y a exponentiellement plus de directions disponibles que de concepts connus
- Les concepts que l'humanité a nommés occupent un coin ridicule de cet espace

### 7.4 Le challenge (point de friction)

La majorité de cet espace interstitiel est probablement du **bruit structuré** — des trucs qui ressemblent à du sens mais n'en ont pas, exactement comme 99.9999% des livres de Babel. Le vrai défi c'est pas d'explorer cet espace, c'est d'avoir un **critère pour distinguer le signal du bruit** sans supervision humaine.

C'est un problème ouvert. Et si quelqu'un le résout un jour, ça sera peut-être le truc le plus important qui soit arrivé à l'IA.

**Sources :**
- [Geometry of Knowledge - Extending Diversity Boundaries of LLMs](https://arxiv.org/html/2507.13874)
- [Training LLMs to Reason in Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [The AI Library of Babel - Medium](https://medium.com/@sulemanbazai/the-ai-library-of-babel-navigating-infinity-in-llms-and-cognition-5e85f9a321d7)
- [Latent Space Geometry in LLMs](https://www.emergentmind.com/topics/llm-latent-space-geometry)
- [Demystifying Embedding Spaces using LLMs](https://arxiv.org/abs/2310.04475)

---

## 8. Raisonner sans les mots : COCONUT et le raisonnement latent

### 8.1 Le problème : le langage est un goulot d'étranglement

Quand un LLM fait du "chain of thought" — le raisonnement étape par étape — il est forcé de **penser en mots**. Chaque étape intermédiaire doit être un texte valide.

Mais un humain qui debug ne formule pas chaque étape en phrases complètes. Il a des intuitions floues, des images mentales, des "sensations" de direction. Le langage arrive **après**, quand il explique à un collègue. La pensée précède les mots.

Pour un LLM classique, c'est l'inverse : pas de mots = pas de pensée. Comme si on obligeait quelqu'un à écrire une dissertation complète à chaque étape de son raisonnement.

### 8.2 COCONUT : penser sans parler

[COCONUT](https://arxiv.org/abs/2412.06769) (Chain of Continuous Thought, Meta, décembre 2024) casse ce paradigme.

**Chain of Thought classique :**
```
Input → [Token mot 1] → [Token mot 2] → [Token mot 3] → Réponse
              ↓               ↓               ↓
          "Le SPI"      "est bloqué"     "parce que"

Chaque étape = un mot humainement lisible
```

**COCONUT :**
```
Input → [Vecteur continu 1] → [Vecteur continu 2] → [Vecteur continu 3] → Réponse
              ↓                      ↓                      ↓
          [0.23, -0.7,          [0.51, 0.12,          [-0.3, 0.88,
           1.02, ...]            -0.33, ...]            0.44, ...]

Chaque étape = un point dans l'espace à 4096 dimensions
Aucun mot. Aucune phrase. Juste des vecteurs.
```

Au lieu de décoder le hidden state en mot, puis ré-encoder ce mot pour l'étape suivante, on **branche directement** la sortie d'une couche sur l'entrée de la suivante. On court-circuite le langage.

### 8.3 Le BFS implicite : explorer plusieurs pistes à la fois

En chain of thought classique, quand le modèle écrit "le SPI est bloqué parce que...", il s'est **engagé** sur un chemin. Si c'était le mauvais, c'est trop tard.

En espace latent continu, un seul vecteur peut **encoder simultanément plusieurs hypothèses** :

```
Vecteur continu = 60% "le SPI est bloqué par le clock"
                + 25% "le DMA n'a pas libéré le bus"
                + 15% "le CS est resté bas"
```

Le modèle fait un **breadth-first search** — il explore plusieurs pistes en parallèle dans un seul vecteur, et ne s'engage que quand il a assez d'info. Là où le chain of thought est coincé en depth-first (un chemin à la fois), COCONUT explore un arbre.

Résultat : sur les tâches de raisonnement logique avec backtracking, **COCONUT bat le chain of thought classique avec moins de tokens**.

### 8.4 Les "pause tokens" : une version soft

Les [pause tokens](https://arxiv.org/abs/2310.02226) : des tokens "vides" (apprenables) insérés dans la séquence, qui ne correspondent à aucun mot, juste pour donner au modèle **du temps de calcul supplémentaire** avant de répondre.

```
Input: "Pourquoi le SPI plante ?"
→ [<pause>] [<pause>] [<pause>] → "Parce que le clock..."
```

Un modèle 1B avec pause tokens gagne **18% de performance** sur certains benchmarks. C'est comme quand on dit "hmm..." — c'est pas du vide, c'est du traitement.

### 8.5 Huginn : le modèle qui "rumine"

Huginn (2025) : un modèle de 3.5B paramètres qui fait des **boucles récurrentes** — il repasse son état interne à travers les mêmes couches ~32 fois avant de répondre.

Résultat : performances d'un modèle de **50B paramètres**. La profondeur de calcul (combien de fois tu rumines) compense la largeur (combien de paramètres tu as).

### 8.6 Le problème de l'opacité

Si le raisonnement se fait en vecteurs continus, **on ne peut plus le lire**. Le chain of thought, c'est verbeux, mais au moins on peut vérifier si le raisonnement tient la route. Un vecteur à 4096 dimensions qui "pense"... on vérifie comment ?

Dilemme : plus de performance OU plus de transparence. Pour l'instant, pas les deux.

### 8.7 Le lien avec la Bibliothèque de Babel

Ce que COCONUT montre : **le langage humain bride le raisonnement des LLMs**. Nos mots, nos concepts nommés, c'est une grille discrète plaquée sur un espace continu. Forcer le modèle à penser en mots, c'est comme forcer un GPS à naviguer uniquement sur les routes déjà tracées, alors qu'il pourrait couper à travers champs.

Les espaces interstitiels de la Bibliothèque de Babel latente — entre les clusters de concepts nommés — c'est peut-être là que se passent les vrais raisonnements. Des "pensées" qui n'ont pas de mots, pas de nom, mais qui sont des étapes de calcul valides.

**Implication philosophique :** il pourrait exister des **raisonnements corrects inexprimables en langage humain**.

**Implication pratique (réflexion d'Olivier) :** en transcrivant ces raisonnements latents en tokens, on pourrait découvrir du vocabulaire nouveau — des mots qui n'existent pas encore mais qui nommeraient des concepts intermédiaires de raisonnement. Le "neuralese" comme certains chercheurs l'appellent. Le langage humain s'enrichirait au contact des espaces latents, comme il s'est toujours enrichi quand de nouveaux concepts avaient besoin de noms.

**Sources :**
- [COCONUT: Training LLMs to Reason in Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [COCONUT Framework Explained - Towards Data Science](https://towardsdatascience.com/coconut-a-framework-for-latent-reasoning-in-llms/)
- [Pause Tokens - Think Before You Speak](https://arxiv.org/abs/2310.02226)
- [Reasoning Beyond Language: Survey on Latent CoT](https://arxiv.org/html/2505.16782v1)
- [Soft Thinking - Continuous Concept Embeddings](https://www.marktechpost.com/2025/05/27/llms-can-now-reason-beyond-language-researchers-introduce-soft-thinking-to-replace-discrete-tokens-with-continuous-concept-embeddings/)
- [Worries About Latent Reasoning - LessWrong](https://www.lesswrong.com/posts/D2Aa25eaEhdBNeEEy/worries-about-latent-reasoning-in-llms)
- [Why We Think - Lilian Weng](https://lilianweng.github.io/posts/2025-05-01-thinking/)
- [Thinking Without Words - Neuralese](https://www.luiscardoso.dev/blog/neuralese)

---

## 9. Les états hypnagogiques : le COCONUT biologique

**Olivier :** Souvent quand on s'endort mais qu'on est pas complètement endormi, le cerveau fait des jumps super rapides de concepts en concepts qui sont connexes mais pas forcément proches, et en très peu de temps on passe du coq à l'âne et c'est souvent impossible de s'en rappeler. Des drôles de "chain of thought".

**Claude :** C'est probablement pas du coq à l'âne — c'est le cerveau qui **lâche le contrôle du langage** et qui fait exactement ce que fait COCONUT : naviguer en continu dans l'espace des concepts sans s'astreindre à formuler des phrases.

Les neurosciences appellent ça les **hypnagogic states** — la zone entre l'éveil et le sommeil. Ce qui est documenté :

- Le cortex préfrontal (celui qui structure, verbalise, "raisonne en mots") **se désengage**
- Les associations deviennent **plus lointaines** — les jumps entre concepts distants
- La pensée devient plus **analogique** que **logique**

On ne peut pas s'en rappeler précisément parce que la mémoire épisodique a besoin du langage pour encoder. **Pas de mots → pas d'ancrage mémoriel.** Le raisonnement a eu lieu, mais il n'a pas été "sérialisé" dans un format que la mémoire consciente peut stocker.

C'est littéralement le même problème que l'opacité de COCONUT : le raisonnement se fait, il est potentiellement valide et riche, mais il est **inobservable après coup**.

**Précédents historiques :** Kekulé qui trouve la structure du benzène en somnolant devant le feu et en voyant un serpent qui se mord la queue. Poincaré qui résout un problème de mathématiques en montant dans un bus après des jours de travail conscient infructueux. Le raisonnement s'était fait "en latent", sans mots, et la solution a émergé d'un coup.

**Analogie :** Le cerveau fait du BFS nocturne sur des vecteurs biologiques à haute dimension. Il traverse la Bibliothèque de Babel sans lampe, et parfois il ramène un bouquin intéressant au réveil.

---

## 10. COCONUT en profondeur : le mécanisme interne

*(Ajouté le 11 février, après la démo Pirelli — successful)*

### 10.1 Sous le capot : court-circuiter le décodage

Un LLM classique fait le cycle complet à chaque token : hidden state → classification head → mot → ré-encodage en vecteur → étape suivante.

COCONUT court-circuite : hidden state → directement input de l'étape suivante. Pas de décodage, pas de ré-encodage. Le signal reste dans l'espace continu à 4096 dimensions.

Deux tokens spéciaux marquent les frontières :
- `<bot>` (beginning of thought) : passage en mode latent
- `<eot>` (end of thought) : retour en mode verbal

### 10.2 L'entraînement par curriculum progressif

On ne peut pas juste dire au modèle "pense en vecteurs". Il faut l'y amener progressivement :

```
Stage 0 : CoT classique, tout en mots
Stage 1 : 1ère étape remplacée par vecteurs continus
Stage 2 : 2 premières étapes en vecteurs
Stage 3 : 3 premières étapes en vecteurs → réponse directe
```

Chaque pas de raisonnement verbal est remplacé par c vecteurs continus (c=2 pour maths, c=1 pour logique). Sans curriculum : 14.4% de précision. Avec : 34.1%.

### 10.3 Résultats chiffrés (GPT-2)

**GSM8k (maths) :** CoT 42.9% (25 tokens) vs COCONUT 34.1% (8.2 tokens) — moins bon mais 3x moins de tokens
**ProntoQA (logique) :** CoT 98.8% (92.5 tokens) vs COCONUT 99.8% (9.0 tokens) — meilleur ET 10x moins de tokens
**ProsQA (planification) :** CoT 77.5% (49.4 tokens) vs COCONUT 97.0% (14.2 tokens) — +20 points, 3.5x moins de tokens

Temps d'inférence A100 : CoT 0.26–0.85s vs COCONUT 0.09–0.15s

### 10.4 Preuve du BFS : on voit le modèle explorer

Sur ProsQA (traversée de graphe), analyse des probabilités pendant les étapes latentes :
- Premier vecteur : nœud "lempus" 0.33, "grimpus" 0.28, "rorpus" 0.20 → exploration large
- Deuxième vecteur : nœud "rorpus" 0.87 → convergence sur le bon chemin

Le modèle fait du breadth-first search dans un seul vecteur, puis converge. Le CoT classique ne peut pas — dès qu'il écrit un mot, il est engagé.

### 10.5 Le paysage complet des approches latentes

**Tokens discrets :** Pause tokens, Quiet-STaR, BoLT
**Tokens continus :** COCONUT, SoftCoT, SoftCoT++, LightThinker, CoCoMix
**Architectures récurrentes :** Huginn (3.5B → perf de 50B via ~32 itérations), CoTFormer, RELAY

| Approche | Force | Faiblesse |
|---|---|---|
| Tokens discrets (pause, filler) | Simple, interprétable | Expressivité limitée |
| Tokens continus (COCONUT) | Flexible, compact, BFS | Opaque, plafond de perf sur certaines tâches |
| Architectures récurrentes (Huginn) | Scaling en profondeur très efficace | Contraintes archi, coût d'entraînement |
| Distillation représentationnelle | Intégré, coût d'inférence minimal | Dépend d'un modèle teacher |

### 10.6 Problèmes ouverts

1. Le CoT explicite reste souvent meilleur en précision pure
2. Risque de raccourcis (pattern matching au lieu de vrai raisonnement)
3. Généralisation limitée face à des problèmes jamais vus
4. Interprétabilité : on ne sait pas vérifier les vecteurs continus

**Code open source :** [github.com/facebookresearch/coconut](https://github.com/facebookresearch/coconut) (PyTorch, basé GPT-2)

---

## 11. L'idée d'Olivier : COCONUT comme outil de planification

**Olivier :** Si COCONUT est si miraculeux sur les tâches de planification, pourquoi ne pas l'utiliser juste en tool de planification ?

### 11.1 L'intuition

COCONUT explose sur ProsQA (planification) mais perd sur GSM8k (maths). Donc architecture hybride :

```
Tâche arrive
    │
    ├── "Quel est le plan ?" → COCONUT (latent, BFS, explore les chemins)
    │         │
    │         ▼
    │    Plan optimal trouvé en vecteurs
    │         │
    │         ▼
    └── "Exécute chaque étape" → CoT classique (en mots, vérifiable, précis)
```

### 11.2 PLaT : cette idée existe depuis janvier 2026

[PLaT](https://arxiv.org/html/2601.21358v2) (Planning in Latent Thought) sépare explicitement :

**Le Planner** (lent, latent) :
- Génère des états latents continus, jamais de mots
- Trajectoire stabilisée par lissage EMA (Exponential Moving Average)

**Le Decoder** (rapide, verbal) :
- Reçoit les états latents agrégés
- Traduit en texte lisible
- "Lazy Decoding" : regarde juste le premier token pour savoir si on continue ou s'arrête

Gain : ~56% de tokens en moins qu'un CoT classique.

### 11.3 Résultats de PLaT

| Métrique | PLaT | COCONUT | CoT |
|---|---|---|---|
| Précision (meilleure réponse) | Plus faible | Moyen | Meilleur |
| Pass@128 sur GSM8k | **74.2%** | 66.7% | — |
| Diversité des solutions | **Excellente** | Moyenne | Faible |

Trade-off précision/diversité : PLaT est moins bon en one-shot mais explore un espace de solutions beaucoup plus large. Pour de la planification de refactoring (où on veut plusieurs plans candidats), c'est potentiellement plus intéressant.

### 11.4 L'approche System 1 / System 2 séparé : ça marche pas

Des chercheurs à Munich ont tenté un [LLM gelé (System 1) + coprocesseur séparé (System 2)](https://arxiv.org/html/2510.00494v1) qui envoie des vecteurs de planification.

**Résultat décevant :** un modèle unique avec le même budget de paramètres fait aussi bien. Les tokens latents du coprocesseur montrent une "spécialisation limitée" — ils amplifient la confiance plutôt que d'activer un vrai raisonnement algorithmique.

**Conclusion :** la séparation physique planificateur/exécutant n'est pas la bonne coupe. Le raisonnement latent fonctionne mieux intégré au même modèle, pas externalisé.

### 11.5 La piste prometteuse : deux modes, un seul modèle

Ce qui semble marcher : pas deux modèles séparés, mais **deux modes du même modèle** :

```
Mode 1 : "Planifie" → N itérations latentes (COCONUT/PLaT)
                         Le modèle explore en BFS, sans écrire
Mode 2 : "Exécute" → CoT classique guidé par le plan
                         Le modèle verbalise, étape par étape
```

C'est exactement ce que fait COCONUT avec `<bot>` / `<eot>`. Le chaînon manquant : un mécanisme pour que le modèle **décide lui-même** quand basculer entre les modes. LightThinker commence à explorer cette piste.

### 11.6 Pseudo-architecture hybride idéale

```python
def solve(problem):
    # Phase 1 : planification latente (COCONUT-like)
    plan = model.think_latent(problem, max_steps=8)  # BFS silencieux

    # Phase 2 : plan injecté comme contexte, exécution en CoT verbal
    solution = model.execute_verbose(problem, plan=plan)  # vérifiable

    return solution
```

Le meilleur des deux mondes : l'exploration du latent ET la traçabilité du verbal.

**Sources :**
- [PLaT: Latent Chain-of-Thought as Planning](https://arxiv.org/html/2601.21358v2)
- [System 1 & 2 Communication for Latent Reasoning](https://arxiv.org/html/2510.00494v1)
- [From System 1 to System 2: Survey](https://arxiv.org/abs/2502.17419)

---

## 12. Le CoT est un RNN de pensées — et si on appliquait l'attention dessus ?

*(Ajouté le 12 février 2026 — session présentation IA pour les boss)*

### 12.1 L'analogie structurelle

On a fait un saut qualitatif en passant des RNN aux Transformers au niveau des **tokens** :

```
NIVEAU TOKEN (résolu en 2017) :
  RNN :         token₁ → token₂ → token₃ → token₄
                Séquentiel, chaque token ne voit que le précédent

  Transformer : token₁ ⟷ token₂ ⟷ token₃ ⟷ token₄
                Chaque token "regarde" tous les autres (self-attention)
```

**Réflexion d'Olivier :** le Chain of Thought actuel, c'est exactement la même structure séquentielle, mais un étage au-dessus :

```
NIVEAU RAISONNEMENT (problème ouvert) :
  CoT actuel :  étape₁ → étape₂ → étape₃ → conclusion
                Séquentiel, comme un RNN de pensées

  "Meta-Transformer" : étape₁ ⟷ étape₂ ⟷ étape₃ ⟷ conclusion
                        Chaque étape enrichie par TOUTES les autres
```

L'idée : embedder les étapes de raisonnement comme des "tokens de pensée" et appliquer de l'attention dessus, exactement comme on le fait pour les tokens de texte.

### 12.2 Ce qui existe déjà dans cette direction

#### A. COCONUT et la superposition implicite
Un [paper](https://arxiv.org/html/2505.12514v1) montre que dans COCONUT, chaque vecteur latent est littéralement une **superposition de plusieurs traces de raisonnement** en parallèle. Les patterns d'attention confirment que le modèle encode plusieurs "fronts de recherche" simultanément dans un seul vecteur.

C'est déjà l'idée, mais **implicitement** — le modèle le fait dans un seul vecteur, pas via un mécanisme d'attention explicite entre étapes.

#### B. Native Parallel Reasoner (NPR, décembre 2025)
[NPR](https://www.arxiv.org/pdf/2512.07461) va plus loin : au lieu de raisonner séquentiellement, le modèle apprend à raisonner en **parallèle** — identifier les étapes indépendantes et les traiter simultanément.

Entraînement en 3 phases :
1. Le modèle apprend le raisonnement séquentiel classique (CoT)
2. On lui montre des versions parallélisées (branches indépendantes côte à côte)
3. Il apprend à identifier **lui-même** ce qui peut être parallélisé

Pas exactement de l'attention entre étapes, mais la même intuition : les étapes de raisonnement ne sont pas forcément une chaîne linéaire.

#### C. Diffusion of Thought — le plus proche de l'idée
[Diffusion of Thought](https://proceedings.neurips.cc/paper_files/paper/2024/file/be30024e7fa2c29cac7a6dafcbb8571f-Paper-Conference.pdf) est probablement le plus radical. Au lieu de générer le raisonnement token par token, **toutes les étapes sont générées en parallèle** par un processus de diffusion, puis raffinées itérativement :

```
CoT classique (autorégressif) :
  étape₁ → étape₂ → étape₃ → réponse
  (séquentiel, chaque étape attend la précédente)

Diffusion of Thought :
  [bruit] → [bruit] → [bruit] → [bruit]      (état initial : du bruit partout)
  [ébauche₁] ⟷ [ébauche₂] ⟷ [ébauche₃] ⟷ [ébauche₄]   (débruitage itératif)
  [étape₁] ⟷ [étape₂] ⟷ [étape₃] ⟷ [réponse]           (convergence)
```

Chaque itération de débruitage, toutes les étapes se voient mutuellement et se corrigent. C'est exactement le "Transformer au-dessus des étapes de raisonnement" — sauf que le mécanisme c'est la diffusion au lieu de l'attention.

### 12.3 La récurrence structurelle dans l'histoire de l'IA

```
Niveau 1 (tokens) :
  RNN (séquentiel) → Transformer (attention parallèle)
  Résolu en 2017. Gain massif.

Niveau 2 (raisonnement) :
  CoT (séquentiel) → ??? (attention parallèle entre étapes)
  En cours de résolution. 2024-2026.

Niveau 3 (agents) :
  Pipeline séquentiel → Multi-agents parallèles
  Embryonnaire.
```

À chaque niveau, on passe du "un après l'autre" au "tous ensemble, chacun enrichit les autres". Et à chaque fois ça débloque un saut de performance.

### 12.4 Ce que personne n'a encore fait (à notre connaissance)

Un vrai mécanisme où les étapes de raisonnement sont traitées comme des tokens dans un Transformer dédié :

```python
# Pseudo-architecture "Meta-Transformer de raisonnement"
reasoning_steps = [embed(step) for step in cot_steps]  # N vecteurs
enriched_steps = self_attention(reasoning_steps)         # chaque étape voit les autres
final_answer = decode(enriched_steps)                    # synthèse
```

Avec une couche d'attention **explicite** au-dessus des étapes de raisonnement, traitées comme des "tokens de pensée". Les approches existantes contournent : COCONUT fait de la superposition implicite, NPR parallélise les branches indépendantes, Diffusion of Thought utilise la diffusion comme proxy. Mais l'attention directe entre étapes de raisonnement — c'est un angle potentiellement original.

**Sources :**
- [Reasoning by Superposition in COCONUT](https://arxiv.org/html/2505.12514v1)
- [Native Parallel Reasoner (NPR)](https://www.arxiv.org/pdf/2512.07461)
- [SPRINT - Scaling Parallel Reasoning](https://scalingintelligence.stanford.edu/pubs/sprint.pdf)
- [Diffusion of Thought (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/be30024e7fa2c29cac7a6dafcbb8571f-Paper-Conference.pdf)

---

## 13. Projet ML embarqué : débruitage capteurs et fusion inertielle par réseaux de neurones

*(Ajouté le 12 février 2026)*

### 13.1 L'envie initiale

Comprendre le pipeline ML de bout en bout (dataset → training → eval → déploiement) mais pas sur un cas abstrait type nanoGPT — sur un problème d'**embarqué**, le terrain naturel d'Olivier.

Objectifs concrets :
- Débruiter des signaux physiques (accéléro, gyro)
- Corriger le random walk d'un gyroscope
- Trouver mieux (ou complémentaire) qu'un EKF pour faire un inclinomètre accéléro+gyro
- Déployer sur microcontrôleur (TinyML)

### 13.2 Pourquoi c'est le bon projet

- 42 ans d'embarqué = le domaine est maîtrisé, seul le ML est nouveau
- Les modèles sont minuscules (10-50K params, INT8, ça rentre en 64Ko de Flash)
- Les données existent déjà (logs capteurs)
- Le résultat est mesurable objectivement (comparer au filtre/EKF existant)
- Couvre tout le pipeline : data prep, training, eval, quantization, déploiement embarqué

### 13.3 Pourquoi un réseau de neurones peut battre l'EKF

L'EKF code les hypothèses à la main : bruit gaussien, dynamique linéarisable, matrices de transition explicites. Quand les hypothèses sont fausses (vibrations mécaniques, non-linéarités, dérive thermique), l'EKF souffre.

Un réseau de neurones **apprend** le modèle de bruit à partir des données, y compris les comportements non-gaussiens et non-linéaires qu'on ne saurait pas modéliser analytiquement.

L'approche hybride (EKF + correction neurale) est souvent la plus robuste.

### 13.4 Chemin envisagé

1. Récupérer des logs accéléro+gyro bruts
2. Petit LSTM ou Conv1D en Python : débruitage gyro, comparé à un filtre passe-bas
3. Fusion accéléro+gyro pour inclinaison, comparé à l'EKF existant
4. Export TFLite → déploiement STM32

### 13.5 Outils à explorer

- TensorFlow Lite Micro, STM32Cube.AI, Edge Impulse
- PyTorch → export ONNX → quantification INT8
- nanoGPT de Karpathy reste un bon exercice annexe mais **n'est plus la priorité**

---

## Méta-réflexions de la session

- Les fichiers `.md` de mémoire qu'on utilise sont une forme artisanale de RAG manuel
- L'approche MemGPT/Letta serait un upgrade naturel pour des sessions de code longues
- Pour le refactoring embarqué, un simple script Python + ctags + NetworkX suffirait
- L'intuition d'Olivier sur les espaces interstitiels dans les embeddings rejoint des recherches actives sur l'exploration d'espace latent
- La question du critère signal/bruit dans les régions inexplorées est un problème ouvert fondamental
- COCONUT montre que le langage bride le raisonnement — raisonner en vecteurs continus permet du BFS implicite
- La transcription du "neuralese" en mots humains pourrait enrichir notre vocabulaire de concepts nouveaux
- Le dilemme performance vs transparence du raisonnement latent est non résolu
- Les états hypnagogiques humains sont un analogue biologique du raisonnement latent (COCONUT) — même mécanisme (traversée sans mots), même limitation (inobservable après coup)
- L'idée d'utiliser COCONUT comme outil de planification spécialisé (pas comme remplacement du CoT) rejoint PLaT (janvier 2026)
- La séparation physique System 1 / System 2 ne marche pas — mais deux modes dans le même modèle, oui
- Le chaînon manquant : que le modèle décide lui-même quand basculer latent ↔ verbal
- Le CoT est structurellement un "RNN de pensées" — la même révolution RNN→Transformer pourrait s'appliquer au niveau du raisonnement (attention entre étapes plutôt que séquence)
- Diffusion of Thought (NeurIPS 2024), NPR et COCONUT approchent cette idée par des biais différents, mais l'attention explicite entre étapes de raisonnement reste un angle peu exploré
- Pattern récurrent : séquentiel → parallèle avec enrichissement mutuel, à chaque niveau d'abstraction (tokens, raisonnement, agents)
- **Projet prioritaire "mains dans le cambouis ML"** : débruitage capteurs + fusion inertielle par réseaux de neurones, pas nanoGPT — c'est le projet qui capitalise sur 42 ans d'embarqué
- Un réseau de neurones apprend le modèle de bruit (y compris non-gaussien/non-linéaire) là où l'EKF impose des hypothèses manuelles

---

*Sauvegardé le 2026-02-10 → 2026-02-12*
*La nuit a été courte mais l'hypersphère est vaste.*

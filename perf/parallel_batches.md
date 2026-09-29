# I15.9 — lots parallèles exacts

État initial : banc préparé, campagne non lancée tant que le créneau CPU n'est
pas libéré. Aucun noyau GPU n'est implémenté ici. Voir le protocole utilisateur
`outputs/garamon-parallelisme-protocole.md` dans l'espace de travail Codex.

## Exécution

Depuis la racine du dépôt, avec un répertoire de sortie explicitement choisi :

```sh
julia --startup-file=no --threads=1,0 perf/parallel_batches_grid.jl /chemin/outputs/parallel-smoke --smoke
julia --startup-file=no --threads=1,0 perf/parallel_batches_grid.jl /chemin/outputs/parallel
```

La première commande vérifie 1/2 threads puis 1/2 processus, sur un lot de
32 produits en 4D. La seconde lance séquentiellement 1/2/4/8/16 threads,
puis 1/2/4/8 processus à un thread. Chaque configuration a un nouveau processus
Julia. Le contrôleur de processus constitue un processus supplémentaire ; il
coordonne et matérialise les résultats. GNU `timeout` borne chaque configuration
à 660 secondes (+20 secondes avant arrêt forcé). Aucun fichier journal temporaire
n'est conservé ; les points d'entrée PerfChecker temporaires sont nettoyés.

Pour une configuration particulière :

```sh
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --threads=4,0 --gcthreads=1 --project=perf/controller perf/parallel_batches.jl threads 4 /chemin/resultat.csv
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --threads=1,0 --gcthreads=1 --project=perf/controller perf/parallel_batches.jl processes 4 /chemin/resultat.csv
```

Le banc emploie l'environnement PerfChecker déjà présent ; il n'installe aucun
paquet. Il requiert Julia 1.13 et les dépendances existantes de `perf/controller`.

## Contrat et grille

- Dimensions 4/12/65/128, algèbres euclidiennes diagonales.
- Deux supports : sous-algèbre de 3 directions dispersées (8 coefficients),
  et de `min(n,6)` directions dispersées (16 coefficients en 4D, 64 ensuite).
- Lots 1/32/1024 ; quatre variantes déterministes de coefficients entiers 1..3.
- API `run_product_values!`, ordre `plan.output_masks`, tous les coefficients
  copiés dans une matrice possédée par l'appelant. Ce n'est pas une comparaison
  de sorties `SparseMultiVector` et ce n'est pas une réduction à un checksum.
- Un workspace par tâche ou processus ; plan et entrées partagés en lecture
  seule entre tâches. Aucune sélection du workspace par `threadid()`.
- Un oracle indépendant calcule les inversions des indices de base. Tous les
  coefficients de tous les produits sont contrôlés avant/après chronométrage.
  Chaque somme entière reste exactement représentable en Float64.
- Les processus comparent des entrées déjà résidentes et l'envoi des quatre
  variantes à chaque lot. Ce dernier mode mesure un trafic périodique compact,
  pas un lot de `B` multivecteurs indépendants. Tous les résultats reviennent
  au contrôleur dans les deux modes.
- Les tâches vides lorsque `B < lanes` sont conservées : le coût de mobilisation
  inutile fait partie de l'expérience. Une future politique adaptative peut
  réduire ce nombre ; elle devra être comparée séparément.

## Mesures et limites

Le lanceur mesure le délai système + démarrage Julia + import jusqu'au marqueur
de disponibilité, ainsi que la durée totale de la configuration (qui comprend
toute la campagne et n'est donc pas une latence de produit). Le CSV de calcul
sépare import du banc, démarrage/import du pool, préparation locale/distante,
premier appel et temps de compilation rapporté par Julia. Les appels suivants
fournissent médiane/p95/débit via PerfChecker + BenchmarkTools, 11 échantillons
maximum, `evals=1`. Les mesures sont séquentielles entre configurations.

Le premier appel est « premier pour cette configuration/cette signature dans
ce processus », pas un cache de compilation Julia vierge. Les premiers appels
distants utilisent quatre produits ; le premier lot du contrôleur utilise B.
Les temps de préparation distants sont additionnés car la préparation actuelle
des workers est séquentielle. Ces temps ne doivent pas être ajoutés plusieurs
fois lorsque plusieurs lignes réutilisent la même famille.

Les allocations BenchmarkTools concernent le contrôleur. Une sonde séparée
mesure les allocations de calcul de chaque worker pour le même nombre de
produits ; elle exclut le décodeur/encodeur RPC et s'exécute séquentiellement.
Elle n'est pas un temps parallèle à ajouter à celui du lot. Les pics RSS de
processus sont additionnés de manière conservatrice : ce n'est pas une mesure
instantanée d'usage physique unique, les pages partagées pouvant être recomptées.
Le volume `Serialization.serialize` des entrées est une sonde diagnostique,
sans les en-têtes/protocoles de transport. Les temps encode/decode de cette
sonde unique ne constituent pas une distribution statistique.

Budgets : 4096 chemins/plan, 4 194 304 chemins/lot, 64 Mio de sortie,
64 Mio par état/workspace admis, 256 Mio d'allocations contrôleur/premier lot,
2 Gio de pic RSS contrôleur et 8 Gio de somme des pics du pool, 600 s de
budget logiciel/configuration. Les contrôles RSS se font entre cas ; ce sont
des gardes d'arrêt, pas des plafonds OS. Le pool est admis si au moins
512 Mio × (workers + contrôleur) sont libres. Les 16 processus sont exclus de
cette première grille par budget mémoire. Limites et fichiers proviennent du
banc ; aucun résultat de performance n'est supposé avant exécution.

Les CSV portent une empreinte SHA256 de `Project.toml`, `src/*.jl` récursifs et
des deux fichiers d'exécution. Le lanceur n'est pas inclus dans cette empreinte.
Une modification du noyau durant une configuration fait échouer la validation.
Une campagne interrompue peut n'avoir que les cas passés dans son CSV : le
verdict de campagne et le nombre attendu (24 threads, 48 processus ; smoke
1 et 2) doivent être vérifiés avant agrégation.

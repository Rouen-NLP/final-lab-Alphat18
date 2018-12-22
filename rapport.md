# Classification des documents du procès des groupes américains du tabac.
**Auteur:** Thomas Barbier

## Introduction
Le but de ce TP est de réaliser un classifieur de texte automatique pour les documents récupérés lors du procès des groupes américains du tabac. Le clasifieur sera entrainné sur une petite proportion des textes numérisés pour le procès (3482 textes sur les 14 millions totaux).
Ce rapport présentera tout d'abord une analyse détaillée des textes utilisé pour l'entrainnement du classifieur. Ensuite, le problème de classification automatique sera évalué brièvement afin de justifier l'utilisation d'un clasifieur Bayésien comme solution. Les performances de ce clasifieur seront alors étudiés pour comprendre si la solution s'avère finalement adapté au problème. Enfin, nous chercherons à voir comment il est possible d'améliorer les dites performances.

# 1. Analyse des données

Les données fournies pour le TP sont regroupées dans un dossier nommé _"data/Tobacco3482-OCR"_. A l'intérieur de ce dossier, on trouve 10 sous dossier nommé respectivement: "_Advertisement_, _Email_, _Form_, _Letter_, _Memo_, _News_, _Note_, _Report_, _Resume_ et _Scientific_". Chacun de ces dossier regroupe en fait les textes par catégories. Ce sont par ailleurs ces catégories que nous allons chercher à prédire grâce à notre classifieur.

final-lab-Alphat18/images/nb_documents.png

Le nombre de texte par catégorie n'est pas identique. Il varie assez fortement, ce qui peut avoir des répercusions sur la qualité de l'apprentissage pour certaines catégories. On peut voir le nombre de textes par catégorie sur l'histogramme suivant:On y voit clairement cette disparité. Par exemple, la catégorie "_Memo_" (la plus fournie) possède 620 textes tandis que la catégorie "_Resume_" n'en possède que 120.

Mais le nombre de textes peut facilement être compensé par le nombre de mots présents dans ces textes. En effet, même si une catégorie comme "_Email_" possède beaucoup de textes, ceux-ci s'avèrent en fait assez court dans l'ensemble par rapport à d'autres catégories (106.8 mots en moyenne contre 627.8 pour la catégorie "_News_").

**IMAGE Average Words**

Cette disparité dans le nombre de mots par catégorie risque d'influencer les résultats du classifieur. Par contre, par rapport à la disparité de taille des catégories, le nombre de mots par catégorie peut au contraire s'avérer comme une caractéristique intéressante pour prédire la catégorie. Un texte très court d'environ 50 mots aura bien plus de chance d'être une "_Note_" qu'une "_News_" par exemple.

En Observant quelqu'un des textes en eux même, on se rend compte d'un problème qui risque de fortement impacter les résultats. Certains textes s'avèrent être très mal numériser et ont perdu une grande partie de leur signification. Si on prends par exemple le premier texte contenu dans la catégorie "_Advertisement_", on obtient le message suivant:
>  A Mpertant as yar
sesiye teaetered cabiieess. Baely
doesn’) keow bea te
Bitton Aau-Fotne bl resin syste. Cant
viduiiliy crafted. Parenter
tiott, Most eapennese liste rn siichinng
Holimars. Costlicr of course
“Has Oetenined -
Wainy: Thy
ie Hoel? h.
That Cia
Marg a Féme awe ii na eager ref Hizon a ol
Ra a
6P9S70099

Après une rapide recherche, le texte qu'il contient ne provient d'aucun language connu. Il est donc évident que la numérisation de ce document a très mal fonctionné et a donné des mots incompréhensibles. Seuls quelques mots anglais subsistent à certains endroits (on peut reconnaitre _can't_, _crafted_, _of course_, ou encore _that_). Ce problème de numérisation va énormément affecter la qualité de notre classifieur. De fait, même un humain ne serait sans doute pas capable de reconnaitre ici une publicité.

Heureusement, d'autres textes ont été bien mieux numérisés. En voici un exemple tiré de la première "_Lettre_":
> Letter to the Editor of Personnel Administrator
To the Editor:
As the owner of a company with about 100 employees, I found
Lewis Solmon's recent article, "The other side of the smoking
worker controversy,” interesting. I'd like to share my own
views, since I've had some experience with the issue.
A small but vocal group of employees have pressured me to
either ban smoking or to segregate smokers and nonsmokers.
When they first approached me I considered the situation but,
for several reasons, decided against any restrictions.
First, implementing smoking policies would have required ;
that I take action against good employees who have worked
for me for quite some time. Second, to implement a smoking
policy would have disrupted my company's work process, since,
as in many offices, employees with similar skills and responsi-
bilities work together.
Furthermore, once I took a hard look at the situation, I dis-
covered that the vast majority of my employees were neither
aware of nor particularly interested in the problem.
I have not read any of Weis' articles to which Solmon referred
nor have I considered the economic aspect of the argument on
which Solmon's article was based. But common sense suggcsts
that rearranging people, changing policies, implementing re-
strictions, and disrupting my workforce won't save me moncy.
Sincerely,
TIOK 0027645

Ce sont principalement les textes bien numérisés qui permettront de classifier efficacement par catégorie. Certaines solutions pour gérer les textes mal numérisés seront envisagés plus tard dans la partie sur les possibles améliorations.

# 2. Analyse du problème et justification de la solution.

Le problème de classification de texte est un problème très connu en **Machine Learning**. De nombreuses solutions existent pour le résoudre.

### 1. Extraction de caractéristiques.
Avant toute chose, l'utilisation brute du texte pour l'entrainnement d'un classifieur est la plupart du temps déconseillé. Il faut donc tout d'abord extraire des caractéristiques de ce dernier afin de lui en soutirer des informations pertinentes. De nombreuses caractéristiques peuvent être extraite du texte. Nous avons déjà parler du nombre de mots par texte, mais bien d'autres existent, tels que le nombre de caractères par texte, la taille moyenne des mots ou encore l'utilisation des caractères spéciaux ou des nombres...

Mais ces caractéristiquent s'avèrent malgré tout assez limitées. Il existe en effet des représentations plus évoluées et efficaces. La plus simple d'entre elles et une des plus utilisées (elle correspond en quelque sorte à la méthode _historique_ de l'éxtraction de caractéristiques dans du texte) est le **Bag of Words** (Sac de Mots). Cette méthode revient à compter le nombre d'instances de certains mots clés dans un texte. Ces mots clés sont tout simplement les mots les plus présents dans le corpus en général. Un texte sera donc représenté par  un vecteur de dimension le nombre de mots clés utilisés. Ce vecteur sera notamment le plus souvent assez vide (beaucoup de valeurs seront nulles), d'où l'utilisation de matrices **sparses** pour la création de la matrice de caractéristiques afin d'accélerer les calculs. 

**BOW EXAMPLE**

Une alternative efficace au Bag of Words est la représentation **TF-IDF** (Term Frequency - Inverse Document Frequency). Il fonctionne de la même manière mais applique en plus une pénalisation basé sur la fréquence du mot dans le corpus. Cela permet de donner plus d'importance à des mots plus significatifs. Par exemple, le mot _the_ en anglais est de très loin le mot le plus fréquent alors qu'il ne comporte que peu d'information quand à la catégorie d'un texte comparé à d'autres mots tels que _letter_ ou _sincerely_.

**TF-IDF EXAMPLE**

Une autre amélioration possible du Bag of Words est l'utilisation des **N-grams**. Un N-gram est simplement un arrangement de N mots ensemble. Si on considère un seul mot, comme dans la version initiale du Bag of Words, on parle d'uni-gram. Deux mots ensembles (tels que _vast majority_) correspondent à un bi-gram... L'introduction des N-grams dans la représentation peut permettre d'accorder plus d'importance au lien entre certains mots.

**N-GRAM EXAMPLE**

Il existe bien évidement de très nombreuses autres représentations pour le texte. Plus récement, des avancées ont été faites sur des nouveaux modèles qui semblent très efficaces, tel que le **Word Embedding** avec notamment le modèle **Word2Vec**. De même, la représentation à l'aide d'arbres semble donner des résultats intéressant. Par ailleurs, nous ne nous intéresseront pas en détail à ces techniques dans ce rapport pour des raisons de temps et de complexité.

### 2. Classifieur.
 
Une fois la matrice de caractéristique extraite du texte, il nous faut désormais choisir un algorithme de classification. Il en existe encore une fois un nombre impréssionants.

Récement, on trouve de nombreux classifieurs basés sur des **réseaux de neuronnes** dont l'efficacité semble actuellement inégalé. Par ailleurs, ils s'avèrent être plus complexe et couteux à entrainner que des classifieurs plus classiques. De plus, ils nécessitent de larges quantitées de données annotées pour donner de bon résultat;

Dans notre cas, nous avons considéré que les données ne se prétaient pas réellement à l'utilisation de ces derniers (on ne possède à près tout que 3282 documents ce qui est assez réduit). C'est pourquoi nous avons envisager l'utilisation d'un algorithme plus simple et qui a déjà fait ses preuves plus d'une fois, un **Classifieur Bayes Multinomial**. C'est une variante du clasifieur **Naive Bayes** fonctionnant sur des données multinomiales. Il est simple et rapide à l'utilisation tout en s'avèrant particulièrement adapté à la classification textuel. De plus, il fonctionne bien avec les représentation vu au paragraphe précédent.

# 3. Analyse des performances.

Nous avons finalement implémenté un classifieur Bayésien que nous avons testé sur deux représentation, le Bag of Words et le TF-IDF. Nous avons par ailleurs permis à ces deux représentations l'utilisation des N-gram afin d'améliorer un peu la classification.

Nous avons découper notre corpus en un jeu d'aprentissage et un jeu de test. Nous avons choisi un ratio entre les deux de 80% (training) / 20% (test) car le nombre de données est assez limité (d'où un jeu d'aprentissage assez grand).

### 1. Optimisation des paramètres.

Une partie importante pour augmenter au maximum la performance fut l'optimisation des paramètres de la représentation en Bag of Words, TF-IDF ainsi que du classifieur Bayésien.
Les paramètres à optimiser pour le Bag of Words sont:
  * **max_features** -> integer: le nombre de mots clés utilisés dans la représentation. Ce paramètre définit donc la dimension colonne de la matrice des caractéristiques.
  * **ngram_range** -> tuple: Les types de N-gram utilisés.
  * **max_df** -> float (entre 0 et 1): Permet d'ignorer les termes qui ont une fréquence supérieur à ce paramètre (notament utile pour la suppression de certains caractères spéciaux très fréquents).
  * **min_df** -> float (entre 0 et 1): Permet d'ignorer les termes qui ont une fréquence inférieure à ce paramètre (utile pour la suppression de certains caractères spéciaux peu fréquents).

La représentation TF-IDF s'appuie directement sur le Bag of Words et se contente de le transformer par pénalisation. Il utilise un paramètre optimisable:
  * **use_idf** (booléen): Permet d'autoriser ou non le changement des poids selon la fréquence inverse par document. 
 
Enfin, le classifieur Bayésien possède un paramètre:
  * **alpha** -> float (entre 0 et 1): Paramètre de lissage.

L'optimisation se fait par recherche exhaustive de l'espace des paramètres, ce qui est assez couteux en temps. Pour autant, cela permet d'obtenir des scores globalement meilleurs que sur une sélection manuelle des parmètres qui s'avère souvent fastidieuse.

### 2. Résultats.

Après optimisation des paramètres, on obtient pour la représentation **Bag of Words** le rapport de classification suivant:

**CLASSIFICATION REPORT**

On voit donc que le **f1 score** est de **0.72** en moyenne. Ces résultats varient légèrement en fonction du découpage aléatoire du corpus en jeu d'apprentissage et jeu de test.

Pour la représentation **TF-IDF**, les résultats sont comparables dans l'ensemble:

**CLASSIFICATION REPORT**

 Nous obtenons un **f1 score** qui tourne autour de **0.71**.
Dans les deux cas, les cores en fonction des catégories varie énormément. Certaines catégories donnent de très bon résultats, notamment la catégorie "_Resume_" qui donne presque toujours un score parfait de 1 ou encore "_Email_" (souvent proche de 0.8-0.9). Au contraire, la catégories "_Note_" est presque toujours la plus basse (0.3-0.4) et la classification des "_Report_" est souvent médiocre (plutôt proche de 0.5-0.6).

Comme nous l'avons vu dans la partie 1, la catégorie "_Note_" possède peux de mots en moyenne mais aussi peu de documents, ce qui explique sans doute son faible score. L'explication pour les "_Report_" est déjà plus difficile (il se peut que beaucoup soient souvent mal numérisé, mais de plus amples recherches seraie,t à effectuer).

Le gros succès de la catégorie "_Resume_" est elle aussi un peu étrange. C'est la catégorie possédant le moins de documents (mais elle possède un bon nombre de mots par document). Il doit donc s'avérer que le texte quel contient est particulièrement facile à reconnaitre pour une représentation de type Bag of Words.

Au final, les résultats de notre classifieur ne sont pas exceptionnels sans pour autant être réellement mauvais. Une majorité des textes sont bien classifiés, ce qui n'est pas évident lorsque l'on a le choix entre 10 catégories possibles. Ces résultats sont d'autant plus encourageants au vu de certains textes très mal numérisés et donc virtuellement inclassables. Ces derniers ne peuvent que peser lourdement sur le score final.

# 4. Pistes d'améliorations. 

Afin d'améliorer notre classification, plusieurs pistes sont enviseagables. la première et probablement la plus sencée serait tout simplement de venir effectuer une action directement sur les données sources. tout d'abord, plus de données d'entrainnements, notamment sur les catégories qui en ont peu, permettrait de limiter la disparité du nombre de documents par classe tout en améliorant sans doute la qualité de l'apprentissage.

De plus, posséder plus de données nous permettrait d'éliminer sans risque tous les textes mal numérisés qui viennent troubler l'apprentissage en apportant des informations éronées. Par ailleurs, leurs supression même dans notre corpus actuel de 3482 documents pourrait s'avérer bénéfique.

Ensuite, le choix d'une représentation plus moderne (telle que **Word2Vec** par exemple) pourrait certainement donner de meilleurs résultats. De même, un classifieur basé sur un réseau de neuronne améliorerait sans doute légèrement le score général si bien réalisé.

Enfin, l'introduction de certaines caractéristiques, notamment la nombre de mots par documents, pourrait possiblement permettre d'aider légèrement à la classification, notamment parce que cette caractéristique reste assez valide même si le texte est mal numérisé. le problème consisterait alors à trouver une répresentation les intégrants efficacement. En effet, le Bag of Words semble peut adapté à ce genre d'opérations.

# Learn_Your_Courses_Intro_to_Programming
Gain the skills you need to do independent data science projects
![Do like this image](https://img.freepik.com/photos-gratuite/vue-dessus-pirate-informatique-non-reconnaissable-effectuant-cyberattaque-nuit_1098-18706.jpg?semt=ais_hybrid)


# Intro_to_Programming
--------
Bienvenue sur ce dépôt git consacré au Challenge DATA.

Thème : Prédire de manière séquentielle l'évolution d'une carrière
par HrFlow.ai. 

Pour plus d'informations concernant le sujet prière de vous rendre sur le [Site web du Challenge ](https://challengedata.ens.fr/participants/challenges/151/).


# Sujet :
Dans certaines entreprises, l'évolution de la carrière des individus suit une trajectoire relativement linéaire, marquée par plusieurs étapes ou phases correspondant à des postes distincts au sein de la hiérarchie de l'entreprise. Cette progression peut être assimilée à des stations de métro, où les individus traversent ces étapes et peuvent éventuellement s'arrêter, que ce soit en quittant l'entreprise ou en modifiant leurs aspirations professionnelles, de manière similaire aux voyageurs descendant à une station spécifique.


Nous avons retenu comme sujet l'étude sur l'évolution de la carrière des individus suit une trajectoire relativement linéaire(consider comme l'un des moyens qui peut permettre la prédiction des évolutions de carrières professionnelles). Concretement nous allons entrainer un certians notre de modle en prenant pour vecteur d'entre x_train : variables explicatives pour l’entraînement,x_test : variables explicatives pour le test,y_train : variable(s) cible(s) pour l’entraînement, y_test : variables explicatives pour le test
Exemple de soumission aléatoire.

# Objectif :
--------
Anticiper la position probable qu'un employé atteindra au sein d'une entreprise. Cette prédiction repose sur l'évaluation d'informations concernant l'employé, telles que les qualifications, l'expérience, les compétences ou les intérêts, ainsi que des détails sur l'entreprise, tels que le secteur, le type d'activité, le style de gestion ou la taille.

Il est important de noter que la définition de ce problème est simplifiée et ne reflète la véritable complexité de l'évolution de carrière. En réalité, il est peu probable que des informations préalables sur l'employé ou l'entreprise déterminent les parcours professionnels. De plus, de nombreuses entreprises ne disposent pas d'une hiérarchie clairement définie et verticale, ce qui entraîne des trajectoires de carrière plus complexes et variées.

# Pré-requis

Afin d'exécuter l'ensemble des scripts présents dans ce projet, vous devez disposer d'un environnement python (>= 3.9) et d'installer les packages nécessaires par la commande suivante :

```bash
pip install -r requirements.txt
```

***
 # Les modèles qui vont être explorés dans la réalisation de la tâche sont les suivants :
 
|  Modèle   |Module python| 
| --------|-----------------|
| Random Forest|skleal.ensemble.RandomForestClassifier|
| Support Vector Machines |sklearn.svm.SVC|
| K-Nearest Neighbor |sklearn.neighbors.KNeighborsClassifier|
|Deep-learning |torch|
| XGBoost Classifier|xgboost.XGBClassifier|


# Structure du dépot:

La structure de dépot qu'il faudra avoir en local chez chacun des contributeurs les la suivante : 

- __data__ : Dossier contenant les fichiers d'entraînement et de test originaux. Ces fichiers originaux doivent être dans le `.gitignore` afin de ne pas apparaitre sur le dépot en ligne.
- __images__ : Dossier contenat les images utiles pour la communication des résultats. Selon l'utilisateur, ils devront être précédés d'un préfixe `user_`.
- __notebook__ : Dossier contenant des notebooks. Selon le contributeur, ils devront être précédé d'un préfixe `user_`
- __src__        
  - **`/data`** : Dossier où on retrouve (format csv) les jeux de données utilisés pour la validation des modèles. Ces jeux de données sont formés uniquement à partir de la base de train et devont être utilisés par tous dans la validation des modèles avant le test final.  
  - **`/models`** : Les modèles entrainés et enregistrés au format pickle (pkl).     
        
  - **`/tools`** : On y retrouve tous les codes Python, en particulier les fonctions de sélections de variables, les fonction de gridsearch, ainsi que des fonctionnalités supplémentaires afin de former et enregistrer les modèles.            
- __README.md__        
- __requirements.txt__ : Fichier contenant la liste de tous les modules nécessaires à l'éxecution des codes Python du projet.   

--------
## Données
--------
Le jeu de données comprend 36K carrières, dont 29K constituent l'ensemble d'entraînement et 7K forment l'ensemble de test. Ces carrières proviennent de 22K employés distincts, sans chevauchement entre les ensembles d'entraînement et de test, et impliquent 5K entreprises différentes.

Il est important de noter le déséquilibre significatif dans les données entre les quatre positions distinctes. Plus précisément, "Assistant", "Executive", "Manager" et "Director" représentent environ 29,5 %, 63 %, 3,5 % et 4 % des carrières, respectivement.


Toutes les données nécessaires au challenge ont été récupérées, nettoyées et mises sous une forme exploitable. Vous trouverez ici un descriptif complet et détaillé de nos données brutes. (se retrouvent dans le répertoire [Description complète des données ](https://challengedata.ens.fr/participants/challenges/151/)
 ***
## Description des fichiers
Le dataset est structuré de la manière suivante :

Carrière (X train / test) :

ID de carrière : Un numéro d'identification unique pour chaque carrière.
Embedding de l'employé : Un vecteur intégrant les qualifications et l'expérience professionnelle de l'employé.
Embedding de l'entreprise : Un vecteur intégrant des informations sur l'entreprise.


Position (y train / test) :

ID de carrière : Un numéro d'identification unique pour chaque carrière.
Poste (TARGET) : Le poste atteint par l'employé au sein de l'entreprise ("Assistant", "Executive", "Manager", "Director").


Les embeddings vectoriels captent des informations textuelles concernant à la fois l'employé et l'entreprise. Ces embeddings ont été générés en utilisant BERT, un modèle reconnu pour sa performance et sa versatilité dans les tâches de compréhension du langage naturel.

Le format de sortie pour les prédictions doit être conforme au format fourni, associant chaque ID de carrière à la position prédite.

Il y a quatre positions différentes (ou étapes de carrière) qu'un employé peut atteindre dans une entreprise :
- Assistant
- Executive
- Manager
- Director

# Répartition des données

![Data_repartition](https://github.com/rbouna/Challenge_DATA/blob/main/images/data_repartition.png)

# Description de la métrique

Le **F1 score** est une mesure de précision pour les systèmes de classification binaire. Il est basé sur la **précision** et le **rappel**. La précision (P) est le nombre de vrais positifs divisé par le nombre total de positifs prédits, tandis que le rappel (R) est le nombre de vrais positifs divisé par le nombre total de positifs réels. Le F1 score est la moyenne harmonique de la précision et du rappel, et est calculé comme suit:

$$ F1 = 2 \times \frac{P \times R}{P + R} $$

Dans les cas multiclasse, il existe le **F1 score macro** et le **F1 score pondéré**.

Le **F1 score macro** calcule le F1 score pour chaque classe indépendamment et ensuite fait la moyenne des scores. C'est utile lorsque vous avez un déséquilibre de classe car il donne un poids égal à chaque classe.

Le **F1 score pondéré** donne plus de poids aux classes avec plus d'instances. Il calcule le F1 score de chaque classe puis fait une moyenne pondérée, où le poids est le nombre d'instances réelles pour chaque classe. C'est utile lorsque vous voulez prendre en compte l'importance relative des classes.

Dans le cadre de notre projet, nous utiliserons le **F1 score macro** afin d'acconrder un poids égal à chaque classe.

# La liste des Tâches à réaliser
--------
- [x] Récupérer les données en local et les ajouter au .gitignore pour ne pas les exposer en ligne ;
- [ ] Configurer l'environnement python grâce au fichier **`requirements.txt`** ;
- [x] Charger et décompresser les données ;
- [x] Nettoyer les données récupérées ;
- [ ] Utiliser la base d'entraînement pour sélectionner les  modèles selon le F1-score ;
- [ ] Tester les modèle sur la base de test ;
- [ ] Soumettre notre travail au challenge.


# Contributeurs

- Rodolphe Bounamari **`@rbouna`** ;
- Kassim Cisse **`?`**
- Jaurès Ememaga **`@ElBaron86`**.
- test

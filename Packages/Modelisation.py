from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import pandas as pd

def split_data(data) :
    # Division des données en jeu d'entraînement et de test
    X=data.drop(columns=['Survived'])
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# Fonction 5 : Entraînement et évaluation du modèle
def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle Random Forest et calcule les scores.

    Arguments:
    - X : pd.DataFrame, les variables explicatives.
    - y : pd.Series, la variable cible.

    Retourne:
    - train_score : float, score d'entraînement.
    - test_score : float, score de test.
    """

    # Initialisation et entraînement du modèle
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prédictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Scores
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)

    return train_score, test_score

def creer_modele_adaboost(X_train, X_test, y_train, y_test):
    """
    Crée et entraîne un modèle AdaBoost.

    Paramètres :
        - X_train : Données d'entraînement (features)
        - y_train : Labels d'entraînement
        - X_test : Données de test (features)
        - y_test : Labels de test

    Retourne :
        - modele : Modèle entraîné (GridSearchCV)
        - accuracy : Précision sur les données de test
    """
    # Définir le pipeline avec un AdaBoostClassifier
    pipe = Pipeline([
        ('adaboost', AdaBoostClassifier())
    ])

    # Définir une grille de paramètres pour AdaBoost
    param_grid = [

               {
            'adaboost__base_estimator': [DecisionTreeClassifier()],
            'adaboost__base_estimator__max_depth': [1, 5, 10],
            'adaboost__n_estimators': [50, 100, 200],
            'adaboost__learning_rate': [0.1, 0.5, 1.0]
        }
    ]

    # Configurer GridSearchCV
    modele = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

    # Effectuer l'entraînement
    modele.fit(X_train, y_train)

    # Prédire sur les données de test
    y_pred = modele.predict(X_test)

    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)

    return modele, accuracy


def entrainer_adaboost(X_train, X_test, y_train, y_test):
    """
    Crée et entraîne un modèle AdaBoost.

    Paramètres :
        - X_train : Données d'entraînement (features)
        - y_train : Labels d'entraînement
        - X_test : Données de test (features)
        - y_test : Labels de test

    Retourne :
        - modele : Modèle entraîné (GridSearchCV)
        - accuracy : Précision sur les données de test
    """
    # Définir le pipeline avec un AdaBoostClassifier
    pipe = Pipeline([
        ('adaboost', AdaBoostClassifier())
    ])

    # Définir une grille de paramètres pour AdaBoost
    param_grid = [

               {
            'adaboost__base_estimator': [DecisionTreeClassifier()],
            'adaboost__base_estimator__max_depth': [1, 5, 10],
            'adaboost__n_estimators': [50, 100, 200],
            'adaboost__learning_rate': [0.1, 0.5, 1.0]
        }
    ]

    # Configurer GridSearchCV
    modele = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

    # Effectuer l'entraînement
    modele.fit(X_train, y_train)

    # Prédire sur les données de test
    y_pred = modele.predict(X_test)

    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)

    return modele, accuracy
def creer_modele_adaboost(X_train, X_test, y_train, y_test):
    """
    Crée et entraîne un modèle AdaBoost.

    Paramètres :
        - X_train : Données d'entraînement (features)
        - y_train : Labels d'entraînement
        - X_test : Données de test (features)
        - y_test : Labels de test

    Retourne :
        - modele : Modèle entraîné (GridSearchCV)
        - accuracy : Précision sur les données de test
    """
    base_estimator = DecisionTreeClassifier(max_depth=5)

    # Configurer le modèle final Adaboost
    modele = AdaBoostClassifier(base_estimator=base_estimator, learning_rate= 0.1,n_estimators= 50)

    # Effectuer l'entraînement
    modele.fit(X_train, y_train)

    return modele 


def predire(modele, X_test):
    """
    Prédit les résultats à l'aide d'un modèle, affiche le nombre de prédictions > 0.5 
    et retourne les résultats sous forme de DataFrame Pandas.
    
    Arguments :
    - modele : le modèle entraîné utilisé pour faire des prédictions.
    - X_test : les données d'entrée pour la prédiction.
    
    Retourne :
    - Un DataFrame contenant les prédictions.
    """
    # Obtenir les prédictions
    y_pred = modele.predict(X_test)
    
    # Compter les survivants
    nb_pred_sup_05 = (y_pred > 0.5).sum()
    pourcentage = (nb_pred_sup_05 / len(y_pred)) * 100

    # Afficher le nombre de survivants
    print(f"Pourcentage de survivants :  {pourcentage:.2f}%")
    
    # Créer un DataFrame avec les prédictions
    df_predictions = pd.DataFrame(y_pred, columns=["Predictions"])
    
    # Retourner le DataFrame
    return df_predictions


def exporter_resultat(X_test, y_pred, output_file="resultats_predictions.csv"):
    """
    Fusionne les données d'entrée (X_test) avec les prédictions (y_pred) 
    dans un seul DataFrame Pandas et exporte le résultat dans un fichier CSV.
    
    Arguments :
    - X_test : le DataFrame des données d'entrée.
    - y_pred : les prédictions du modèle (array-like).
    - pred_column_name : le nom de la colonne pour les prédictions (par défaut "Predictions").
    - output_file : le nom du fichier CSV de sortie (par défaut "resultats_predictions.csv").
    
    Retourne :
    - Un DataFrame Pandas fusionné contenant X_test et les prédictions.
    """
    # Vérifier que y_pred a la même longueur que X_test
    if len(X_test) != len(y_pred):
        raise ValueError("Le nombre de lignes de X_test et y_pred doivent être égaux.")
    
    # Convertir y_pred en DataFrame avec un nom de colonne
    df_predictions = pd.DataFrame(y_pred)
    
    # Fusionner X_test et les prédictions
    df_resultat = pd.concat([X_test, df_predictions], axis=1)
    
    # Exporter le DataFrame en CSV
    df_resultat.to_csv(output_file, index=False)
    print(f"Les résultats ont été exportés dans le fichier : {output_file}")

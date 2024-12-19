def split_data(data) :
        # Division des données en jeu d'entraînement et de test
    X = data.drop(columns=['Survived', 'Name', 'Ticket'])
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

def creer_modele_adaboost(X_train, y_train, X_test, y_test, n_estimators=50, learning_rate=1.0, base_estimator=None):
    """
    Crée et entraîne un modèle AdaBoost.

    Paramètres :
        - X_train : Données d'entraînement (features)
        - y_train : Labels d'entraînement
        - X_test : Données de test (features)
        - y_test : Labels de test
        - n_estimators : Nombre d'estimateurs faibles (default=50)
        - learning_rate : Taux d'apprentissage (default=1.0)
        - base_estimator : Estimateur de base (default=None, DecisionTreeClassifier utilisé par défaut)

    Retourne :
        - modele : Modèle entraîné
        - accuracy : Précision sur les données de test
    """
    # Si aucun estimateur de base n'est spécifié, utiliser un arbre de décision simple
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(max_depth=1)

    # Créer le modèle AdaBoost
    modele = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    # Entraîner le modèle
    modele.fit(X_train, y_train)

    # Prédire sur les données de test
    y_pred = modele.predict(X_test)

    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)

    return modele, accuracy
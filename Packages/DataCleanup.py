def handle_missing_values(data):
    """
    Remplace les valeurs manquantes pour les colonnes spécifiques.

    Arguments:
    - data : pd.DataFrame, le DataFrame contenant les données.

    Retourne:
    - pd.DataFrame, le DataFrame avec les valeurs manquantes traitées.
    """
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data.dropna(subset=['Embarked'], inplace=True)  # Supprimer les lignes où 'Embarked' est NaN
    return data
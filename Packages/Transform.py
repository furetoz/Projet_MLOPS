import pandas as pd


def create_dummies(data, columns=None, drop_first=False):
    """
    Transforme les variables qualitatives d'un DataFrame en variables dummy.

    Arguments :
    - data : pd.DataFrame, le DataFrame contenant les données.
    - columns : list ou None, les colonnes à convertir. Si None, toutes les variables qualitatives seront transformées.
    - drop_first : bool, si True, supprime la première catégorie pour éviter le problème de multicolinéarité.

    Retourne :
    - pd.DataFrame, le DataFrame avec les colonnes qualitatives transformées en dummies.
    """
    if columns is None:
        columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Vérification que les colonnes existent dans le DataFrame
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    # Création des variables dummy
    data_with_dummies = pd.get_dummies(data, columns=columns, drop_first=drop_first)

    return data_with_dummies

def categorize(value, lower_bound, upper_bound, q1, q3):
    """
    Catégorise une valeur en fonction des seuils définis.

    Arguments :
    - value : float, la valeur à catégoriser.
    - lower_bound : float, limite inférieure.
    - upper_bound : float, limite supérieure.
    - q1 : float, premier quartile.
    - q3 : float, troisième quartile.

    Retourne :
    - str, la catégorie de la valeur.
    """
    if value < lower_bound:
        return 'Très bas'
    elif lower_bound <= value < q1:
        return 'Bas'
    elif q1 <= value <= q3:
        return 'Moyen'
    elif q3 < value <= upper_bound:
        return 'Haut'
    else:
        return 'Très haut'

def transform_quantitative_to_categories(data, columns):
    """
    Transforme des colonnes quantitatives en catégories basées sur l'écart interquartile (IQR).

    Arguments :
    - data : pd.DataFrame, le DataFrame contenant les données.
    - columns : list, les noms des colonnes quantitatives à transformer.

    Retourne :
    - pd.DataFrame, un DataFrame avec les colonnes catégorisées.
    """
    categorized_data = pd.DataFrame()

    for column in columns:
        if column in data.columns:  # Vérifier que la colonne existe
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1

            # Définir les limites pour les classes
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Appliquer la fonction de catégorisation à la colonne
            categorized_data[column] = data[column].apply(
                lambda value: categorize(value, lower_bound, upper_bound, q1, q3)
            )
        else:
            raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    return categorized_data

def create_dummies(data, columns=None, drop_first=False):
    """
    Transforme les variables qualitatives d'un DataFrame en variables dummy.

    Arguments :
    - data : pd.DataFrame, le DataFrame contenant les données.
    - columns : list ou None, les colonnes à convertir. Si None, toutes les variables qualitatives seront transformées.
    - drop_first : bool, si True, supprime la première catégorie pour éviter le problème de multicolinéarité.

    Retourne :
    - pd.DataFrame, le DataFrame avec les colonnes qualitatives transformées en dummies.
    """
    if columns is None:
        columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Vérification que les colonnes existent dans le DataFrame
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    # Création des variables dummy
    data_with_dummies = pd.get_dummies(data, columns=columns, drop_first=drop_first)

    return data_with_dummies

def transform_all_columns(data):
    """
    Transforme toutes les colonnes d'un DataFrame :
    - Les colonnes quantitatives sont transformées en catégories.
    - Les colonnes qualitatives sont transformées en variables dummy.

    Arguments :
    - Train_1 : pd.DataFrame, le DataFrame contenant les données.

    Retourne :
    - pd.DataFrame, le DataFrame transformé.
    """
    # Liste des colonnes quantitatives et qualitatives
    quantitative_columns = ["Age", "SibSp", "Parch", "Fare"]
    qualitative_columns = ["Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"]

    # Vérification des colonnes existantes pour éviter les erreurs
    quantitative_columns = [col for col in quantitative_columns if col in data.columns]
    qualitative_columns = [col for col in qualitative_columns if col in data.columns]

    # Transformation des colonnes quantitatives
    transformed_quantitative = transform_quantitative_to_categories(data, quantitative_columns)

    # Transformation des colonnes qualitatives
    transformed_qualitative = create_dummies(data, qualitative_columns, drop_first=False)

    # Combiner les deux transformations
    df_combined = pd.concat([transformed_quantitative, transformed_qualitative], axis=1)
    columns_to_drop = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 
                       'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    df_final = df_combined.drop(columns=[col for col in columns_to_drop if col in df_combined.columns], errors='ignore')

    return df_final
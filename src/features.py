import pandas as pd
from sklearn.preprocessing import LabelEncoder

def crear_variables_basicas(df):

    df = df.copy()

    # Convertir la fecha
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Crear año
    df["year"] = df["date"].dt.year

    # Variable objetivo: diferencia de goles
    df["goal_diff"] = df["home_score"] - df["away_score"]

    # Variable binaria: ganó el local
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    return df

def codificar_categorias(df):

    df = df.copy()
    encoder = LabelEncoder()

    # Equipos
    df["home_team_encoded"] = encoder.fit_transform(df["home_team"])
    df["away_team_encoded"] = encoder.fit_transform(df["away_team"])

    # Torneo
    df["tournament_encoded"] = encoder.fit_transform(df["tournament"])

    return df

def seleccionar_features(df):

    df = df.copy()

    # Variables que usaremos en el modelo
    use_features = [
        "year",
        "neutral",
        "home_team_encoded",
        "away_team_encoded",
        "tournament_encoded"
    ]

    # Verificacion de existencia
    for col in use_features:
        if col not in df.columns:
            raise ValueError(f"La columna necesaria '{col}' no existe en el DataFrame.")

    X = df[use_features].copy()
    return X


def preparar_X_y(df):

    df = crear_variables_basicas(df)
    df = codificar_categorias(df)

    # Variable objetivo (diferencia de goles)
    y = df["goal_diff"].copy()

    # Features
    X = seleccionar_features(df)

    return X, y


if __name__ == "__main__":
    path = "../data/raw/results.csv"
    df = pd.read_csv(path)

    X, y = preparar_X_y(df)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X.head())
    print(y.head())

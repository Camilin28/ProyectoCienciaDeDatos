import pandas as pd
from sklearn.preprocessing import LabelEncoder

def crear_variables_basicas(df):
    df = df.copy()

    # Convertir la fecha
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Crear año del partido
    df["year"] = df["date"].dt.year

    # Variable objetivo: diferencia de goles
    df["goal_diff"] = df["home_score"] - df["away_score"]

    # Variable binaria: ganó el local
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    return df

def crear_features_historial(df):

    df = df.sort_values("date").copy()

    # Promedio de los goles locales los últimos 5 partidos
    df["home_last5_goals"] = (
        df.groupby("home_team")["home_score"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Promedio de los goles visitantes los últimos 5 partidos
    df["away_last5_goals"] = (
        df.groupby("away_team")["away_score"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Promedio de la diferencia de goles en últimos 5 partidos del local
    df["home_last5_diff"] = (
        df.groupby("home_team")["goal_diff"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Promedio de la diferencia de goles en últimos 5 partidos del visitante
    df["away_last5_diff"] = (
        df.groupby("away_team")["goal_diff"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df

def codificar_categorias(df):

    df = df.copy()

    encoder_team = LabelEncoder()
    encoder_tournament = LabelEncoder()

    # Codificar equipos
    df["home_team_encoded"] = encoder_team.fit_transform(df["home_team"])
    df["away_team_encoded"] = encoder_team.fit_transform(df["away_team"])

    # Codificar torneo
    df["tournament_encoded"] = encoder_tournament.fit_transform(df["tournament"])

    return df


def seleccionar_features(df):

    columnas = [
        "year",
        "neutral",
        "home_team_encoded",
        "away_team_encoded",
        "tournament_encoded",
        "home_last5_goals",
        "away_last5_goals",
        "home_last5_diff",
        "away_last5_diff"
    ]

    # Verificación de columnas
    for col in columnas:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no está disponible en el DataFrame.")

    X = df[columnas].copy()
    return X


def preparar_X_y(df):

    # 1. Variables básicas
    df = crear_variables_basicas(df)

    # 2. Variables históricas
    df = crear_features_historial(df)

    # 3. Codificar categorías
    df = codificar_categorias(df)

    # 4. Variable objetivo
    y = df["goal_diff"].copy()

    # 5. Features
    X = seleccionar_features(df)

    # 6. Eliminar filas con nulos provocados por rolling
    X = X.dropna()
    y = y.loc[X.index]

    return X, y


if __name__ == "__main__":
    path = "data/results.csv"
    df = pd.read_csv(path)

    X, y = preparar_X_y(df)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X.head())
    print(y.head())

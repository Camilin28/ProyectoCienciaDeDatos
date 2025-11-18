# src/modelo.py
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

def entrenar_modelos(X, y, save_models: bool = True, random_state: int = 42):
    """
    Entrenamiento del modelo de regresión lineal y RandomForest"""
    # División temporal: entrenar hasta 2017, test 2018+
    if 'year' in X.columns:
        train_mask = X['year'] <= 2017
        X_train = X[train_mask].drop(columns=['year'])
        X_test = X[~train_mask].drop(columns=['year'])
        y_train = y[train_mask]
        y_test = y[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Escalar numéricos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo lineal
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    # RandomForest
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train) 
    y_pred_rf = rf.predict(X_test)

    # Métricas
    def metrics(y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False),
            'R2': r2_score(y_true, y_pred)
        }

    metrics_lr = metrics(y_test, y_pred_lr)
    metrics_rf = metrics(y_test, y_pred_rf)

    resultados = {
        'lr_model': lr,
        'rf_model': rf,
        'scaler': scaler,
        'X_train_columns': X_train.columns.tolist(),
        'metrics': {
            'lr': metrics_lr,
            'rf': metrics_rf
        },
        'y_test': y_test,
        'y_pred_lr': y_pred_lr,
        'y_pred_rf': y_pred_rf,
        'X_test': X_test,
    }

    if save_models:
        modelos_dir = ROOT / "modelos"
        modelos_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr, modelos_dir / "modelo_lineal.pkl")
        joblib.dump(rf, modelos_dir / "modelo_random_forest.pkl")
        joblib.dump(scaler, modelos_dir / "scaler.pkl")
        # guardamos columnas para reconstruir dataset en predicción
        joblib.dump(X_train.columns.tolist(), modelos_dir / "features_columns.pkl")
        print(f"Modelos guardados en {modelos_dir}")

    return resultados

if __name__ == "__main__":
    # Prueba rápida
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "results.csv")
    from proceso_db import limpieza_basica
    from features import crear_features_historial, preparar_X_y
    df = limpieza_basica(df)
    df = crear_features_historial(df)
    X, y = preparar_X_y(df)
    resultados = entrenar_modelos(X, y, save_models=False)
    print(resultados['metrics'])

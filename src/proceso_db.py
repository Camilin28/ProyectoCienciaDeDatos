import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def cargar_datos(path_csv: str = None):
    if path_csv is None:
        path_csv = ROOT /"data"/"results.csv"
    df = pd.read_csv(path_csv)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df

def limpieza_basica(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: duplicados, tipos correctos, columnas útiles."""
   
    df = df.drop_duplicates().copy()

    
    df['tournament'] = df['tournament'].astype(str).str.strip()
    df['city'] = df['city'].astype(str).str.strip()
    df['country'] = df['country'].astype(str).str.strip()

    
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)

    
    df['goal_diff'] = df['home_score'] - df['away_score']

    
    df['neutral'] = df['neutral'].astype(str).str.lower()
    df['is_neutral'] = df['neutral'].isin(['true','1','t','yes','y']).astype(int)
    
    if df['is_neutral'].sum() == 0 and df['neutral'].dtype == bool:
        df['is_neutral'] = df['neutral'].astype(int)

    
    df['year'] = df['date'].dt.year

    return df

def guardar_limpio(df: pd.DataFrame, filename: str = "results_limpio.csv"):
    out = ROOT /"data"/ filename
    df.to_csv(out, index=False)
    print(f"Guardado dataset limpio en: {out}")
    return out

if __name__ == "__main__":
    df = cargar_datos()
    df = limpieza_basica(df)
    guardar_limpio(df)

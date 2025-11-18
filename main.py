
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = 'database futbol/results.csv' # Ajusta ruta si es necesario


df = pd.read_csv(DATA_PATH)
# Convertir fecha
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Ordenar por fecha
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)


print('Dimensiones:', df.shape)
print(df.columns.tolist())


# Mostrar primeras filas
print(df.head())
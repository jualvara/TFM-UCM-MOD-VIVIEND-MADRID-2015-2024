# Auto instalación de pandas si no está (solo para debugging en entornos con problemas)
try:
    import pandas as pd
except ImportError:
    import os
    os.system("pip install pandas")
    import pandas as pd


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Cargar los datos
df = pd.read_excel("vivienda_imputada.xlsx")

# Guardar catálogo de distritos y barrios
catalogo = df[['DISTRITO_x', 'BARRIO']].drop_duplicates().sort_values(by=['DISTRITO_x', 'BARRIO'])
catalogo.to_csv("catalogo_distritos_barrios.csv", index=False)

# Variables
target = 'PRECIO_EUR_M2_x'
categorical_features = ['DISTRITO_x', 'BARRIO', 'TIPO_VIVIENDA']
X = df.drop(columns=[target])
y = df[target]

# Limpiar columnas redundantes
drop_cols = [col for col in X.columns if 'PRECIO_EUR_M2' in col or (col.endswith('_x') and col not in categorical_features)]
X = X.drop(columns=drop_cols)

# Asegurar tipo string
for col in categorical_features:
    X[col] = X[col].astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, "modelo_precio.pkl")
print("✅ Modelo guardado como modelo_precio.pkl")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Cargar los datos
df = pd.read_excel('vivienda_imputada.xlsx')

# 2. Definir target y features categóricas
target = 'PRECIO_EUR_M2_x'
categorical_features = ['DISTRITO_x', 'BARRIO', 'TIPO_VIVIENDA']

# X = todas las variables excepto target
X = df.drop(columns=[target])
y = df[target]

# 3. Eliminar columnas redundantes (precios ya calculados, duplicados _x/_y, etc.)
drop_cols = [
    col for col in X.columns
    if 'PRECIO_EUR_M2' in col or (col.endswith('_x') and col not in categorical_features)
]
X = X.drop(columns=drop_cols)

# 4. Asegurar que las categóricas sean string
for col in categorical_features:
    X[col] = X[col].astype(str)

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Pipeline: preprocesamiento + modelo
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Entrenar
model.fit(X_train, y_train)

# 8. Evaluación
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# 9. Guardar modelo (compatible con joblib 1.5.2 / sklearn 1.7.2)
joblib.dump(model, 'modelo_precio.pkl')
print("✅ Modelo guardado como modelo_precio.pkl")

import json

with open("columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)


import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump
import matplotlib.pyplot as plt

# === Configuración de paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === Cargar todos los Excel ===
files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
dfs = []
for f in files:
    try:
        df = pd.read_excel(f, engine='openpyxl')
        df['__source_file'] = os.path.basename(f)
        dfs.append(df)
    except Exception as e:
        print("❌ Error leyendo", f, e)

if not dfs:
    raise SystemExit("No se cargaron archivos .xlsx")

data = pd.concat(dfs, ignore_index=True)
print(f"✅ Archivos cargados: {len(dfs)} | Registros totales: {len(data)}")

# === Seleccionar columnas necesarias ===
cols = ["edad_persona", "xx", "yy", "fec_hora_hecho"]
data = data[cols].dropna()

# Convertir fecha/hora a componentes
data["fec_hora_hecho"] = pd.to_datetime(data["fec_hora_hecho"], errors="coerce")
data["hora"] = data["fec_hora_hecho"].dt.hour
data["mes"] = data["fec_hora_hecho"].dt.month
data = data.drop(columns=["fec_hora_hecho"])

# === Crear variable objetivo (riesgo) ===
data["lat_rounded"] = data["xx"].round(2)
data["lon_rounded"] = data["yy"].round(2)
freq = data.groupby(["lat_rounded", "lon_rounded"]).size().reset_index(name="count")
data = data.merge(freq, on=["lat_rounded", "lon_rounded"])
data["riesgo"] = (data["count"] > data["count"].median()).astype(int)

# === Balancear las clases ===
df_majority = data[data.riesgo == 0]
df_minority = data[data.riesgo == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
data_bal = pd.concat([df_majority, df_minority_upsampled])

# === Features y Target ===
X = data_bal[["edad_persona", "hora", "mes", "xx", "yy"]]
y = data_bal["riesgo"]

# === Escalado ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
print("📁 Escalador guardado en models/scaler.joblib")

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Modelo ===
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === Entrenamiento ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# === Evaluación ===
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("\n📊 Métricas del modelo:")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}")

# === Guardar métricas ===
metrics = pd.DataFrame([{
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "roc_auc": auc
}])
metrics.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)
print("📁 Métricas guardadas en models/model_metrics.csv")

# === Guardar modelo ===
model.save(os.path.join(MODEL_DIR, "model.h5"))
print("✅ Modelo guardado en models/model.h5")

# === Gráfico de pérdida ===
plt.figure()
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Evolución del error (loss)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

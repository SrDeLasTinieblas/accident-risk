import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# === CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === CARGA DE DATOS ===
print("🔄 Cargando archivos Excel...")
files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
dfs = []

for f in files:
    try:
        df = pd.read_excel(f, engine='openpyxl')
        df['__source_file'] = os.path.basename(f)
        dfs.append(df)
        print(f"  ✓ {os.path.basename(f)}: {len(df)} registros")
    except Exception as e:
        print(f"  ✗ Error en {os.path.basename(f)}: {e}")

if not dfs:
    raise SystemExit("❌ No se cargaron archivos .xlsx")

data = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Total: {len(dfs)} archivos | {len(data)} registros")

# === PREPARACIÓN DE DATOS ===
print("\n🔧 Preparando datos...")

# Seleccionar columnas relevantes
cols_necesarias = ["edad_persona", "xx", "yy", "fec_hora_hecho"]
data = data[cols_necesarias].dropna()
print(f"  • Registros después de limpiar: {len(data)}")

# Convertir fecha/hora
data["fec_hora_hecho"] = pd.to_datetime(data["fec_hora_hecho"], errors="coerce")
data = data.dropna(subset=["fec_hora_hecho"])

# Extraer componentes temporales
data["hora"] = data["fec_hora_hecho"].dt.hour
data["mes"] = data["fec_hora_hecho"].dt.month
data["dia_semana"] = data["fec_hora_hecho"].dt.dayofweek  # 0=Lunes, 6=Domingo
data["es_fin_semana"] = (data["dia_semana"] >= 5).astype(int)

# Categorizar horas
def categorizar_hora(hora):
    if 6 <= hora < 12:
        return 0  # Mañana
    elif 12 <= hora < 18:
        return 1  # Tarde
    elif 18 <= hora < 24:
        return 2  # Noche
    else:
        return 3  # Madrugada

data["periodo_dia"] = data["hora"].apply(categorizar_hora)

# === CREAR VARIABLE OBJETIVO (NIVEL DE RIESGO POR ZONA) ===
print("\n📊 Calculando nivel de riesgo por zona...")

# Redondear coordenadas para crear zonas (~1km²)
data["lat_zona"] = data["xx"].round(2)
data["lon_zona"] = data["yy"].round(2)

# Contar accidentes por zona
accidentes_zona = data.groupby(["lat_zona", "lon_zona"]).size().reset_index(name="total_accidentes")

# Clasificar en 3 niveles de riesgo
percentil_33 = accidentes_zona["total_accidentes"].quantile(0.33)
percentil_66 = accidentes_zona["total_accidentes"].quantile(0.66)

def clasificar_riesgo(total):
    if total <= percentil_33:
        return 0  # Riesgo BAJO
    elif total <= percentil_66:
        return 1  # Riesgo MEDIO
    else:
        return 2  # Riesgo ALTO

accidentes_zona["nivel_riesgo"] = accidentes_zona["total_accidentes"].apply(clasificar_riesgo)

# Unir con datos originales
data = data.merge(accidentes_zona[["lat_zona", "lon_zona", "nivel_riesgo"]], 
                  on=["lat_zona", "lon_zona"])

print(f"  • Distribución de riesgo:")
print(data["nivel_riesgo"].value_counts().sort_index())

# === BALANCEAR CLASES ===
print("\n⚖️ Balanceando clases...")
df_por_clase = [data[data.nivel_riesgo == i] for i in range(3)]
max_samples = max(len(df) for df in df_por_clase)

df_balanceados = []
for i, df in enumerate(df_por_clase):
    df_upsampled = resample(df, replace=True, n_samples=max_samples, random_state=42)
    df_balanceados.append(df_upsampled)
    print(f"  • Clase {i}: {len(df)} → {len(df_upsampled)}")

data_bal = pd.concat(df_balanceados, ignore_index=True)

# === FEATURES Y TARGET ===
features = ["edad_persona", "hora", "mes", "dia_semana", "es_fin_semana", 
            "periodo_dia", "xx", "yy"]
X = data_bal[features]
y = data_bal["nivel_riesgo"]

# === ESCALADO ===
print("\n🔢 Escalando features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
dump(scaler, scaler_path)
print(f"  ✓ Escalador guardado: {scaler_path}")

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📦 Datos de entrenamiento: {len(X_train)}")
print(f"📦 Datos de prueba: {len(X_test)}")

# === MODELO MEJORADO ===
print("\n🤖 Construyendo modelo...")

model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.1),
    
    Dense(3, activation='softmax')  # 3 clases de riesgo
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# === ENTRENAMIENTO ===
print("\n🏋️ Entrenando modelo...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === EVALUACIÓN ===
print("\n📊 Evaluando modelo...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc:.3f}")
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, 
                          target_names=['Bajo', 'Medio', 'Alto']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\n🔢 Matriz de Confusión:")
print(cm)

# === GUARDAR MÉTRICAS ===
metrics = pd.DataFrame([{
    "accuracy": acc,
    "fecha_entrenamiento": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_features": X_train.shape[1],
    "num_samples_train": len(X_train),
    "num_samples_test": len(X_test)
}])

metrics_path = os.path.join(MODEL_DIR, "model_metrics.csv")
metrics.to_csv(metrics_path, index=False)
print(f"\n💾 Métricas guardadas: {metrics_path}")

# === GUARDAR MODELO ===
model_path = os.path.join(MODEL_DIR, "model_riesgo_accidentes.h5")
model.save(model_path)
print(f"✅ Modelo guardado: {model_path}")

# === VISUALIZACIONES ===
print("\n📈 Generando visualizaciones...")
fecha_hora = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Gráfico 1: Pérdida
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
plt.plot(history.history['val_loss'], label='Validación', linewidth=2)
plt.title('Evolución de la Pérdida', fontsize=14, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
plt.title('Evolución de la Precisión', fontsize=14, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
loss_plot = os.path.join(MODEL_DIR, f"training_history_{fecha_hora}.png")
plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ {loss_plot}")
plt.close()

# Gráfico 2: Matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bajo', 'Medio', 'Alto'],
            yticklabels=['Bajo', 'Medio', 'Alto'])
plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')

cm_plot = os.path.join(MODEL_DIR, f"confusion_matrix_{fecha_hora}.png")
plt.savefig(cm_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ {cm_plot}")
plt.close()

# === FUNCIÓN DE PREDICCIÓN DE EJEMPLO ===
def predecir_riesgo(edad, hora, mes, dia_semana, latitud, longitud):
    """
    Predice el nivel de riesgo de un accidente.
    
    Retorna:
        - nivel: 0 (Bajo), 1 (Medio), 2 (Alto)
        - probabilidades: array con probabilidades para cada clase
    """
    es_fin_semana = 1 if dia_semana >= 5 else 0
    periodo = categorizar_hora(hora)
    
    X_nuevo = np.array([[edad, hora, mes, dia_semana, es_fin_semana, 
                        periodo, latitud, longitud]])
    X_nuevo_scaled = scaler.transform(X_nuevo)
    
    prob = model.predict(X_nuevo_scaled, verbose=0)[0]
    nivel = np.argmax(prob)
    
    return nivel, prob

# Ejemplo de uso
print("\n🧪 Ejemplo de predicción:")
nivel, probs = predecir_riesgo(
    edad=35, 
    hora=18,  # 6 PM
    mes=4,    # Abril
    dia_semana=4,  # Viernes
    latitud=-16.4,
    longitud=-71.5
)

nombres_riesgo = ['BAJO', 'MEDIO', 'ALTO']
print(f"  Nivel de riesgo: {nombres_riesgo[nivel]}")
print(f"  Probabilidades: Bajo={probs[0]:.2%}, Medio={probs[1]:.2%}, Alto={probs[2]:.2%}")

print("\n✨ ¡Entrenamiento completado exitosamente!")
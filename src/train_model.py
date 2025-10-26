import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.cluster import KMeans
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
data["dia_semana"] = data["fec_hora_hecho"].dt.dayofweek
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

# === FEATURES GEOESPACIALES MEJORADAS ===
print("\n🗺️ Creando features geoespaciales...")

# 1. Clustering de zonas (identifica "hotspots")
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
coords = data[["xx", "yy"]].values
data["cluster_zona"] = kmeans.fit_predict(coords)

# Guardar el modelo de clustering
kmeans_path = os.path.join(MODEL_DIR, "kmeans_zonas.joblib")
dump(kmeans, kmeans_path)
print(f"  ✓ Modelo KMeans guardado: {kmeans_path}")

# 2. Distancia al centro de cada cluster
centroids = kmeans.cluster_centers_
for i in range(n_clusters):
    data[f"dist_cluster_{i}"] = np.sqrt(
        (data["xx"] - centroids[i][0])**2 + 
        (data["yy"] - centroids[i][1])**2
    )

# Usar solo las 5 distancias más cercanas para cada punto
dist_cols = [f"dist_cluster_{i}" for i in range(n_clusters)]
data["dist_min_1"] = data[dist_cols].min(axis=1)
data["dist_min_2"] = data[dist_cols].apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
data["dist_min_3"] = data[dist_cols].apply(lambda x: x.nsmallest(3).iloc[-1], axis=1)

# 3. Densidad local (accidentes en radio de ~1km)
data["lat_zona"] = data["xx"].round(2)
data["lon_zona"] = data["yy"].round(2)
densidad_local = data.groupby(["lat_zona", "lon_zona"]).size().reset_index(name="densidad_local")
data = data.merge(densidad_local, on=["lat_zona", "lon_zona"])

# 4. Estadísticas temporales por cluster
for cluster in range(n_clusters):
    cluster_data = data[data["cluster_zona"] == cluster]
    accidentes_cluster = len(cluster_data)
    data.loc[data["cluster_zona"] == cluster, "accidentes_cluster"] = accidentes_cluster

# === CREAR VARIABLE OBJETIVO ===
print("\n📊 Calculando nivel de riesgo...")

# Usar densidad local como base para el riesgo
percentil_33 = data["densidad_local"].quantile(0.33)
percentil_66 = data["densidad_local"].quantile(0.66)

def clasificar_riesgo(densidad):
    if densidad <= percentil_33:
        return 0  # Riesgo BAJO
    elif densidad <= percentil_66:
        return 1  # Riesgo MEDIO
    else:
        return 2  # Riesgo ALTO

data["nivel_riesgo"] = data["densidad_local"].apply(clasificar_riesgo)

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
# NO incluir coordenadas directas, solo features derivadas
features = [
    "edad_persona", "hora", "mes", "dia_semana", "es_fin_semana", "periodo_dia",
    "cluster_zona", "dist_min_1", "dist_min_2", "dist_min_3", 
    "densidad_local", "accidentes_cluster"
]

X = data_bal[features]
y = data_bal["nivel_riesgo"]

# Guardar info de features para predicción
feature_info = {
    "feature_names": features,
    "n_clusters": n_clusters,
    "percentil_33": percentil_33,
    "percentil_66": percentil_66,
    "lat_min": data["xx"].min(),
    "lat_max": data["xx"].max(),
    "lon_min": data["yy"].min(),
    "lon_max": data["yy"].max()
}
dump(feature_info, os.path.join(MODEL_DIR, "feature_info.joblib"))

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
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
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

# Gráfico 1: Pérdida y Accuracy
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

# Gráfico 3: Mapa de clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['yy'], data['xx'], c=data['cluster_zona'], 
                     cmap='tab20', alpha=0.6, s=1)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.title('Zonas de Accidentes (Clusters)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

cluster_plot = os.path.join(MODEL_DIR, f"clusters_map_{fecha_hora}.png")
plt.savefig(cluster_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ {cluster_plot}")
plt.close()

# === FUNCIÓN DE PREDICCIÓN MEJORADA ===
print("\n🧪 Creando función de predicción...")

def calcular_features_prediccion(edad, hora, mes, dia_semana, latitud, longitud):
    """
    Calcula todas las features necesarias para una predicción.
    Maneja coordenadas fuera del rango de entrenamiento.
    """
    from joblib import load
    
    # Cargar modelos guardados
    kmeans = load(os.path.join(MODEL_DIR, "kmeans_zonas.joblib"))
    feature_info = load(os.path.join(MODEL_DIR, "feature_info.joblib"))
    
    # Features temporales
    es_fin_semana = 1 if dia_semana >= 5 else 0
    periodo = categorizar_hora(hora)
    
    # Verificar si está dentro del rango conocido
    fuera_rango = (
        latitud < feature_info["lat_min"] or latitud > feature_info["lat_max"] or
        longitud < feature_info["lon_min"] or longitud > feature_info["lon_max"]
    )
    
    if fuera_rango:
        print(f"⚠️ ADVERTENCIA: Coordenadas fuera del área de entrenamiento")
        print(f"   Rango válido: Lat [{feature_info['lat_min']:.2f}, {feature_info['lat_max']:.2f}]")
        print(f"                  Lon [{feature_info['lon_min']:.2f}, {feature_info['lon_max']:.2f}]")
        print(f"   Predicción puede ser imprecisa.\n")
    
    # Asignar al cluster más cercano
    coords_nueva = np.array([[latitud, longitud]])
    cluster_zona = kmeans.predict(coords_nueva)[0]
    
    # Calcular distancias a clusters
    centroids = kmeans.cluster_centers_
    distancias = []
    for i in range(feature_info["n_clusters"]):
        dist = np.sqrt(
            (latitud - centroids[i][0])**2 + 
            (longitud - centroids[i][1])**2
        )
        distancias.append(dist)
    
    distancias_sorted = sorted(distancias)
    dist_min_1 = distancias_sorted[0]
    dist_min_2 = distancias_sorted[1]
    dist_min_3 = distancias_sorted[2]
    
    # Densidad local estimada (basada en distancia al cluster)
    # Asumimos que más cerca del centro = mayor densidad
    densidad_estimada = max(1, int(10 / (1 + dist_min_1)))
    
    # Accidentes del cluster (aproximado)
    accidentes_cluster = 50  # Valor promedio
    
    return {
        "edad_persona": edad,
        "hora": hora,
        "mes": mes,
        "dia_semana": dia_semana,
        "es_fin_semana": es_fin_semana,
        "periodo_dia": periodo,
        "cluster_zona": cluster_zona,
        "dist_min_1": dist_min_1,
        "dist_min_2": dist_min_2,
        "dist_min_3": dist_min_3,
        "densidad_local": densidad_estimada,
        "accidentes_cluster": accidentes_cluster
    }

def predecir_riesgo(edad, hora, mes, dia_semana, latitud, longitud):
    """
    Predice el nivel de riesgo de un accidente.
    
    Retorna:
        - nivel: 0 (Bajo), 1 (Medio), 2 (Alto)
        - probabilidades: dict con probabilidades
        - advertencia: bool si está fuera de rango
    """
    from joblib import load
    
    # Calcular features
    features_dict = calcular_features_prediccion(
        edad, hora, mes, dia_semana, latitud, longitud
    )
    
    # Verificar rango
    feature_info = load(os.path.join(MODEL_DIR, "feature_info.joblib"))
    fuera_rango = (
        latitud < feature_info["lat_min"] or latitud > feature_info["lat_max"] or
        longitud < feature_info["lon_min"] or longitud > feature_info["lon_max"]
    )
    
    # Crear array en el orden correcto
    feature_names = feature_info["feature_names"]
    X_nuevo = np.array([[features_dict[name] for name in feature_names]])
    
    # Escalar y predecir
    scaler = load(os.path.join(MODEL_DIR, "scaler.joblib"))
    X_nuevo_scaled = scaler.transform(X_nuevo)
    
    prob = model.predict(X_nuevo_scaled, verbose=0)[0]
    nivel = np.argmax(prob)
    
    return {
        "nivel": nivel,
        "probabilidades": {
            "bajo": float(prob[0]),
            "medio": float(prob[1]),
            "alto": float(prob[2])
        },
        "fuera_rango": fuera_rango,
        "cluster_asignado": int(features_dict["cluster_zona"])
    }

# Ejemplos de predicción
print("\n🧪 Ejemplos de predicción:\n")

# Ejemplo 1: Dentro del rango
print("1️⃣ Ubicación dentro del área de estudio:")
resultado1 = predecir_riesgo(
    edad=35, 
    hora=18,
    mes=4,
    dia_semana=4,
    latitud=data["xx"].median(),
    longitud=data["yy"].median()
)
nombres_riesgo = ['BAJO', 'MEDIO', 'ALTO']
print(f"   Nivel de riesgo: {nombres_riesgo[resultado1['nivel']]}")
print(f"   Probabilidades: Bajo={resultado1['probabilidades']['bajo']:.2%}, "
      f"Medio={resultado1['probabilidades']['medio']:.2%}, "
      f"Alto={resultado1['probabilidades']['alto']:.2%}")
print(f"   Cluster: {resultado1['cluster_asignado']}\n")

# Ejemplo 2: Fuera del rango
print("2️⃣ Ubicación fuera del área de estudio:")
resultado2 = predecir_riesgo(
    edad=35,
    hora=18,
    mes=4,
    dia_semana=4,
    latitud=-11.98,  # Lima (lejos de Arequipa)
    longitud=-77.09
)
print(f"   Nivel de riesgo: {nombres_riesgo[resultado2['nivel']]}")
print(f"   Probabilidades: Bajo={resultado2['probabilidades']['bajo']:.2%}, "
      f"Medio={resultado2['probabilidades']['medio']:.2%}, "
      f"Alto={resultado2['probabilidades']['alto']:.2%}")

print("\n✨ ¡Entrenamiento completado exitosamente!")
print("\n📝 IMPORTANTE:")
print("   • El modelo fue entrenado solo con datos de accidentes")
print("   • Las predicciones fuera del área de estudio son aproximaciones")
print("   • Para mejor precisión, entrena con datos de toda la región")
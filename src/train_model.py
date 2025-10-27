import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚗 MODELO PREDICTIVO DE ACCIDENTES - Ruta Arequipa-Camaná")
print("="*70)

# === CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === FASE 1: CARGA DE DATOS ===
print("\n📂 FASE 1: Cargando archivos Excel...")
files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
files.sort()

dfs = []
for f in files:
    try:
        df = pd.read_excel(f, engine='openpyxl')
        df['__source_file'] = os.path.basename(f)
        dfs.append(df)
        print(f"   ✓ {os.path.basename(f)}: {len(df):,} registros")
    except Exception as e:
        print(f"   ✗ Error en {os.path.basename(f)}: {e}")

if not dfs:
    raise SystemExit("❌ No se cargaron archivos Excel")

data = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Total cargado: {len(dfs)} archivos | {len(data):,} registros")

# === FASE 2: PREPROCESAMIENTO ===
print("\n🔧 FASE 2: Preprocesamiento de datos...")

print("   → Limpiando datos básicos...")
# Columnas necesarias
cols_necesarias = ["edad_persona", "xx", "yy", "fec_hora_hecho"]

# Verificar que existan las columnas
for col in cols_necesarias:
    if col not in data.columns:
        raise ValueError(f"❌ Columna '{col}' no encontrada en los datos")

# Seleccionar solo columnas necesarias
data = data[cols_necesarias].copy()

# Eliminar filas con coordenadas nulas (crítico)
data = data.dropna(subset=["xx", "yy", "fec_hora_hecho"])

print(f"      Registros después de limpieza: {len(data):,}")

# === PROCESAMIENTO DE FECHAS ===
print("   → Procesando fechas y horas...")
data["fec_hora_hecho"] = pd.to_datetime(data["fec_hora_hecho"], errors="coerce")
data = data.dropna(subset=["fec_hora_hecho"])

# Extraer componentes temporales
data["hora"] = data["fec_hora_hecho"].dt.hour
data["mes"] = data["fec_hora_hecho"].dt.month
data["dia_semana"] = data["fec_hora_hecho"].dt.dayofweek
data["es_fin_semana"] = (data["dia_semana"] >= 5).astype(int)

# === PROCESAMIENTO DE EDAD ===
print("   → Procesando edad...")

# Llenar edades nulas con la mediana
edad_mediana = data["edad_persona"].median()
data["edad_persona"] = data["edad_persona"].fillna(edad_mediana)

# Categorizar horas del día
def categorizar_hora(hora):
    if pd.isna(hora):
        return 1  # Default: tarde
    if 6 <= hora < 12:
        return 0  # Mañana
    elif 12 <= hora < 18:
        return 1  # Tarde
    elif 18 <= hora < 24:
        return 2  # Noche
    else:
        return 3  # Madrugada

data["periodo_dia"] = data["hora"].apply(categorizar_hora)

# Categorizar edad
def categorizar_edad(edad):
    if pd.isna(edad):
        return 2  # Default: adulto
    if edad < 18:
        return 0  # Menor
    elif edad < 30:
        return 1  # Joven
    elif edad < 60:
        return 2  # Adulto
    else:
        return 3  # Mayor

data["grupo_edad"] = data["edad_persona"].apply(categorizar_edad).astype(int)

print(f"      Registros finales: {len(data):,}")

# === FASE 3: CREAR GRID DE RIESGO ===
print("\n📊 FASE 3: Calculando mapa de riesgo...")

# Crear grid (0.01 grados ≈ 1.1 km)
GRID_SIZE = 0.01
data["lat_grid"] = (data["xx"] / GRID_SIZE).round() * GRID_SIZE
data["lon_grid"] = (data["yy"] / GRID_SIZE).round() * GRID_SIZE

# Contar accidentes por celda
grid_counts = data.groupby(["lat_grid", "lon_grid"]).size().reset_index(name="num_accidentes")

# Estadísticas adicionales por celda
grid_stats = data.groupby(["lat_grid", "lon_grid"]).agg({
    "hora": "mean",
    "mes": lambda x: x.mode().iloc[0] if len(x) > 0 else x.iloc[0],
    "edad_persona": "mean",
    "es_fin_semana": "mean"
}).reset_index()

grid_data = grid_counts.merge(grid_stats, on=["lat_grid", "lon_grid"])

# Clasificar nivel de riesgo
percentiles = grid_data["num_accidentes"].quantile([0.33, 0.66])

def clasificar_riesgo(num):
    if num <= percentiles[0.33]:
        return 0  # Bajo
    elif num <= percentiles[0.66]:
        return 1  # Medio
    else:
        return 2  # Alto

grid_data["nivel_riesgo"] = grid_data["num_accidentes"].apply(clasificar_riesgo)

print(f"   ✓ Celdas del grid: {len(grid_data):,}")
print(f"   ✓ Distribución de riesgo:")
for nivel, nombre in enumerate(["Bajo", "Medio", "Alto"]):
    count = (grid_data["nivel_riesgo"] == nivel).sum()
    pct = count / len(grid_data) * 100
    print(f"      • {nombre}: {count:,} celdas ({pct:.1f}%)")

# Guardar grid
grid_path = os.path.join(MODEL_DIR, "risk_grid.csv")
grid_data.to_csv(grid_path, index=False)
print(f"   💾 Grid guardado: {grid_path}")

# === FASE 4: CALCULAR DENSIDAD LOCAL ===
print("\n🔍 FASE 4: Calculando densidad local...")

# Unir datos con grid
data = data.merge(
    grid_data[["lat_grid", "lon_grid", "nivel_riesgo", "num_accidentes"]],
    on=["lat_grid", "lon_grid"]
)

# Calcular densidad en radio (KD-Tree)
coords = data[["xx", "yy"]].values
tree = cKDTree(coords)

print("   → Calculando vecinos cercanos (esto puede tardar)...")
RADIO_KM = 2  # 2 km
RADIO_GRADOS = RADIO_KM / 111  # Conversión aproximada

densidades = []
batch_size = 1000

for i in range(0, len(coords), batch_size):
    batch = coords[i:i+batch_size]
    for coord in batch:
        indices = tree.query_ball_point(coord, r=RADIO_GRADOS)
        densidades.append(len(indices))
    
    if (i + batch_size) % 5000 == 0:
        print(f"      Procesados: {min(i+batch_size, len(coords)):,} / {len(coords):,}")

data["densidad_local"] = densidades
print(f"   ✓ Densidad calculada para {len(data):,} registros")

# === FASE 5: BALANCEO DE CLASES ===
print("\n⚖️ FASE 5: Balanceando clases...")

from sklearn.utils import resample

df_por_clase = [data[data.nivel_riesgo == i] for i in range(3)]

# Encontrar el tamaño objetivo (promedio de las clases)
sizes = [len(df) for df in df_por_clase]
target_size = int(np.mean(sizes))

print(f"   → Tamaño objetivo por clase: {target_size:,}")

df_balanceados = []
for i, df in enumerate(df_por_clase):
    if len(df) < target_size:
        # Oversample
        df_resampled = resample(df, replace=True, n_samples=target_size, random_state=42)
        print(f"      • Clase {i} (↑): {len(df):,} → {target_size:,}")
    elif len(df) > target_size:
        # Undersample
        df_resampled = resample(df, replace=False, n_samples=target_size, random_state=42)
        print(f"      • Clase {i} (↓): {len(df):,} → {target_size:,}")
    else:
        df_resampled = df
        print(f"      • Clase {i} (=): {len(df):,}")
    
    df_balanceados.append(df_resampled)

data_bal = pd.concat(df_balanceados, ignore_index=True)
print(f"   ✓ Dataset balanceado: {len(data_bal):,} registros")

# === FASE 6: PREPARAR FEATURES ===
print("\n📦 FASE 6: Preparando features...")

features = [
    "edad_persona", "grupo_edad",
    "hora", "mes", "dia_semana", "es_fin_semana", "periodo_dia",
    "xx", "yy",
    "densidad_local", "num_accidentes"
]

X = data_bal[features]
y = data_bal["nivel_riesgo"]

print(f"   ✓ Features: {len(features)}")
print(f"   ✓ Muestras: {len(X):,}")
print(f"   ✓ Distribución del target:")
for nivel, nombre in enumerate(["Bajo", "Medio", "Alto"]):
    count = (y == nivel).sum()
    pct = count / len(y) * 100
    print(f"      • {nombre}: {count:,} ({pct:.1f}%)")

# === FASE 7: ESCALADO ===
print("\n🔢 FASE 7: Escalando features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
dump(scaler, scaler_path)
print(f"   💾 Scaler guardado: {scaler_path}")

# === FASE 8: SPLIT ===
print("\n✂️ FASE 8: Dividiendo datos...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"   ✓ Entrenamiento: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   ✓ Prueba: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# === FASE 9: ENTRENAMIENTO ===
print("\n🌲 FASE 9: Entrenando Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("   → Iniciando entrenamiento...")
model.fit(X_train, y_train)
print("   ✓ Entrenamiento completado")

# === FASE 10: EVALUACIÓN ===
print("\n📊 FASE 10: Evaluando modelo...")

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n   🎯 Accuracy: {acc:.3f} ({acc*100:.1f}%)")

print("\n   📋 Reporte de Clasificación:")
print(classification_report(
    y_test, y_pred, 
    target_names=['Bajo', 'Medio', 'Alto'],
    digits=3
))

cm = confusion_matrix(y_test, y_pred)
print("   🔢 Matriz de Confusión:")
print("        Pred: Bajo  Medio  Alto")
for i, nombre in enumerate(['Bajo', 'Medio', 'Alto']):
    print(f"   Real {nombre:5s}: {cm[i][0]:5d}  {cm[i][1]:5d}  {cm[i][2]:5d}")

# Importancia de features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   📈 Top 5 Features más importantes:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"      • {row['feature']:20s}: {row['importance']:.4f}")

# === FASE 11: GUARDAR MODELO ===
print("\n💾 FASE 11: Guardando modelo y metadatos...")

model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
dump(model, model_path)
print(f"   ✓ Modelo: {model_path}")

# Guardar metadata
metadata = {
    "feature_names": features,
    "lat_min": float(data["xx"].min()),
    "lat_max": float(data["xx"].max()),
    "lon_min": float(data["yy"].min()),
    "lon_max": float(data["yy"].max()),
    "grid_size": GRID_SIZE,
    "percentil_33": float(percentiles[0.33]),
    "percentil_66": float(percentiles[0.66]),
    "model_type": "RandomForest",
    "accuracy": float(acc),
    "n_estimators": 200,
    "fecha_entrenamiento": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

metadata_path = os.path.join(MODEL_DIR, "model_metadata.joblib")
dump(metadata, metadata_path)
print(f"   ✓ Metadata: {metadata_path}")

# Guardar métricas
metrics = pd.DataFrame([{
    "accuracy": acc,
    "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_features": len(features),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "model": "RandomForest"
}])

metrics_path = os.path.join(MODEL_DIR, "model_metrics.csv")
metrics.to_csv(metrics_path, index=False)
print(f"   ✓ Métricas: {metrics_path}")

# === FASE 12: VISUALIZACIONES ===
print("\n📈 FASE 12: Generando visualizaciones...")

fecha_hora = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='RdYlGn_r',
    xticklabels=['Bajo', 'Medio', 'Alto'],
    yticklabels=['Bajo', 'Medio', 'Alto'],
    cbar_kws={'label': 'Cantidad'}
)
plt.title('Matriz de Confusión - Random Forest', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Predicción', fontsize=12)
plt.tight_layout()

cm_plot = os.path.join(MODEL_DIR, f"confusion_matrix_{fecha_hora}.png")
plt.savefig(cm_plot, dpi=300, bbox_inches='tight')
print(f"   ✓ {os.path.basename(cm_plot)}")
plt.close()

# 2. Importancia de features
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importancia', fontsize=12)
plt.title('Importancia de Features', fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.tight_layout()

importance_plot = os.path.join(MODEL_DIR, f"feature_importance_{fecha_hora}.png")
plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
print(f"   ✓ {os.path.basename(importance_plot)}")
plt.close()

# 3. Mapa de riesgo
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    grid_data['lon_grid'], 
    grid_data['lat_grid'],
    c=grid_data['nivel_riesgo'],
    s=grid_data['num_accidentes'] * 5,
    cmap='RdYlGn_r',
    alpha=0.6,
    edgecolors='black',
    linewidth=0.5
)

cbar = plt.colorbar(scatter, label='Nivel de Riesgo', ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Bajo', 'Medio', 'Alto'])

plt.xlabel('Longitud', fontsize=12)
plt.ylabel('Latitud', fontsize=12)
plt.title('Mapa de Riesgo de Accidentes\nArequipa-Camaná', 
          fontsize=16, fontweight='bold', pad=20)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

map_plot = os.path.join(MODEL_DIR, f"risk_map_{fecha_hora}.png")
plt.savefig(map_plot, dpi=300, bbox_inches='tight')
print(f"   ✓ {os.path.basename(map_plot)}")
plt.close()

# === RESUMEN FINAL ===
print("\n" + "="*70)
print("✨ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print("="*70)
print(f"\n📊 RESUMEN:")
print(f"   • Accuracy: {acc:.1%}")
print(f"   • Zonas mapeadas: {len(grid_data):,}")
print(f"   • Área cubierta:")
print(f"      Latitud:  [{metadata['lat_min']:.4f}, {metadata['lat_max']:.4f}]")
print(f"      Longitud: [{metadata['lon_min']:.4f}, {metadata['lon_max']:.4f}]")
print(f"\n📁 Archivos generados:")
print(f"   • {os.path.basename(model_path)}")
print(f"   • {os.path.basename(scaler_path)}")
print(f"   • {os.path.basename(grid_path)}")
print(f"   • {os.path.basename(metadata_path)}")
print(f"   • 3 visualizaciones PNG")
print("\n🚀 El modelo está listo para usar en la API")
print("="*70 + "\n")
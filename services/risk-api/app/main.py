import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
from datetime import datetime
from scipy.spatial import cKDTree

# === PATHS ===
# Layout: services/risk-api/{app,artifacts,training}  y  <repo>/data/raw
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # services/risk-api
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_ROOT, "artifacts"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "data", "raw")))
print(f"📂 Directorio actual: {CURRENT_DIR}")
print(f"📂 Raíz del proyecto: {PROJECT_ROOT}")
print(f"📂 Directorio de modelos: {MODEL_DIR}")

# === PATHS DE ARCHIVOS ===
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.joblib")
GRID_PATH = os.path.join(MODEL_DIR, "risk_grid.csv")

# === VALIDAR ARCHIVOS ===
required_files = {
    "modelo Random Forest": MODEL_PATH,
    "scaler": SCALER_PATH,
    "metadata": METADATA_PATH,
    "grid de riesgo": GRID_PATH
}

print("\n🔍 Verificando archivos necesarios...")
for name, path in required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró {name}: {path}")
    print(f"   ✓ {name}")

# === CARGAR MODELOS ===
print("\n🔄 Cargando modelos...")
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
metadata = load(METADATA_PATH)
grid_data = pd.read_csv(GRID_PATH)

# Crear árbol espacial para búsqueda rápida
grid_coords = grid_data[["lat_grid", "lon_grid"]].values
grid_tree = cKDTree(grid_coords)

print("✅ Modelos cargados exitosamente")
print(f"   • Tipo: {metadata['model_type']}")
print(f"   • Accuracy: {metadata['accuracy']:.1%}")
print(f"   • Zonas mapeadas: {len(grid_data):,}")
print(f"   • Features esperadas: {len(metadata['feature_names'])}")
print(f"   • Features: {metadata['feature_names']}")

app = Flask(__name__)

# === CARGAR DATOS HISTÓRICOS ===
print("\n📊 Calculando estadísticas históricas globales...")

# Cargar TODOS los datos originales para calcular promedios
import glob

all_data = []
for file in glob.glob(os.path.join(DATA_DIR, "*.xlsx")):
    if not file.endswith("~$"):  # Ignorar archivos temporales de Excel
        try:
            df = pd.read_excel(file, engine='openpyxl')
            all_data.append(df)
        except:
            pass

if all_data:
    datos_completos = pd.concat(all_data, ignore_index=True)
    datos_completos["fec_hora_hecho"] = pd.to_datetime(datos_completos["fec_hora_hecho"], errors="coerce")
    datos_completos["hora"] = datos_completos["fec_hora_hecho"].dt.hour
    datos_completos["mes"] = datos_completos["fec_hora_hecho"].dt.month
    datos_completos["dia_semana"] = datos_completos["fec_hora_hecho"].dt.dayofweek
    
    # Calcular promedios históricos globales
    HIST_HORA = datos_completos.groupby("hora").size().to_dict()
    HIST_MES = datos_completos.groupby("mes").size().to_dict()
    HIST_DIA = datos_completos.groupby("dia_semana").size().to_dict()
    
    print(f"   ✓ Promedios históricos calculados: {len(HIST_HORA)} horas, {len(HIST_MES)} meses, {len(HIST_DIA)} días")
else:
    # Valores por defecto si no se pueden cargar los datos
    HIST_HORA = {h: 500 for h in range(24)}
    HIST_MES = {m: 1000 for m in range(1, 13)}
    HIST_DIA = {d: 800 for d in range(7)}
    print("   ⚠️  Usando valores históricos por defecto")

# === FUNCIONES AUXILIARES ===

def categorizar_hora(hora):
    """Categoriza la hora del día"""
    if 6 <= hora < 12:
        return 0  # Mañana
    elif 12 <= hora < 18:
        return 1  # Tarde
    elif 18 <= hora < 24:
        return 2  # Noche
    else:
        return 3  # Madrugada

def categorizar_edad(edad):
    """Categoriza la edad en grupos"""
    if edad < 18:
        return 0  # Menor
    elif edad < 30:
        return 1  # Joven
    elif edad < 60:
        return 2  # Adulto
    else:
        return 3  # Mayor

def calcular_densidad_local(lat, lon, radio=0.045):
    """
    Calcula densidad de accidentes en un radio (~5km).
    Busca celdas del grid cercanas.
    """
    punto = np.array([lat, lon])
    indices = grid_tree.query_ball_point(punto, r=radio)
    
    if not indices or len(indices) == 0:
        return 1  # Densidad mínima
    
    # Sumar accidentes de celdas cercanas
    densidad = grid_data.iloc[indices]["num_accidentes"].sum()
    return int(densidad)

def obtener_info_grid(lat, lon):
    """
    Obtiene información histórica de la celda del grid más cercana.
    """
    lat_grid = round(lat / metadata["grid_size"]) * metadata["grid_size"]
    lon_grid = round(lon / metadata["grid_size"]) * metadata["grid_size"]
    
    # Buscar celda exacta
    celda = grid_data[
        (grid_data["lat_grid"] == lat_grid) & 
        (grid_data["lon_grid"] == lon_grid)
    ]
    
    if len(celda) > 0:
        return {
            "nivel_riesgo": int(celda.iloc[0]["nivel_riesgo"]),
            "num_accidentes": int(celda.iloc[0]["num_accidentes"]),
            "metodo": "grid_exacto",
            "confianza": "alta"
        }
    
    # Si no hay celda exacta, buscar la más cercana
    distancias, indices = grid_tree.query([lat, lon], k=1)
    
    if isinstance(indices, (int, np.integer)):
        idx = indices
        dist = distancias
    else:
        idx = indices[0] if len(indices) > 0 else 0
        dist = distancias[0] if len(distancias) > 0 else 0
    
    celda_cercana = grid_data.iloc[idx]
    distancia_km = float(dist) * 111
    
    return {
        "nivel_riesgo": int(celda_cercana["nivel_riesgo"]),
        "num_accidentes": int(celda_cercana["num_accidentes"]),
        "metodo": "grid_aproximado",
        "distancia_km": round(distancia_km, 2),
        "confianza": "media" if distancia_km < 5 else "baja"
    }

def calcular_radio_zona(num_accidentes, nivel_riesgo):
    """Calcula el radio de la zona de riesgo"""
    radio_base = {0: 50, 1: 100, 2: 150}
    
    if num_accidentes >= 10:
        multiplicador = 1.5
    elif num_accidentes >= 5:
        multiplicador = 1.2
    else:
        multiplicador = 1.0
    
    return int(radio_base.get(nivel_riesgo, 100) * multiplicador)

def calcular_risk_score(nivel_riesgo, num_accidentes):
    """Calcula un score de riesgo normalizado entre 0 y 1"""
    score_base = {0: 0.3, 1: 0.6, 2: 0.85}
    base = score_base.get(nivel_riesgo, 0.5)
    ajuste = min(num_accidentes / 100, 0.15)
    return min(base + ajuste, 1.0)

def predecir_riesgo_con_seguridad(features_scaled, prob_grid=None):
    """
    Predice con umbral de seguridad ajustado.
    Prioriza detectar riesgo ALTO (evita falsos negativos).
    """
    # Obtener probabilidades del modelo
    probas = model.predict_proba(features_scaled)[0]
    
    # Si hay datos históricos del grid, combinar
    if prob_grid is not None:
        probas = probas * 0.6 + prob_grid * 0.4
        probas = probas / probas.sum()
        metodo = "hibrido"
    else:
        metodo = "modelo_ml"
    
    # ESTRATEGIA DE SEGURIDAD: Umbral bajo para riesgo alto
    if probas[2] > 0.20:  # 20% de probabilidad de riesgo alto
        return 2, probas, metodo
    elif probas[1] > 0.30 and probas[0] < 0.50:
        return 1, probas, metodo
    else:
        return np.argmax(probas), probas, metodo

def predecir_riesgo(edad, hora, mes, dia_semana, lat, lon):
    """
    Realiza predicción usando estrategia híbrida.
    """
    # Verificar rango
    fuera_rango = (
        lat < metadata["lat_min"] or lat > metadata["lat_max"] or
        lon < metadata["lon_min"] or lon > metadata["lon_max"]
    )
    
    # Features temporales
    es_fin_semana = 1 if dia_semana >= 5 else 0
    periodo = categorizar_hora(hora)
    grupo_edad = categorizar_edad(edad)
    
    # Información del grid
    info_grid = obtener_info_grid(lat, lon)
    densidad_local = calcular_densidad_local(lat, lon)
    
    # Obtener promedios históricos
    acc_hora_hist = HIST_HORA.get(hora, 500)
    acc_mes_hist = HIST_MES.get(mes, 1000)
    acc_dia_hist = HIST_DIA.get(dia_semana, 800)
    
    # ⚠️ CRÍTICO: Crear features en el ORDEN EXACTO del entrenamiento
    # Orden: hora, mes, dia_semana, es_fin_semana, periodo_dia,
    #        edad_persona, grupo_edad, xx, yy,
    #        acc_hora_hist, acc_mes_hist, acc_dia_hist, densidad_local
    features = np.array([[
        hora,              # 0
        mes,               # 1
        dia_semana,        # 2
        es_fin_semana,     # 3
        periodo,           # 4
        edad,              # 5
        grupo_edad,        # 6
        lat,               # 7
        lon,               # 8
        acc_hora_hist,     # 9
        acc_mes_hist,      # 10
        acc_dia_hist,      # 11
        densidad_local     # 12
    ]])
    
    # Escalar
    features_scaled = scaler.transform(features)
    
    # Crear probabilidad del grid si hay datos exactos
    prob_grid = None
    if info_grid["metodo"] == "grid_exacto":
        prob_grid = np.zeros(3)
        prob_grid[info_grid["nivel_riesgo"]] = 1.0
    
    # Predecir con seguridad
    nivel_final, prob_final, metodo = predecir_riesgo_con_seguridad(
        features_scaled, prob_grid
    )
    
    return {
        "nivel": nivel_final,
        "probabilidades": {
            "bajo": float(prob_final[0]),
            "medio": float(prob_final[1]),
            "alto": float(prob_final[2])
        },
        "metodo": metodo,
        "confianza": info_grid["confianza"],
        "fuera_rango": fuera_rango,
        "info_zona": {
            "accidentes_historicos": info_grid["num_accidentes"],
            "densidad_local": densidad_local,
            "tipo_dato": info_grid["metodo"]
        }
    }

# === ENDPOINTS ===

@app.route("/health", methods=["GET"])
def health():
    """Verificar que el servidor está activo"""
    return jsonify({
        "status": "ok",
        "message": "🚀 Servidor activo",
        "model_info": {
            "tipo": metadata["model_type"],
            "accuracy": round(metadata["accuracy"], 3),
            "features": len(metadata["feature_names"]),
            "zonas_grid": len(grid_data),
            "fecha_entrenamiento": metadata.get("fecha_entrenamiento", "N/A"),
            "area_cobertura": {
                "lat": [
                    round(metadata["lat_min"], 4),
                    round(metadata["lat_max"], 4)
                ],
                "lon": [
                    round(metadata["lon_min"], 4),
                    round(metadata["lon_max"], 4)
                ]
            }
        }
    }), 200

@app.route("/risk-zones", methods=["GET"])
def risk_zones():
    """Retorna todas las zonas de riesgo del grid"""
    try:
        min_risk = request.args.get('min_risk', default=1, type=int)
        min_accidents = request.args.get('min_accidents', default=1, type=int)
        limit = request.args.get('limit', default=None, type=int)
        sort_by = request.args.get('sort_by', default='risk', type=str)
        
        if min_risk not in [0, 1, 2]:
            return jsonify({
                "error": "min_risk debe ser 0 (Bajo), 1 (Medio) o 2 (Alto)",
                "valor_recibido": min_risk
            }), 400
        
        if sort_by not in ['risk', 'accidents']:
            return jsonify({
                "error": "sort_by debe ser 'risk' o 'accidents'",
                "valor_recibido": sort_by
            }), 400
        
        zonas_filtradas = grid_data[
            (grid_data["nivel_riesgo"] >= min_risk) &
            (grid_data["num_accidentes"] >= min_accidents)
        ].copy()
        
        if len(zonas_filtradas) == 0:
            return jsonify({
                "zones": [],
                "count": 0,
                "message": "No se encontraron zonas con los filtros especificados"
            }), 200
        
        if sort_by == 'accidents':
            zonas_filtradas = zonas_filtradas.sort_values('num_accidentes', ascending=False)
        else:
            zonas_filtradas = zonas_filtradas.sort_values(
                ['nivel_riesgo', 'num_accidentes'], 
                ascending=[False, False]
            )
        
        if limit and limit > 0:
            zonas_filtradas = zonas_filtradas.head(limit)
        
        zones = []
        for _, row in zonas_filtradas.iterrows():
            nivel_riesgo = int(row["nivel_riesgo"])
            num_accidentes = int(row["num_accidentes"])
            
            zone = {
                "latitude": float(row["lat_grid"]),
                "longitude": float(row["lon_grid"]),
                "risk_level": ["Bajo", "Medio", "Alto"][nivel_riesgo],
                "risk_level_num": nivel_riesgo,
                "risk_score": round(calcular_risk_score(nivel_riesgo, num_accidentes), 3),
                "accident_count": num_accidentes,
                "radius": calcular_radio_zona(num_accidentes, nivel_riesgo),
            }
            zones.append(zone)
        
        distribucion = zonas_filtradas["nivel_riesgo"].value_counts().to_dict()
        
        return jsonify({
            "zones": zones,
            "count": len(zones),
            "statistics": {
                "total_zones": len(zones),
                "distribution": {
                    "bajo": int(distribucion.get(0, 0)),
                    "medio": int(distribucion.get(1, 0)),
                    "alto": int(distribucion.get(2, 0))
                },
                "total_accidents": int(zonas_filtradas["num_accidentes"].sum()),
                "avg_accidents_per_zone": round(zonas_filtradas["num_accidentes"].mean(), 2),
                "max_accidents": int(zonas_filtradas["num_accidentes"].max())
            },
            "filters_applied": {
                "min_risk_level": ["Bajo", "Medio", "Alto"][min_risk],
                "min_accidents": min_accidents,
                "limit": limit,
                "sort_by": sort_by
            },
            "generated_at": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        app.logger.error(f"Error en /risk-zones: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Error al obtener zonas de riesgo",
            "detalle": str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint principal de predicción"""
    try:
        data = request.json
        
        edad = data.get("edad")
        hora = data.get("hora")
        mes = data.get("mes")
        xx = data.get("xx")
        yy = data.get("yy")
        
        dia_semana = data.get("dia_semana")
        if dia_semana is None:
            dia_semana = datetime.now().weekday()
        
        if None in [edad, hora, mes, xx, yy]:
            return jsonify({
                "error": "Faltan datos requeridos",
                "campos_requeridos": ["edad", "hora", "mes", "xx", "yy"],
                "campos_opcionales": ["dia_semana"]
            }), 400
        
        if not (0 <= hora <= 23):
            return jsonify({"error": "hora debe estar entre 0 y 23"}), 400
        if not (1 <= mes <= 12):
            return jsonify({"error": "mes debe estar entre 1 y 12"}), 400
        if not (0 <= dia_semana <= 6):
            return jsonify({"error": "dia_semana debe estar entre 0 y 6"}), 400
        if not (0 <= edad <= 120):
            return jsonify({"error": "edad debe estar entre 0 y 120"}), 400
        
        resultado = predecir_riesgo(edad, hora, mes, dia_semana, xx, yy)
        
        nombres_riesgo = ['Bajo', 'Medio', 'Alto']
        nivel = resultado["nivel"]
        
        response = {
            "riskLevel": nombres_riesgo[nivel],
            "probability": max(resultado["probabilidades"].values()),
            "probabilities": resultado["probabilidades"],
            "latitude": xx,
            "longitude": yy,
            "mensaje": (
                "✅ Zona de bajo riesgo" if nivel == 0 else
                "⚠️ Zona de riesgo moderado" if nivel == 1 else
                "🚨 Zona de alto riesgo"
            ),
            "metodo_prediccion": resultado["metodo"],
            "confianza": resultado["confianza"],
            "info_zona": resultado["info_zona"]
        }
        
        if resultado["fuera_rango"]:
            response["advertencia"] = {
                "mensaje": "Las coordenadas están fuera del área de entrenamiento",
                "area_valida": {
                    "latitud": [
                        round(metadata["lat_min"], 2),
                        round(metadata["lat_max"], 2)
                    ],
                    "longitud": [
                        round(metadata["lon_min"], 2),
                        round(metadata["lon_max"], 2)
                    ]
                },
                "nota": "La predicción tiene menor confiabilidad"
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Error interno del servidor",
            "detalle": str(e)
        }), 500

@app.route("/grid-info", methods=["POST"])
def grid_info():
    """Obtiene información histórica de una zona específica"""
    try:
        data = request.json
        lat = data.get("xx")
        lon = data.get("yy")
        
        if lat is None or lon is None:
            return jsonify({
                "error": "Faltan coordenadas",
                "campos_requeridos": ["xx", "yy"]
            }), 400
        
        info = obtener_info_grid(lat, lon)
        densidad = calcular_densidad_local(lat, lon)
        
        return jsonify({
            "nivel_riesgo": ["Bajo", "Medio", "Alto"][info["nivel_riesgo"]],
            "accidentes_historicos": info["num_accidentes"],
            "densidad_radio_5km": densidad,
            "metodo": info["metodo"],
            "confianza": info["confianza"],
            "distancia_zona_cercana_km": info.get("distancia_km", 0)
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": "Error al obtener información del grid",
            "detalle": str(e)
        }), 500

@app.route("/stats", methods=["GET"])
def stats():
    """Estadísticas generales del modelo y dataset"""
    return jsonify({
        "modelo": {
            "tipo": metadata["model_type"],
            "accuracy": round(metadata["accuracy"], 3),
            "fecha_entrenamiento": metadata.get("fecha_entrenamiento", "N/A")
        },
        "zonas": {
            "total": len(grid_data),
            "distribucion_riesgo": {
                "bajo": int((grid_data["nivel_riesgo"] == 0).sum()),
                "medio": int((grid_data["nivel_riesgo"] == 1).sum()),
                "alto": int((grid_data["nivel_riesgo"] == 2).sum())
            }
        },
        "accidentes": {
            "total": int(grid_data["num_accidentes"].sum()),
            "promedio_por_zona": round(grid_data["num_accidentes"].mean(), 1),
            "maximo_en_zona": int(grid_data["num_accidentes"].max())
        },
        "zona_mayor_riesgo": {
            "latitud": float(grid_data.loc[grid_data["num_accidentes"].idxmax(), "lat_grid"]),
            "longitud": float(grid_data.loc[grid_data["num_accidentes"].idxmax(), "lon_grid"]),
            "accidentes": int(grid_data["num_accidentes"].max())
        }
    }), 200

@app.route("/info", methods=["GET"])
def info():
    """Información detallada sobre el modelo"""
    return jsonify({
        "modelo": "Predicción de Riesgo de Accidentes - Arequipa",
        "version": "2.5",
        "tipo": metadata["model_type"],
        "accuracy": round(metadata["accuracy"], 3),
        "features": metadata["feature_names"],
        "niveles_riesgo": ["Bajo", "Medio", "Alto"],
        "mejoras_seguridad": {
            "umbral_ajustado": "20% probabilidad de riesgo alto activa alerta",
            "prioridad": "Evitar falsos negativos (predecir bajo cuando es alto)"
        },
        "endpoints": {
            "/health": "Estado del servidor",
            "/predict": "Predicción de riesgo",
            "/risk-zones": "Zonas de riesgo",
            "/grid-info": "Info histórica de zona",
            "/stats": "Estadísticas",
            "/info": "Info del modelo"
        }
    }), 200

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 API de Predicción de Riesgo de Accidentes")
    print("="*60)
    print(f"📊 Modelo: {metadata['model_type']}")
    print(f"🎯 Accuracy: {metadata['accuracy']:.1%}")
    print(f"🗺️  Zonas: {len(grid_data):,}")
    print(f"🔧 Features: {len(metadata['feature_names'])}")
    print("\n📋 Endpoints disponibles:")
    print("   • GET  /health")
    print("   • GET  /risk-zones")
    print("   • POST /predict")
    print("   • POST /grid-info")
    print("   • GET  /stats")
    print("   • GET  /info")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
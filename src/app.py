import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
from datetime import datetime
from scipy.spatial import cKDTree

# === PATHS FLEXIBLES (funciona igual en local y Render) ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Determinar si estamos en Render o local
if "/opt/render/project/src" in CURRENT_DIR:
    PROJECT_ROOT = "/opt/render/project/src"
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

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

app = Flask(__name__)

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

def calcular_densidad_local(lat, lon, radio=0.02):
    """
    Calcula densidad de accidentes en un radio (~2km).
    Busca celdas del grid cercanas.
    """
    punto = np.array([lat, lon])
    indices = grid_tree.query_ball_point(punto, r=radio)
    
    if not indices or len(indices) == 0:
        return 1  # Densidad mínima si no hay datos cercanos
    
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
    
    # Manejar resultado escalar o array
    if isinstance(indices, (int, np.integer)):
        idx = indices
        dist = distancias
    else:
        idx = indices[0] if len(indices) > 0 else 0
        dist = distancias[0] if len(distancias) > 0 else 0
    
    celda_cercana = grid_data.iloc[idx]
    distancia_km = float(dist) * 111  # Convertir grados a km
    
    return {
        "nivel_riesgo": int(celda_cercana["nivel_riesgo"]),
        "num_accidentes": int(celda_cercana["num_accidentes"]),
        "metodo": "grid_aproximado",
        "distancia_km": round(distancia_km, 2),
        "confianza": "media" if distancia_km < 5 else "baja"
    }

def calcular_radio_zona(num_accidentes, nivel_riesgo):
    """
    Calcula el radio de la zona de riesgo basado en el número de accidentes.
    Más accidentes = radio mayor.
    """
    # Radio base según nivel de riesgo
    radio_base = {
        0: 50,   # Bajo: 50m
        1: 100,  # Medio: 100m
        2: 150   # Alto: 150m
    }
    
    # Radio ajustado por densidad de accidentes
    if num_accidentes >= 10:
        multiplicador = 1.5
    elif num_accidentes >= 5:
        multiplicador = 1.2
    else:
        multiplicador = 1.0
    
    return int(radio_base.get(nivel_riesgo, 100) * multiplicador)

def calcular_risk_score(nivel_riesgo, num_accidentes):
    """
    Calcula un score de riesgo normalizado entre 0 y 1.
    Combina nivel de riesgo con número de accidentes.
    """
    # Score base según nivel
    score_base = {
        0: 0.3,   # Bajo
        1: 0.6,   # Medio
        2: 0.85   # Alto
    }
    
    base = score_base.get(nivel_riesgo, 0.5)
    
    # Ajuste por número de accidentes (máximo +0.15)
    ajuste = min(num_accidentes / 100, 0.15)
    
    return min(base + ajuste, 1.0)

def predecir_riesgo(edad, hora, mes, dia_semana, lat, lon):
    """
    Realiza predicción usando estrategia híbrida:
    1. Obtiene información del grid histórico
    2. Combina con predicción del modelo ML
    """
    # Verificar si está en el área de entrenamiento
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
    num_accidentes = info_grid["num_accidentes"]
    
    # Crear features en el orden correcto
    features = np.array([[
        edad,
        grupo_edad,
        hora,
        mes,
        dia_semana,
        es_fin_semana,
        periodo,
        lat,
        lon,
        densidad_local,
        num_accidentes
    ]])
    
    # Escalar y predecir
    features_scaled = scaler.transform(features)
    
    # Obtener probabilidades del modelo
    prob_ml = model.predict_proba(features_scaled)[0]
    
    # Si hay datos históricos confiables, combinar predicciones
    if info_grid["metodo"] == "grid_exacto":
        # 60% modelo ML + 40% histórico
        prob_final = prob_ml * 0.6
        prob_final[info_grid["nivel_riesgo"]] += 0.4
        prob_final = prob_final / prob_final.sum()
        metodo = "hibrido"
    else:
        # Usar solo modelo ML
        prob_final = prob_ml
        metodo = "modelo_ml"
    
    nivel_final = int(np.argmax(prob_final))
    
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
            "accidentes_historicos": num_accidentes,
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
    """
    Retorna todas las zonas de riesgo del grid.
    
    Parámetros query opcionales:
    - min_risk: nivel mínimo de riesgo (0=Bajo, 1=Medio, 2=Alto). Default: 1 (Medio y Alto)
    - min_accidents: número mínimo de accidentes. Default: 1
    - limit: límite de zonas a retornar. Default: todas
    - sort_by: ordenar por 'accidents' o 'risk'. Default: 'risk'
    
    Ejemplos:
    - /risk-zones                          -> Todas las zonas con riesgo medio o alto
    - /risk-zones?min_risk=2               -> Solo zonas de alto riesgo
    - /risk-zones?min_accidents=5          -> Zonas con 5+ accidentes
    - /risk-zones?limit=50&sort_by=accidents -> Top 50 zonas con más accidentes
    """
    try:
        # Obtener parámetros
        min_risk = int(request.args.get('min_risk', 1))  # Default: medio y alto
        min_accidents = int(request.args.get('min_accidents', 1))
        limit = request.args.get('limit', None)
        sort_by = request.args.get('sort_by', 'risk')  # 'risk' o 'accidents'
        
        # Validar parámetros
        if min_risk not in [0, 1, 2]:
            return jsonify({
                "error": "min_risk debe ser 0 (Bajo), 1 (Medio) o 2 (Alto)"
            }), 400
        
        # Filtrar zonas
        zonas_filtradas = grid_data[
            (grid_data["nivel_riesgo"] >= min_risk) &
            (grid_data["num_accidentes"] >= min_accidents)
        ].copy()
        
        # Ordenar
        if sort_by == 'accidents':
            zonas_filtradas = zonas_filtradas.sort_values('num_accidentes', ascending=False)
        else:  # sort_by == 'risk'
            zonas_filtradas = zonas_filtradas.sort_values(
                ['nivel_riesgo', 'num_accidentes'], 
                ascending=[False, False]
            )
        
        # Aplicar límite si existe
        if limit:
            try:
                limit = int(limit)
                zonas_filtradas = zonas_filtradas.head(limit)
            except ValueError:
                pass
        
        # Construir respuesta
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
        
        # Estadísticas de las zonas retornadas
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
    """
    Endpoint principal de predicción.
    
    Body JSON esperado:
    {
        "edad": 35,
        "hora": 18,
        "mes": 10,
        "dia_semana": 4,  // Opcional, default: día actual
        "xx": -16.4,      // Latitud
        "yy": -71.5       // Longitud
    }
    """
    try:
        data = request.json
        
        # Validar datos requeridos
        edad = data.get("edad")
        hora = data.get("hora")
        mes = data.get("mes")
        xx = data.get("xx")  # Latitud
        yy = data.get("yy")  # Longitud
        
        # Día de la semana opcional
        dia_semana = data.get("dia_semana")
        if dia_semana is None:
            dia_semana = datetime.now().weekday()
        
        # Validar campos requeridos
        if None in [edad, hora, mes, xx, yy]:
            return jsonify({
                "error": "Faltan datos requeridos",
                "campos_requeridos": ["edad", "hora", "mes", "xx", "yy"],
                "campos_opcionales": ["dia_semana"]
            }), 400
        
        # Validar rangos
        if not (0 <= hora <= 23):
            return jsonify({"error": "hora debe estar entre 0 y 23"}), 400
        if not (1 <= mes <= 12):
            return jsonify({"error": "mes debe estar entre 1 y 12"}), 400
        if not (0 <= dia_semana <= 6):
            return jsonify({"error": "dia_semana debe estar entre 0 (lunes) y 6 (domingo)"}), 400
        if not (0 <= edad <= 120):
            return jsonify({"error": "edad debe estar entre 0 y 120"}), 400
        
        # Realizar predicción
        resultado = predecir_riesgo(edad, hora, mes, dia_semana, xx, yy)
        
        nombres_riesgo = ['Bajo', 'Medio', 'Alto']
        nivel = resultado["nivel"]
        
        # Construir respuesta
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
        
        # Agregar advertencia si está fuera de rango
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
    """
    Obtiene información histórica de una zona específica.
    
    Body JSON:
    {
        "xx": -16.4,
        "yy": -71.5
    }
    """
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
            "densidad_radio_2km": densidad,
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
        },
        "area_cobertura": {
            "latitud": {
                "min": round(metadata["lat_min"], 4),
                "max": round(metadata["lat_max"], 4)
            },
            "longitud": {
                "min": round(metadata["lon_min"], 4),
                "max": round(metadata["lon_max"], 4)
            }
        }
    }), 200

@app.route("/info", methods=["GET"])
def info():
    """Información detallada sobre el modelo"""
    return jsonify({
        "modelo": "Predicción de Riesgo de Accidentes - Arequipa",
        "version": "2.1",
        "tipo": metadata["model_type"],
        "accuracy": round(metadata["accuracy"], 3),
        "features": metadata["feature_names"],
        "niveles_riesgo": ["Bajo", "Medio", "Alto"],
        "metodos_prediccion": {
            "hibrido": "Combina datos históricos (40%) con modelo ML (60%)",
            "modelo_ml": "Usa solo el modelo cuando no hay datos históricos exactos"
        },
        "area_cobertura": {
            "descripcion": "Coordenadas de entrenamiento (Arequipa-Camaná)",
            "latitud": {
                "min": round(metadata["lat_min"], 4),
                "max": round(metadata["lat_max"], 4)
            },
            "longitud": {
                "min": round(metadata["lon_min"], 4),
                "max": round(metadata["lon_max"], 4)
            }
        },
        "endpoints": {
            "/health": "Estado del servidor",
            "/predict": "Predicción de riesgo para coordenadas específicas",
            "/risk-zones": "Obtener todas las zonas de riesgo del mapa",
            "/grid-info": "Información histórica de una zona",
            "/stats": "Estadísticas generales",
            "/info": "Información del modelo"
        },
        "ejemplo_request_predict": {
            "edad": 35,
            "hora": 18,
            "mes": 10,
            "dia_semana": 4,
            "xx": -16.4,
            "yy": -71.5
        },
        "ejemplo_request_risk_zones": {
            "url": "/risk-zones?min_risk=2&min_accidents=5&limit=100",
            "descripcion": "Top 100 zonas de alto riesgo con 5+ accidentes"
        }
    }), 200

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 API de Predicción de Riesgo de Accidentes")
    print("="*60)
    print(f"📊 Modelo: {metadata['model_type']}")
    print(f"🎯 Accuracy: {metadata['accuracy']:.1%}")
    print(f"🗺️  Zonas: {len(grid_data):,}")
    print(f"📍 Área:")
    print(f"   Lat: [{metadata['lat_min']:.2f}, {metadata['lat_max']:.2f}]")
    print(f"   Lon: [{metadata['lon_min']:.2f}, {metadata['lon_max']:.2f}]")
    print("\n📋 Endpoints disponibles:")
    print("   • GET  /health")
    print("   • GET  /risk-zones")
    print("   • POST /predict")
    print("   • POST /grid-info")
    print("   • GET  /stats")
    print("   • GET  /info")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
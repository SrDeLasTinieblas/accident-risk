import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from joblib import load
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# Paths de archivos
MODEL_PATH = os.path.join(MODEL_DIR, "model_riesgo_accidentes.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_zonas.joblib")
FEATURE_INFO_PATH = os.path.join(MODEL_DIR, "feature_info.joblib")

# Validar que existan todos los archivos
required_files = {
    "modelo": MODEL_PATH,
    "scaler": SCALER_PATH,
    "kmeans": KMEANS_PATH,
    "feature_info": FEATURE_INFO_PATH
}

for name, path in required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró {name}: {path}")

# Cargar modelos (una sola vez al iniciar)
print("🔄 Cargando modelos...")
model = load_model(MODEL_PATH)
scaler = load(SCALER_PATH)
kmeans = load(KMEANS_PATH)
feature_info = load(FEATURE_INFO_PATH)
print("✅ Modelos cargados exitosamente")

app = Flask(__name__)

def categorizar_hora(hora):
    """Categoriza la hora del día en periodos"""
    if 6 <= hora < 12:
        return 0  # Mañana
    elif 12 <= hora < 18:
        return 1  # Tarde
    elif 18 <= hora < 24:
        return 2  # Noche
    else:
        return 3  # Madrugada

def calcular_features(edad, hora, mes, dia_semana, latitud, longitud):
    """
    Calcula todas las features necesarias para la predicción.
    
    Returns:
        tuple: (features_array, fuera_rango, info_dict)
    """
    # Features temporales
    es_fin_semana = 1 if dia_semana >= 5 else 0
    periodo = categorizar_hora(hora)
    
    # Verificar si está dentro del rango de entrenamiento
    fuera_rango = (
        latitud < feature_info["lat_min"] or 
        latitud > feature_info["lat_max"] or
        longitud < feature_info["lon_min"] or 
        longitud > feature_info["lon_max"]
    )
    
    # Asignar al cluster más cercano
    coords_nueva = np.array([[latitud, longitud]])
    cluster_zona = int(kmeans.predict(coords_nueva)[0])
    
    # Calcular distancias a todos los clusters
    centroids = kmeans.cluster_centers_
    n_clusters = feature_info["n_clusters"]
    
    distancias = []
    for i in range(n_clusters):
        dist = np.sqrt(
            (latitud - centroids[i][0])**2 + 
            (longitud - centroids[i][1])**2
        )
        distancias.append(dist)
    
    # Obtener las 3 distancias mínimas
    distancias_sorted = sorted(distancias)
    dist_min_1 = distancias_sorted[0]
    dist_min_2 = distancias_sorted[1]
    dist_min_3 = distancias_sorted[2]
    
    # Estimar densidad local (inversa a la distancia del cluster más cercano)
    # Valores típicos entre 1-50
    densidad_estimada = max(1, int(20 / (1 + dist_min_1 * 10)))
    
    # Estimar accidentes del cluster (valor promedio)
    accidentes_cluster = 50
    
    # Crear array de features en el orden correcto
    # ["edad_persona", "hora", "mes", "dia_semana", "es_fin_semana", 
    #  "periodo_dia", "cluster_zona", "dist_min_1", "dist_min_2", 
    #  "dist_min_3", "densidad_local", "accidentes_cluster"]
    
    features = np.array([[
        edad,
        hora,
        mes,
        dia_semana,
        es_fin_semana,
        periodo,
        cluster_zona,
        dist_min_1,
        dist_min_2,
        dist_min_3,
        densidad_estimada,
        accidentes_cluster
    ]])
    
    info = {
        "cluster_zona": cluster_zona,
        "dist_min_1": round(dist_min_1, 4),
        "densidad_estimada": densidad_estimada,
        "fuera_rango": fuera_rango
    }
    
    return features, fuera_rango, info

@app.route("/health", methods=["GET"])
def health():
    """Endpoint para verificar que el servidor está activo"""
    return jsonify({
        "status": "ok", 
        "message": "🚀 Servidor activo",
        "model_info": {
            "features_esperadas": len(feature_info["feature_names"]),
            "clusters": feature_info["n_clusters"],
            "area_cobertura": {
                "lat": [feature_info["lat_min"], feature_info["lat_max"]],
                "lon": [feature_info["lon_min"], feature_info["lon_max"]]
            }
        }
    }), 200

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
        
        # Día de la semana opcional (default: hoy)
        dia_semana = data.get("dia_semana")
        if dia_semana is None:
            dia_semana = datetime.now().weekday()
        
        # Validar que no falten datos
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
        
        # Calcular features
        features, fuera_rango, info = calcular_features(
            edad, hora, mes, dia_semana, xx, yy
        )
        
        # Escalar features
        X_scaled = scaler.transform(features)
        
        # Predecir
        prob = model.predict(X_scaled, verbose=0)[0]
        nivel = int(np.argmax(prob))
        
        nombres_riesgo = ['Bajo', 'Medio', 'Alto']
        
        # Construir respuesta
        response = {
            "riskLevel": nombres_riesgo[nivel],
            "probabilities": {
                "bajo": float(prob[0]),
                "medio": float(prob[1]),
                "alto": float(prob[2])
            },
            "latitude": xx,
            "longitude": yy,
            "cluster": info["cluster_zona"],
            "mensaje": (
                "✅ Zona de bajo riesgo" if nivel == 0 else
                "⚠️ Zona de riesgo moderado" if nivel == 1 else
                "🚨 Zona de alto riesgo"
            )
        }
        
        # Agregar advertencia si está fuera de rango
        if fuera_rango:
            response["advertencia"] = {
                "mensaje": "⚠️ Las coordenadas están fuera del área de entrenamiento del modelo",
                "area_valida": {
                    "latitud": [
                        round(feature_info["lat_min"], 2),
                        round(feature_info["lat_max"], 2)
                    ],
                    "longitud": [
                        round(feature_info["lon_min"], 2),
                        round(feature_info["lon_max"], 2)
                    ]
                },
                "nota": "La predicción es una aproximación basada en el cluster más cercano"
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            "error": "Error interno del servidor",
            "detalle": str(e)
        }), 500

@app.route("/info", methods=["GET"])
def info():
    """Endpoint con información sobre el modelo"""
    return jsonify({
        "modelo": "Predicción de Riesgo de Accidentes",
        "version": "2.0",
        "features": feature_info["feature_names"],
        "num_clusters": feature_info["n_clusters"],
        "niveles_riesgo": ["Bajo", "Medio", "Alto"],
        "area_cobertura": {
            "descripcion": "Coordenadas de entrenamiento",
            "latitud": {
                "min": round(feature_info["lat_min"], 4),
                "max": round(feature_info["lat_max"], 4)
            },
            "longitud": {
                "min": round(feature_info["lon_min"], 4),
                "max": round(feature_info["lon_max"], 4)
            }
        },
        "ejemplo_request": {
            "edad": 35,
            "hora": 18,
            "mes": 10,
            "dia_semana": 4,
            "xx": -16.4,
            "yy": -71.5
        }
    }), 200

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Servidor de Predicción de Riesgo de Accidentes")
    print("="*50)
    print(f"📊 Features del modelo: {len(feature_info['feature_names'])}")
    print(f"🗺️  Clusters: {feature_info['n_clusters']}")
    print(f"📍 Área de cobertura:")
    print(f"   Latitud:  [{feature_info['lat_min']:.2f}, {feature_info['lat_max']:.2f}]")
    print(f"   Longitud: [{feature_info['lon_min']:.2f}, {feature_info['lon_max']:.2f}]")
    print("="*50 + "\n")
    
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000)

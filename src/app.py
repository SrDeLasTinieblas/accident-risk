import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model_riesgo_accidentes.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ No se encontró el modelo en {MODEL_PATH}. Entrena el modelo primero con train_model.py"
    )

# === Cargar modelo ===
model = load_model(MODEL_PATH)

app = Flask(__name__)

# === Health Check ===
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "🚀 Modelo cargado correctamente y servidor activo"
    }), 200


# === Predicción ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    edad = data.get("edad")
    hora = data.get("hora")
    mes = data.get("mes")
    xx = data.get("xx")
    yy = data.get("yy")

    # Validación de entrada
    if None in [edad, hora, mes, xx, yy]:
        return jsonify({"error": "Faltan datos"}), 400

    # Normalizar los valores (igual que en el entrenamiento)
    arr = np.array([[edad, hora, mes, xx, yy]])
    prob = float(model.predict(arr)[0][0])

    # Determinar el nivel de riesgo
    if prob >= 0.7:
        nivel_riesgo = "Alto"
        mensaje = "🚨 Zona de alto riesgo"
    elif prob >= 0.4:
        nivel_riesgo = "Medio"
        mensaje = "⚠️ Zona de riesgo moderado"
    else:
        nivel_riesgo = "Bajo"
        mensaje = "✅ Zona de bajo riesgo"

    # Estructura estándar esperada por el front
    return jsonify({
        "riskLevel": nivel_riesgo,
        "probability": prob,
        "latitude": xx,
        "longitude": yy,
        "mensaje": mensaje,
        "riesgo": prob  # por compatibilidad con versiones anteriores
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

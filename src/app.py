import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from joblib import load
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model_riesgo_accidentes.h5")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ No se encontró el modelo")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("❌ No se encontró el scaler")

model = load_model(MODEL_PATH)
scaler = load(SCALER_PATH)

app = Flask(__name__)

def categorizar_hora(hora):
    if 6 <= hora < 12:
        return 0
    elif 12 <= hora < 18:
        return 1
    elif 18 <= hora < 24:
        return 2
    else:
        return 3

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "🚀 Servidor activo"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    edad = data.get("edad")
    hora = data.get("hora")
    mes = data.get("mes")
    dia_semana = data.get("dia_semana", datetime.now().weekday())
    xx = data.get("xx")
    yy = data.get("yy")

    if None in [edad, hora, mes, xx, yy]:
        return jsonify({"error": "Faltan datos"}), 400

    es_fin_semana = 1 if dia_semana >= 5 else 0
    periodo = categorizar_hora(hora)

    X = np.array([[edad, hora, mes, dia_semana, es_fin_semana, periodo, xx, yy]])
    X_scaled = scaler.transform(X)

    prob = model.predict(X_scaled, verbose=0)[0]
    nivel = np.argmax(prob)
    nombres_riesgo = ['Bajo', 'Medio', 'Alto']

    return jsonify({
        "riskLevel": nombres_riesgo[nivel],
        "probabilities": {
            "bajo": float(prob[0]),
            "medio": float(prob[1]),
            "alto": float(prob[2])
        },
        "latitude": xx,
        "longitude": yy,
        "mensaje": (
            "✅ Zona de bajo riesgo" if nivel == 0 else
            "⚠️ Zona de riesgo moderado" if nivel == 1 else
            "🚨 Zona de alto riesgo"
        )
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

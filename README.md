# Accident Risk

Monorepo con la app móvil y el servicio del modelo de riesgo de accidentes.

```
apps/mobile/          App Expo / React Native
services/risk-api/    API Flask + modelo Random Forest
  app/                API (app.main:app)
  training/           Entrenamiento (train_model.py)
  artifacts/          Modelo entrenado, scaler, grid de riesgo, métricas
  notebooks/
data/raw/             Excel históricos (NO se versionan, ver abajo)
docs/
```

## App móvil
```
cd apps/mobile
npm install
npx expo start
```
Para EAS: ejecutar `eas build` desde `apps/mobile`.

## API del modelo
```
cd services/risk-api
py -3.10 -m venv .venv310
.\.venv310\Scripts\activate
pip install -r requirements.txt
python -m app.main            # http://localhost:5000
python -m training.train_model   # reentrenar (necesita data/raw/*.xlsx)
```
Variables opcionales: `MODEL_DIR`, `DATA_DIR`.

**Despliegue (Render):** configurar *Root Directory* = `services/risk-api` y
comando de inicio `gunicorn app.main:app` (antes era `src.app:app`).

## Datos
Los Excel mensuales van en `data/raw/` y están en `.gitignore`. La API en
producción funciona sin ellos si no existen; son necesarios para reentrenar.
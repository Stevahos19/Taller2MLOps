# api_salary_prediction.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# Ruta al modelo (ajusta según tu entorno)
MODEL_PATH = '/home/david-v/Documentos/Especializacion/MLOps/predictive_maintenance/src/salary_prediction_pipeline_v2.joblib'
pipeline = None

# --- Carga del modelo al iniciar ---
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        pipeline = None

# --- Definición del esquema de entrada con Pydantic ---
class SalaryInput(BaseModel):
    work_year: int = Field(..., example=2023)
    remote_ratio: int = Field(..., example=100)
    experience_level: str = Field(..., example="SE")
    employment_type: str = Field(..., example="FT")
    job_title: str = Field(..., example="Data Scientist")
    employee_residence: str = Field(..., example="US")
    company_location: str = Field(..., example="US")
    company_size: str = Field(..., example="M")
    salary_currency: str = Field(..., example="USD")

# --- Esquema de salida ---
class SalaryPredictionResponse(BaseModel):
    predicted_salary_in_usd: float
    message: str

# --- Inicialización de la app ---
app = FastAPI(
    title="API de Predicción de Salarios de Data Science",
    description="Predice el salario anual en USD según características del empleo.",
    version="1.0"
)

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "API funcionando. Ir a /docs para probar."}

@app.post("/predict", response_model=SalaryPredictionResponse)
async def predict_salary(data: SalaryInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado correctamente.")

    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = pipeline.predict(input_df)[0]

        return SalaryPredictionResponse(
            predicted_salary_in_usd=round(float(prediction), 2),
            message="Predicción generada con éxito."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción: {e}")

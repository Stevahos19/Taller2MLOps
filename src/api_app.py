from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
from typing import List, Optional

# --- 1. Cargar el Modelo Entrenado ---
MODEL_PATH = os.path.join('models', 'modelo_regresion_logistica_mantenimiento_v1.joblib') 
pipeline = None # Variable global para almacenar el modelo cargado

def load_model_on_startup():
    global pipeline # Indicamos que vamos a modificar la variable global 'pipeline'
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: El archivo del modelo no se encontró en '{MODEL_PATH}'. Asegúrate de que la ruta es correcta.")
        pipeline = None # Aseguramos que el pipeline sea None si no se carga
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        pipeline = None

# --- 2. Definir Modelos de Pydantic para la Entrada y Salida de Datos ---

# Modelo para los datos de entrada (lo que la API espera recibir)
class DeviceMetrics(BaseModel):
    # Field se usa para añadir metadatos como descripción y ejemplos a la documentación de la API
    metric1: float = Field(..., description="Valor de la Métrica 1", example=10000000.0)
    metric2: float = Field(..., description="Valor de la Métrica 2", example=50.0)
    metric3: float = Field(..., description="Valor de la Métrica 3", example=1.0)
    metric4: float = Field(..., description="Valor de la Métrica 4", example=40.0)
    metric5: float = Field(..., description="Valor de la Métrica 5", example=5.0)
    metric6: float = Field(..., description="Valor de la Métrica 6", example=300000.0)
    metric7: float = Field(..., description="Valor de la Métrica 7", example=0.0)
    metric8: float = Field(..., description="Valor de la Métrica 8", example=0.0)
    metric9: float = Field(..., description="Valor de la Métrica 9", example=5.0)

# Modelo para la respuesta de la API (lo que la API devolverá)
class PredictionResponse(BaseModel):
    predicted_failure: int
    failure_probability: float
    message: str

# --- 3. Inicializar la Aplicación FastAPI ---
app = FastAPI(
    title="API de Mantenimiento Predictivo",
    description="API para predecir fallas de dispositivos usando un modelo de Regresión Logística.",
    version="1.0.0"
)

# --- 4. Definir Eventos de Inicio/Apagado (para cargar el modelo) ---
@app.on_event("startup")
async def startup_event():
    """Se ejecuta cuando la aplicación FastAPI se inicia."""
    load_model_on_startup()

# --- 5. Definir Endpoints de la API ---

@app.get("/") # Decorador que define que esta función manejará solicitudes GET a la ruta raíz "/"
async def root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {"message": "API de Mantenimiento Predictivo funcionando. Ve a /docs para la documentación."}

@app.post("/predict", response_model=PredictionResponse) # Decorador para solicitudes POST a "/predict"
async def predict_failure(metrics: DeviceMetrics):
    """
    Realiza una predicción de falla para un dispositivo basándose en sus métricas.
    """
    # Verifica si el modelo se cargó correctamente al inicio de la aplicación
    if pipeline is None:
        # Si no, lanza una excepción HTTP 500 (Internal Server Error)
        raise HTTPException(status_code=500, detail="Modelo no cargado. Contacta al administrador.")

    # Convertir los datos de entrada de Pydantic a un DataFrame de Pandas
    # metrics es un objeto de tipo DeviceMetrics, y .dict() lo convierte a un diccionario
    input_df = pd.DataFrame([metrics.dict()])
    
    # Asegurarse de que el DataFrame tenga las columnas en el orden esperado
    # Esto es VITAL para que el pipeline de scikit-learn (especialmente el StandardScaler)
    # procese los datos correctamente en el mismo orden que fue entrenado.
    ordered_columns = [f'metric{i}' for i in range(1, 10)]
    input_df = input_df[ordered_columns]

    try:
        # Realizar la predicción usando el pipeline cargado
        # pipeline.predict() devuelve un array, [0] lo desenvuelve para obtener el valor único
        prediction = pipeline.predict(input_df)[0]
        # pipeline.predict_proba() devuelve probabilidades para ambas clases [0, 1].
        # [: , 1] selecciona la probabilidad de la clase positiva (1, "Falla")
        # [0] la desenvuelve para obtener el valor único.
        probability = pipeline.predict_proba(input_df)[:, 1][0]

        message = "No hay signos de Falla Detectada"
        if prediction == 1:
            message = "Posible Falla Detectada"

        # Devolver la respuesta usando el modelo Pydantic PredictionResponse
        return PredictionResponse(
            predicted_failure=int(prediction), # Aseguramos que sea int
            failure_probability=float(probability), # Aseguramos que sea float
            message=message
        )
    except Exception as e:
        # Captura cualquier error inesperado durante la predicción
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")
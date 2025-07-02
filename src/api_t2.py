# api_t2.py (o el nombre que uses para tu archivo FastAPI)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
from typing import List, Optional

# --- 1. Load the Trained Model ---
# IMPORTANT: Update this path to the exact location of your .joblib file
# Make sure this path is accessible from where your FastAPI application will run.
MODEL_PATH = '/home/david-v/Documentos/Especializacion/MLOps/predictive_maintenance/src/salary_prediction_pipeline_v2.joblib'
pipeline = None # Global variable to store the loaded model

def load_model_on_startup():
    global pipeline # Indicate that we will modify the global 'pipeline' variable
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'. Ensure the path is correct and the file exists.")
        pipeline = None # Ensure pipeline is None if loading fails
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        pipeline = None

# --- 2. Define Pydantic Models for Data Input and Output ---

# Model for input data (what the API expects to receive)
class SalaryInput(BaseModel):
    # These fields must match the features your salary prediction model expects
    work_year: int = Field(..., description="The year the salary was paid.", example=2023)
    experience_level: str = Field(..., description="The experience level in the job during the year (e.g., SE, MI, EN, EX).", example="SE")
    employment_type: str = Field(..., description="The type of employment for the role (e.g., FT, PT, CT, FL).", example="FT")
    job_title: str = Field(..., description="The role worked in during the year (e.g., Data Scientist, ML Engineer).", example="Data Scientist")
    employee_residence: str = Field(..., description="Employee's primary country of residence (ISO 3166 country code).", example="US")
    remote_ratio: int = Field(..., description="The overall amount of work done remotely (0, 50, or 100).", example=100)
    company_location: str = Field(..., description="The country of the employer's main office (ISO 3166 country code).", example="US")
    company_size: str = Field(..., description="The median number of people that worked for the company (S, M, L).", example="M")

# Model for the API response (what the API will return)
class SalaryPredictionResponse(BaseModel):
    predicted_salary_in_usd: float
    message: str

# --- 3. Initialize the FastAPI Application ---
app = FastAPI(
    title="API de Predicción de Salarios de Data Science",
    description="API para predecir salarios en USD usando un modelo de regresión.",
    version="1.0.0"
)

# --- 4. Define Startup/Shutdown Events (to load the model) ---
@app.on_event("startup")
async def startup_event():
    """Executes when the FastAPI application starts."""
    load_model_on_startup()

# --- 5. Define API Endpoints ---

@app.get("/") # Decorator that defines this function will handle GET requests to the root path "/"
async def root():
    """Root endpoint to verify that the API is working."""
    return {"message": "API de Predicción de Salarios de Data Science funcionando. Ve a /docs para la documentación."}

@app.post("/predict", response_model=SalaryPredictionResponse) # Decorator for POST requests to "/predict"
async def predict_salary(data: SalaryInput):
    """
    Performs a salary prediction in USD based on the provided input data.
    """
    # Check if the model was loaded successfully at application startup
    if pipeline is None:
        # If not, raise an HTTP 500 (Internal Server Error) exception
        raise HTTPException(status_code=500, detail="Modelo no cargado. Contacta al administrador.")

    # Convert the Pydantic input data to a Pandas DataFrame
    # It's crucial that the column names and order in the DataFrame
    # match the data with which the pipeline was trained.
    input_df = pd.DataFrame([data.model_dump()])

    # Ensure the DataFrame has the columns in the expected order
    # This is VITAL for the scikit-learn pipeline (especially the preprocessor)
    # to process the data correctly in the same order it was trained.
    # Based on your model_pipeline.py, the order of features is:
    # numerical_features = ['work_year', 'remote_ratio']
    # categorical_features = ['experience_level', 'employment_type', 'job_title',
    #                         'employee_residence', 'company_location', 'company_size']
    ordered_columns = [
        'work_year', 'remote_ratio',
        'experience_level', 'employment_type', 'job_title',
        'employee_residence', 'company_location', 'company_size'
    ]
    input_df = input_df[ordered_columns]

    try:
        # Perform the prediction using the loaded pipeline
        # pipeline.predict() returns an array, [0] unwraps it to get the single value
        prediction = pipeline.predict(input_df)[0]

        message = f"Predicción de salario generada."

        # Return the response using the Pydantic SalaryPredictionResponse model
        return SalaryPredictionResponse(
            predicted_salary_in_usd=float(round(prediction, 2)), # Ensure it's float and rounded
            message=message
        )
    except Exception as e:
        # Catch any unexpected errors during prediction
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

# src/streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import os # Para manejar rutas de archivos

# --- 1. Configuración de la aplicación ---
st.set_page_config(page_title="Mantenimiento Predictivo", layout="centered")

# --- 2. Cargar el modelo entrenado ---
# Asegúrate de que esta ruta sea correcta para donde está tu modelo .joblib
# Si tu modelo está en 'mi_proyecto_mantenimiento/models/modelo_regresion_logistica_mantenimiento_v1.joblib'
# y este script está en 'mi_proyecto_mantenimiento/src/streamlit_app.py',
# entonces la ruta relativa sería:
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'modelo_regresion_logistica_mantenimiento_v1.joblib')

# Intentar cargar el modelo
@st.cache_resource # Almacena en caché el modelo para no cargarlo cada vez que Streamlit se actualiza
def load_model(path):
    try:
        loaded_pipeline = joblib.load(path)
        st.success("Modelo cargado exitosamente.")
        return loaded_pipeline
    except FileNotFoundError:
        st.error(f"Error: El archivo del modelo no se encontró en '{path}'. Asegúrate de que la ruta es correcta.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

pipeline = load_model(model_path)

# --- 3. Título y descripción de la GUI ---
st.title("⚙️ Mantenimiento Predictivo de Dispositivos")
st.markdown("""
Esta aplicación predice la probabilidad de falla de un dispositivo
basándose en sus métricas operativas.
""")

if pipeline is not None:
    st.subheader("Ingrese las métricas del dispositivo:")

    # --- 4. Inputs de usuario para cada métrica ---
    # Usamos st.number_input para permitir al usuario introducir valores numéricos
    # Puedes ajustar los valores min_value, max_value y value (por defecto)
    # según el rango esperado de tus métricas.
    col1, col2, col3 = st.columns(3)

    with col1:
        metric1 = st.number_input("Métrica 1", min_value=0.0, value=10000000.0, format="%.2f")
        metric2 = st.number_input("Métrica 2", min_value=0.0, value=50.0, format="%.2f")
        metric3 = st.number_input("Métrica 3", min_value=0.0, value=1.0, format="%.2f")
    with col2:
        metric4 = st.number_input("Métrica 4", min_value=0.0, value=40.0, format="%.2f")
        metric5 = st.number_input("Métrica 5", min_value=0.0, value=5.0, format="%.2f")
        metric6 = st.number_input("Métrica 6", min_value=0.0, value=300000.0, format="%.2f")
    with col3:
        metric7 = st.number_input("Métrica 7", min_value=0.0, value=0.0, format="%.2f")
        metric8 = st.number_input("Métrica 8", min_value=0.0, value=0.0, format="%.2f")
        metric9 = st.number_input("Métrica 9", min_value=0.0, value=5.0, format="%.2f")

    # --- 5. Botón para realizar la predicción ---
    if st.button("Predecir Falla"):
        if pipeline: # Solo procede si el pipeline se cargó correctamente
            # Crear un DataFrame con los datos de entrada del usuario
            # ¡Es CRÍTICO que el orden y los nombres de las columnas coincidan con los usados en el entrenamiento!
            input_data = pd.DataFrame([[
                metric1, metric2, metric3, metric4, metric5,
                metric6, metric7, metric8, metric9
            ]], columns=[f'metric{i}' for i in range(1, 10)])

            st.write("---")
            st.subheader("Resultados de la Predicción:")

            # Realizar la predicción
            prediction = pipeline.predict(input_data)[0] # [0] para obtener el valor único
            # Obtener la probabilidad de la clase positiva (falla = 1)
            prediction_proba = pipeline.predict_proba(input_data)[:, 1][0]

            if prediction == 1:
                st.error(f"⚠️ **¡ALERTA! Posible Falla Detectada.**")
                st.markdown(f"La probabilidad de falla es del **{prediction_proba:.2%}**.")
                st.warning("Se recomienda realizar una inspección o mantenimiento preventivo pronto.")
            else:
                st.success(f"✅ **No se predice Falla.**")
                st.markdown(f"La probabilidad de falla es del **{prediction_proba:.2%}**.")
                st.info("El dispositivo parece estar funcionando normalmente. Continúe monitoreando.")

            st.markdown("---")
            st.markdown("Datos ingresados para la predicción:")
            st.dataframe(input_data)
else:
    st.warning("El modelo no se cargó. No se pueden realizar predicciones.")

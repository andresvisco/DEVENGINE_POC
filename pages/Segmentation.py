from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf.json_format import MessageToDict  # Usamos para convertir la respuesta en diccionario
import streamlit as st
import json
import os
from google.oauth2 import service_account
import asyncio

def generate():
    # Leer las credenciales de la variable de entorno
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
        try:
            # Obtener el secreto de Streamlit
            raw_secret = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
            
            # Eliminar los saltos de línea adicionales y asegurarse de que sea una única línea
            raw_secret = raw_secret.replace("\n", "").replace("\r", "")  # Eliminar saltos de línea
    
            # Cargar el JSON del string
            credentials_dict = json.loads(raw_secret)  # Convertir a diccionario
    
            # Guardar el JSON en un archivo temporal
            credentials_path = "/tmp/gcp_credentials.json"
            with open(credentials_path, "w") as f:
                json.dump(credentials_dict, f)  # Escribirlo correctamente
    
            # Configurar la variable de entorno
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    
            # Cargar credenciales en Vertex AI
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
    
            # Inicializar Vertex AI
            aiplatform.init(
                credentials=credentials,
                project="test-interno-trendit",
                location="global"
            )
    
            st.success("Vertex AI inicializado correctamente.")
        except json.JSONDecodeError as e:
            st.error(f"Error al decodificar JSON de credenciales: {e}")
        except Exception as e:
            st.error(f"Error inesperado: {e}")
    else:
        st.error("No se encontró GOOGLE_APPLICATION_CREDENTIALS_JSON en Streamlit Secrets.")
    
    # Crear un bucle de eventos si no existe uno
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Agregar el título
    st.title("DEVENGINE - PROTOTYPE Summ.")

    topics_to_extract = st.multiselect(
        "Select topics to extract",
        [
            "the skills or aptitudes of the candidate interviewed",
            "all about the education and knowledge",
            "all about the previous experience",
            "additional personal interests and hobbies",
            "the candidate's expectations and goals",
            "the candidate's availability and flexibility",
            "the candidate's salary expectations",
            "the candidate's references",
            "the candidate's contact information"
        ]
    )
    text_other_topic = st.text_input("Add a custom topic")
    if text_other_topic:
        topics_to_extract.append(text_other_topic)
    text1 = f"""You are an expert in profiling and summarizing the transcription of conversations between an interviewer and an interviewee.

    You will receive a text file that you must analyze, and answer about:
                                {". ".join(topics_to_extract)}
    """
    st.success(f"The selected topics {topics_to_extract} will be the focus in the analysis.")
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        document1 = uploaded_file.read().decode("utf-8")
    else:
        st.error("Please upload a file.")
        return

    # Grounding (reemplaza la colección con la correcta de tu entorno)
    grounding_config = {
        "tools": [
            {
                "retrieval": {
                    "vertex_ai_search": {
                        "datastore": "projects/test-interno-trendit/locations/global/collections/default_collection/dataStores/ds-deven_1741511856262"
                    }
                }
            }
        ]
    }

    # Configuración del modelo
    model = "projects/test-interno-trendit/locations/us-central1/models/gemini-2.0-flash-001"
    instances = [{"content": text1 + "\n\n" + document1}]
    parameters = {"temperature": 0.1, "top_p": 0.95, "max_output_tokens": 8192}

    # Crear la solicitud para predecir
    prediction_request = {
        "instances": instances,
        "parameters": parameters
    }

    # Realizar la predicción
    result = ""
    try:
        prediction_response = aiplatform.gapic.PredictionServiceClient().predict(
            instances=instances,
            parameters=parameters
        )
        for response in prediction_response.predictions:
            result += response["content"]
            st.write(response["content"])
    except Exception as e:
        st.error(f"Prediction request failed: {e}")
        return

    # Guardar el resultado en un archivo JSON local
    candidate_name = st.text_input("Enter the candidate's name")
    if st.button("Save Result") and candidate_name:
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        file_name = os.path.join(results_dir, f"{candidate_name}.json")
        with open(file_name, "w") as f:
            json.dump({"result": result}, f)
        st.success(f"Result saved as {file_name}")

        # Proporcionar un botón de descarga
        with open(file_name, "r") as f:
            st.download_button(
                label="Download JSON",
                data=f,
                file_name=f"{candidate_name}.json",
                mime="application/json"
            )

generate()

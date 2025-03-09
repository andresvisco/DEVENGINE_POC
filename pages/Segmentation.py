import streamlit as st
import json
import os
from google.oauth2 import service_account
from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1.types import (
    Tool, Retrieval, VertexAISearch, GenerateContentRequest, Content, Part
)

def generate():
    # Leer las credenciales de la variable de entorno
    credentials_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    if not credentials_json:
        st.error("Google Cloud credentials not found. Please set the GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable.")
        return

    credentials_dict = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)

    aiplatform.init(credentials=credentials, project="test-interno-trendit", location="us-central1")

    st.title("DEVENGINE - PROTOTYPE Summ.")

    # Selección de tópicos de análisis
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

    st.success(f"The selected topics {topics_to_extract} will be the focus in the analysis.")

    # Subir archivo de transcripción
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        document_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("Please upload a file.")
        return

    # Instrucción del sistema
    system_instruction = """You are an expert in profiling and summarizing transcription of conversations between an interviewer and an interviewee."""

    # Prompt para la IA
    text1 = f"""
    You will receive a text file that you must analyze, and answer about:
    {". ".join(topics_to_extract)}
    """

    # Definir herramientas de Grounding (Vertex AI Search)
    tools = [
        Tool(
            retrieval=Retrieval(
                vertex_ai_search=VertexAISearch(
                    datastore="projects/test-interno-trendit/locations/global/collections/default_collection/dataStores/ds-deven_1741511856262"
                )
            )
        ),
    ]

    # Definir la configuración de generación
    request = GenerateContentRequest(
        model="gemini-2.0-flash-001",
        contents=[
            Content(
                role="user",
                parts=[Part.from_text(text1 + "\n\n" + document_text)]
            )
        ],
        system_instruction=[Part.from_text(system_instruction)],
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=8192,
        tools=tools
    )

    # Ejecutar la generación
    client = aiplatform.gapic.PredictionServiceClient()
    response = client.predict(
        endpoint=f"projects/test-interno-trendit/locations/us-central1/publishers/google/models/gemini-2.0-flash-001",
        instances=[{"content": text1 + "\n\n" + document_text}],
        parameters={"temperature": 0.1, "top_p": 0.95, "max_output_tokens": 8192},
    )

    result = response.predictions[0]["content"]
    st.write(result)

    # Guardar resultado
    candidate_name = st.text_input("Enter the candidate's name")
    if st.button("Save Result") and candidate_name:
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        file_name = os.path.join(results_dir, f"{candidate_name}.json")
        with open(file_name, "w") as f:
            json.dump({"result": result}, f)
        st.success(f"Result saved as {file_name}")

        # Botón de descarga
        with open(file_name, "r") as f:
            st.download_button(
                label="Download JSON",
                data=f,
                file_name=f"{candidate_name}.json",
                mime="application/json"
            )

generate()

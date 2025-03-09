from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1.types import Tool, Retrieval, VertexAISearch, GenerateContentRequest, Content, Part, GenerateContentResponse
import streamlit as st
import asyncio
import json
import os
from google.oauth2 import service_account

def generate():
    # Leer credenciales desde Streamlit Secrets
    credentials_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    if not credentials_json:
        st.error("Google Cloud credentials not found. Please set the GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable.")
        return

    credentials_dict = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)

    # Inicializar Vertex AI
    aiplatform.init(credentials=credentials, project="test-interno-trendit", location="us-central1")

    # Agregar título en Streamlit
    st.title("DEVENGINE - PROTOTYPE Summ.")

    # Selección de temas para análisis
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

    # Agregar tema personalizado
    text_other_topic = st.text_input("Add a custom topic")
    if text_other_topic:
        topics_to_extract.append(text_other_topic)
    
    # Mensaje de éxito
    st.success(f"The selected topics {topics_to_extract} will be the focus in the analysis.")
    
    # Cargar archivo
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        document1 = uploaded_file.read().decode("utf-8")
    else:
        st.error("Please upload a file.")
        return

    # Contexto del sistema
    si_text1 = "You are an expert in profiling and summarizing conversations between an interviewer and an interviewee."

    # Instrucción del usuario con los temas seleccionados
    text1 = f"""You are an expert in profiling and summarizing the transcription of conversations between an interviewer and an interviewee.
    You will receive a text file that you must analyze, and answer about:
    {". ".join(topics_to_extract)}
    """

    # Configurar grounding con Vertex AI Search
    tools = [
    Tool(
        retrieval=Retrieval(
            vertex_ai_search=VertexAISearch(
                datastore="projects/test-interno-trendit/locations/global/collections/default_collection/dataStores/ds-deven_1741511856262"
            )
        )
    ),
]

    # Configuración del modelo con grounding
    model = "gemini-2.0-flash-001"
    client = aiplatform.generation.GenerationServiceClient(credentials=credentials)

    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        tools=tools,  # Agrega el grounding
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text1 + "\n\n" + document1)
            ]
        )
    ]

    # Llamada al modelo con grounding
    response = client.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # Extraer respuesta
    result = "".join([chunk.text for chunk in response if chunk.candidates and chunk.candidates[0].content.parts])

    # Mostrar resultado en Streamlit
    st.write(result)

    # Guardar en JSON si se ingresa el nombre del candidato
    candidate_name = st.text_input("Enter the candidate's name")
    if st.button("Save Result") and candidate_name:
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        file_name = os.path.join(results_dir, f"{candidate_name}.json")
        with open(file_name, "w") as f:
            json.dump({"result": result}, f)
        st.success(f"Result saved as {file_name}")

        # Proporcionar botón de descarga
        with open(file_name, "r") as f:
            st.download_button(
                label="Download JSON",
                data=f,
                file_name=f"{candidate_name}.json",
                mime="application/json"
            )

generate()

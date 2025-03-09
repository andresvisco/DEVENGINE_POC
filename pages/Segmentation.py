from google import genai
from google.genai import types
import streamlit as st
import asyncio
import json
import os
from google.oauth2 import service_account

def generate():
    # Leer las credenciales de la variable de entorno
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not credentials_json:
        st.error("Google Cloud credentials not found. Please set the GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable.")
        return

    credentials_dict = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)

    # Crear un bucle de eventos si no existe uno
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    client = genai.Client(
        vertexai=True,
        project="test-interno-trendit",
        location="us-central1",
        credentials=credentials
    )

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
    text1 = types.Part.from_text(text=f"""You are an expert in profiling and summarizing the transcription of conversations between an interviewer and an interviewee.

    You will receive a text file that you must analyze, and answer about:
                                {". ".join(topics_to_extract)}
    """)
    st.success(f"The selected topics {topics_to_extract} will be the focus in the analysis.")
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        document1 = types.Part.from_bytes(
            data=uploaded_file.read(),
            mime_type="text/plain",
        )
    else:
        st.error("Please upload a file.")
        return

    si_text1 = """You are an expert in profiling and summarizing transcription of conversations between an interviewer and an interviewee."""

    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                text1,
                document1
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    result = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config
    ):
        result += chunk.text
        st.write(chunk.text)

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

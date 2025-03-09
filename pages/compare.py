import streamlit as st
import json
import os
    
def compare_candidates():
    st.title("Compare Candidates")

    # Directorio donde se almacenan los archivos JSON
    results_dir = "results"

    # Crear el directorio si no existe
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Obtener la lista de archivos JSON en el directorio de resultados
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    if not json_files:
        st.warning("No candidates found. Please generate and save results first.")
        return

    # Seleccionar candidatos para comparar
    candidates_to_compare = st.multiselect("Select candidates to compare", json_files)

    if st.button("Compare"):
        if len(candidates_to_compare) < 2:
            st.warning("Please select at least two candidates to compare.")
            return

        comparisons = {}
        for file_name in candidates_to_compare:
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                comparisons[file_name] = data["result"]

        # Mostrar comparaciones
        for candidate, result in comparisons.items():
            st.subheader(candidate)
            st.write(result)

if __name__ == "__main__":
    compare_candidates()

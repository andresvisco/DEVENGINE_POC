import pickle
from langchain_community.vectorstores import FAISS

def cargar_vectorstore(archivo_vectorstore, archivo_embeed):
    """Carga un vectorstore desde un archivo."""
    try:
        with open(archivo_embeed, "rb") as f:
            embeeds = pickle.load(f)

        vectorstore = FAISS.load_local(archivo_vectorstore, embeddings=embeeds, allow_dangerous_deserialization=True)
        return vectorstore
    except FileNotFoundError:
        print(f"Error: El archivo '{archivo_vectorstore}' no se encontró.")
        return None
    except PermissionError:
        print(f"Error: No tienes permiso para acceder al archivo '{archivo_vectorstore}'.")
        return None
    except pickle.UnpicklingError:
        print(f"Error: El archivo '{archivo_vectorstore}' está corrupto o no es un archivo pickle válido.")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None

def buscar_segmentos_similares(consulta, vectorstore, top_k=5):
    """Busca segmentos similares en el vectorstore y devuelve el contenido de texto."""
    resultados = vectorstore.similarity_search(consulta, k=top_k)
    return [res.page_content for res in resultados]

def hacer_pregunta(consulta, vectorstore, top_k=5):
    """Realiza una consulta en el vectorstore e imprime los resultados."""
    resultados = buscar_segmentos_similares(consulta, vectorstore, top_k)

    for i, resultado in enumerate(resultados):
        print(f"Resultado {i+1}:")
        print(resultado)
        print("\n")

if __name__ == "__main__":
    archivo_vectorstore = "vectorstore.pkl"
    archivo_embeed = "embeddings.pkl"
    vectorstore = cargar_vectorstore(archivo_vectorstore, archivo_embeed)

    if vectorstore:
        print("Vectorstore cargado exitosamente.")

        pregunta = "¿Cuáles son las principales habilidades del candidato?"
        hacer_pregunta(pregunta, vectorstore)
    else:
        print("No se pudo cargar el vectorstore.")
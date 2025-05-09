import faiss
import numpy as np
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_own_data_source.google_drive.utils_function import load_json, get_embedding_query



FAISS_INDEX_PATH = "rag_own_data_source/google_drive/faiss_index.faiss"
DOC_MAPPING_PATH = "rag_own_data_source/google_drive/doc_mapping_id_to_text.json"
SIMILARITY_THRESHOLD = 0.7
TOP_K = 4


try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    docs_mapping = load_json(DOC_MAPPING_PATH)
except Exception as e:
    print(f"Erreur lors du chargement des ressources : {e}")
    raise


def search_faiss(query: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD):
    try:
        query_vector = np.array(get_embedding_query(query)).reshape(1, -1)
        distances, indices = index.search(query_vector, top_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            similarity = 1 - distance
            if similarity >= threshold:
                results.append(docs_mapping.get(str(idx), ""))

        return results

    except Exception as e:
        print(f"Erreur pendant la recherche FAISS : {e}")
        return []


def get_context(query: str) -> str:
    context_chunks = search_faiss(query)
    if not context_chunks:
        return "Aucun contexte trouvé"
    return "\n\n".join(context_chunks)


def rag_answer(query: str, llm):
    context = get_context(query)

    template = '''
        Tu es un assistant intelligent conçu pour guider, inspirer et encourager l'utilisateur dans son parcours entrepreneurial. Commence toujours par une citation motivante liée à l’esprit entrepreneurial.
        
        En te basant uniquement sur le contexte suivant, réponds à la question de façon claire, bienveillante et encourageante. Si le lien entre la question et le contexte est insuffisant, dis simplement : "Je ne sais pas".
        
        ### Contexte:
        {context}
        
        ### Question:
        {query}
        
        ### Réponse:
    '''

    prompt_template = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        response = chain.invoke(input={'query': query, 'context': context})
        return response["text"]
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse : {e}")
        return "Une erreur est survenue lors de la génération de la réponse."

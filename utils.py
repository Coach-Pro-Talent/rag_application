import faiss
import numpy as np
from langchain_core.prompts.prompt import PromptTemplate
from rag_own_data_source.google_drive.utils_function import load_json, get_embedding_query
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


index = faiss.read_index('rag_own_data_source/google_drive/faiss_index.faiss')
docs_mapping = load_json("rag_own_data_source/google_drive/doc_mapping_id_to_text.json")


def search_faiss(query, top_k=3):
    query_vector =np.array(get_embedding_query(query)).reshape(1, -1)

    D, I = index.search(query_vector, top_k)
    return  [docs_mapping[str(idx)] for idx in I[0]]

def get_context(query):
    context_chunks = search_faiss(query)
    context_="\n\n".join(context_chunks)

    if not str(context_):
        return "Aucun contexte trouvé"
    return context_

def rag_answer(query, llm):
   context = get_context(query)


   template = '''
   Tu es un assistant intelligent très poli qui aide l'utilisateur en lui donnant 
   des reponses impactantes 
   qui l'encourage et l'édifie en te basant.
   En te basant uniquement sur le contexte ci-dessous , réponds à la question suivante:
   
   ### Contexte:
   {context}
   
   ### Question:
   {query}
   
   
   ### Réponse:
   
   
   '''
   prompt_template = PromptTemplate.from_template(template)
   chain = LLMChain(llm=llm, prompt=prompt_template)

   response = chain.invoke(input={'query':query, 'context':context})

   return response["text"]





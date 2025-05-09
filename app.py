from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from utils import rag_answer
import streamlit as st
import pyttsx3
from rag_own_data_source.google_drive.rag_faiss_drive import start_index_documents
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os


load_dotenv()
api_key_env = os.environ.get("OPENAI_API_KEY")
elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
st.set_page_config(page_title="Coach Pro Talent", layout="centered")

client = ElevenLabs(api_key=elevenlabs_api_key)

with st.sidebar:
    st.header("🔧 Configuration")
    api_key_input = st.text_input("🔑 Entrez votre API Key OpenAI", type="password")
    if api_key_input:
        st.success("✅ Clé API enregistrée.")
        api_key = api_key_input
    else:
        api_key = api_key_env

    st.markdown("---")
    if st.button("🔄 Mettre à jour"):
        st.info("Fonctionnalité de mise à jour non encore implémentée.")

if not api_key:
    st.warning("Veuillez fournir une clé API OpenAI dans la barre latérale.")
    st.stop()

# Initialisation de llm
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=api_key,
    max_tokens=200
)

def speak(text):

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)




tab_selection = st.radio("Sélectionnez une section", ["Chat", "À propos de moi", "Update Data"])


if tab_selection == "Chat":
    st.title("🤖 Coach & Pro Talent Bot")
    st.subheader("🚀 Build your Future with Us")
    st.markdown("---")

    st.markdown("### 💬 Coach Pro Talent Assistant")

    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

    input_text = st.chat_input("Posez une question...")

    if input_text:
        with st.spinner("🔎 Recherche en cours..."):
            result = rag_answer(input_text, llm)
            st.session_state.last_response = result
            st.success("✅ Réponse générée :")
            st.markdown(result)

    if st.session_state.last_response:
        st.markdown("### 🧠 Dernière réponse")
        st.markdown(st.session_state.last_response)

        if st.button("🔊 Lire la réponse à voix haute"):
            speak(st.session_state.last_response)


elif tab_selection == "À propos de moi":
    st.title("👨‍💻 À propos de moi")
    st.markdown("---")
    st.markdown("""
        Bonjour ! Je suis un passionné de **Data Engineering** et d'**IA**. 
        Actuellement, je travaille sur divers projets utilisant **OpenAI**, **Streamlit**, et d'autres technologies pour améliorer les solutions technologiques. 
        Mon objectif est de créer des outils intelligents pour résoudre des problèmes complexes.
        
        🌐 **Compétences** :
        - Data Engineering
        - Intelligence Artificielle
        - Cloud Computing
        - Développement d'agents IA

        📈 **Projets en cours** :
        - Optimisation des coûts cloud avec IA
        - Automatisation des services publics en Afrique
        - Développement de solutions d'analyses de données en temps réel
    """)
elif tab_selection == "Update Data":
    st.title("🔄 Mise à jour des données")

    if "logs_update" not in st.session_state:
        st.session_state["logs_update"] = ""

    st.text_area("Logs", value=st.session_state["logs_update"], height=400, disabled=True, key="logs_update_display")

    if st.button("Mettre à jour les données"):
        with st.spinner("🔄 Exécution de l'indexation..."):
            logs = start_index_documents()
            st.session_state["logs_update"] = "\n".join(logs)

        st.success("✅ Mise à jour terminée !")
        st.rerun()
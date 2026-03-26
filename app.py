import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDF")

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = llm
Settings.embed_model = embed_model

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

faiss_index = faiss.IndexFlatL2(384)

vector_store = FaissVectorStore(faiss_index=faiss_index)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    docs = SimpleDirectoryReader("data").load_data()

    st.info("Processing document...")

    st.session_state.index = VectorStoreIndex.from_documents(
        docs,
        vector_store=vector_store
    )

    st.success("Index created!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.button("Clear Chat"):
    st.session_state.messages = []

query = st.chat_input("Ask something about the PDF...")

if query and st.session_state.index:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Query engine
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=3
    )

    # generate answers
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(query)
            answer = str(response)

            st.write(answer)

            # sources
            with st.expander("Sources"):
                for node in response.source_nodes:
                    st.write(node.text[:200] + "...")

    # save response
    st.session_state.messages.append({"role": "assistant", "content": answer})

if not st.session_state.index:
    st.info("Upload a PDF to start chatting")
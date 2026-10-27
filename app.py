import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader,
                              StorageContext, load_index_from_storage, Settings)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv(override=True)

MAX_QUESTIONS = 10   # per visitor, protects your budget
STORAGE = "storage"

SYSTEM_PROMPT = """You answer questions about a set of research papers.
Rules:
- Use ONLY the context excerpts provided. Do not use outside knowledge.
- If the answer is not clearly in the excerpts, say: "I couldn't find that in the papers."
- Be precise with numbers, dataset names and results, and say which paper they come from.
- Distinguish between data the study collected/used for its experiments and
  datasets only mentioned in passing (e.g. used to pre-train a model)."""

# ---------- Page setup ----------
st.set_page_config(page_title="Ask my PhD research", page_icon="📄")
st.title("📄 Ask my research")
st.caption("Ask questions about my published papers across my BSc, MSc and PhD. "
           "Answers come only from these papers, with page citations. "
           "PhD papers currently under review are not yet included.")
st.subheader("Papers included")
st.markdown("""
- **PhD:** Image-based classification of robotic mastication (J. Food Engineering)
- **PhD:** ChewNet multimodal food-bolus dataset (Data in Brief)
- **PhD:** A novel approach for robotic bolus formation using GLCM image analysis
- **PhD:** Moisture estimation of boluses by image processing
- **PhD:** Real-time deep learning framework for chewing-cycle estimation 
- **MSc:** Automatic colourisation of greyscale images using deep learning
- **BSc:** Modelling and Optimisation of Vision-Guided Pick-and-Place Tasks Using Parallel Manipulators
""")

# ---------- Load models and index (only once, then cached) ----------
@st.cache_resource(show_spinner="Loading papers...")
def get_index():
    azure = dict(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    Settings.llm = AzureOpenAI(engine=os.getenv("CHAT_DEPLOYMENT"),
                               model=os.getenv("CHAT_DEPLOYMENT"),
                               temperature=0, **azure)
    Settings.embed_model = AzureOpenAIEmbedding(deployment_name=os.getenv("EMBED_DEPLOYMENT"),
                                                model=os.getenv("EMBED_DEPLOYMENT"), **azure)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    if os.path.exists(STORAGE):
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=STORAGE))
    docs = SimpleDirectoryReader("Data", file_extractor={".pdf": PDFReader()}).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=STORAGE)
    return index

index = get_index()

# ---------- Each visitor gets their own conversation ----------
if "engine" not in st.session_state:
    st.session_state.engine = index.as_chat_engine(
        chat_mode="condense_plus_context", similarity_top_k=8, system_prompt=SYSTEM_PROMPT)
    st.session_state.messages = []
    st.session_state.count = 0

with st.sidebar:
    st.subheader("About")
    st.write("A RAG chatbot built with LlamaIndex and Azure OpenAI.")
    st.write(f"Questions used: {st.session_state.count} / {MAX_QUESTIONS}")
    if st.button("Start a new conversation"):
        st.session_state.engine.reset()
        st.session_state.messages = []
        st.rerun()

# Show the conversation so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Handle a new question ----------
if question := st.chat_input("Ask about my research..."):
    if st.session_state.count >= MAX_QUESTIONS:
        st.warning("You've reached the question limit for this session. Thanks for trying it!")
    else:
        st.session_state.count += 1
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the papers..."):
                response = st.session_state.engine.chat(question)

            # Unique paper + page list for citations
            seen, sources = set(), []
            for node in response.source_nodes:
                key = (node.metadata.get("file_name"), node.metadata.get("page_label"))
                if key not in seen:
                    seen.add(key)
                    sources.append(f"- {key[0]}, page {key[1]}")

            answer = str(response)
            st.markdown(answer)
            with st.expander("Sources"):
                st.markdown("\n".join(sources))

        st.session_state.messages.append({"role": "assistant", "content": answer})
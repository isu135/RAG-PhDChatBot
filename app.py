import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader,
                              StorageContext, load_index_from_storage, Settings)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import uuid
import gspread
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from llama_index.core import Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

load_dotenv(override=True)

MAX_QUESTIONS = 10   # per visitor, protects your budget
FIG_MIN_SCORE = 0.5   # only show a figure if its match score is at least this
STORAGE = "storage"

SYSTEM_PROMPT = """You answer questions about a set of research papers.
Rules:
- Use ONLY the context excerpts provided. Do not use outside knowledge.
- If the answer is not clearly in the excerpts, say: "I couldn't find that in the papers."
- Be precise with numbers, dataset names and results, and say which paper they come from.
- Distinguish between data the study collected/used for its experiments and
  datasets only mentioned in passing (e.g. used to pre-train a model).
- Some excerpts describe figures. The relevant figure images are shown automatically
  below your answer, so never offer to show them; just mention the figure and its page."""

# ---------- Page setup ----------
st.set_page_config(page_title="Ask about my PhD research", page_icon="📄")
st.title("📄 Ask about my research")
st.caption("Ask questions about my published papers across my BSc, MSc and PhD. "
           "Answers come only from these papers, with page citations. "
           "PhD papers currently under review are not yet included.")
st.subheader("Papers included")
st.markdown("""
- **PhD · ChewBot:** Image-based Classification of Bolus Phases in Robotic Mastication using Hybrid Machine Learning Analysis
- **PhD · ChewBot:** ChewNet: A Multimodal Dataset for Invivo and Invitro Beef and Plant-Based Burger Patty Boluses with Images, Texture, and Force Profiles 
- **PhD · ChewBot:** A Novel Approach to Evaluate Robotic in Vitro Chewing Effect on Food Bolus Formation Using the GLCM Image Analysis Technique 
- **PhD · ChewBot:** Estimation of the Moisture Content of In Vitro Chewed Food Boluses by Image Processing 
- **PhD · ChewBot:** A Conceptual Framework for a Real-Time Deep Learning Model to Estimate In vitro Chewing for Safe Swallowing 
- **MSc:** Automatic Colorization of Greyscale Images Using Deep Learning Fusion Technique and Accuracy Level Analysis 
- **BSc · Delta robot:** Optimization of Regular Pick and Place Tasks Via Parallel Manipulators and Modelling
""")

FIG_JSON = "figures/figures.json"

def load_figure_docs():
    """Turn each described figure into a searchable document."""
    if not os.path.exists(FIG_JSON):
        return []
    with open(FIG_JSON, encoding="utf-8") as f:
        figs = json.load(f)
    docs = []
    for fig in figs:
        if not fig.get("description") or not os.path.exists(fig["image_path"]):
            continue
        hidden = ["type", "image_path", "caption", "boost"]
        docs.append(Document(
            text=(f"Figure on page {fig['page']} of {fig['file_name']}. "
                  f"Caption: {fig['caption'] or 'none'}. "
                  f"Description: {fig['description']}"),
            metadata={"file_name": fig["file_name"], "page_label": str(fig["page"]),
                      "type": "figure", "image_path": fig["image_path"],
                      "caption": fig["caption"], "boost": float(fig.get("boost", 0))},
            excluded_embed_metadata_keys=hidden,
            excluded_llm_metadata_keys=hidden,
        ))
    return docs

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
    docs += load_figure_docs()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=STORAGE)
    return index

index = get_index()

# A separate search over figures only
fig_retriever = index.as_retriever(
    similarity_top_k=6,
    filters=MetadataFilters(filters=[ExactMatchFilter(key="type", value="figure")]),
)

# ---------- Question log (Google Sheet) ----------
@st.cache_resource
def get_log_sheet():
    try:
        gc = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
        return gc.open_by_key(st.secrets["SHEET_ID"]).sheet1
    except Exception:
        return None   # logging is simply off if secrets are missing (e.g. locally)

def log_question(question, answer, sources):
    sheet = get_log_sheet()
    if sheet is None:
        return
    try:
        now = datetime.now(ZoneInfo("Pacific/Auckland")).strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row(
            [now, st.session_state.session_id, question, answer[:2000], "; ".join(sources)],
            value_input_option="RAW",
        )
    except Exception:
        pass          # never let a logging problem break the chatbot

# ---------- Each visitor gets their own conversation ----------
if "engine" not in st.session_state:
    st.session_state.engine = index.as_chat_engine(
        chat_mode="condense_plus_context", similarity_top_k=8, system_prompt=SYSTEM_PROMPT)
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []
    st.session_state.count = 0

with st.sidebar:
    st.subheader("About")
    st.write("Hi there! This is a RAG chatbot built with LlamaIndex and Azure OpenAI - Isurie Akarawita.")
    st.write(f"Questions used: {st.session_state.count} / {MAX_QUESTIONS}")
    if st.button("Start a new conversation"):
        st.session_state.engine.reset()
        st.session_state.messages = []
        st.rerun()

# Show the conversation so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for fig in m.get("figures", []):
            st.image(fig["path"], caption=fig["caption"])

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

            # Show up to 3 relevant figures, preferring boosted ones
            candidates = [n for n in fig_retriever.retrieve(question)
                          if (n.score or 0) >= FIG_MIN_SCORE]
            candidates.sort(key=lambda n: (n.score or 0) + n.metadata.get("boost", 0),
                            reverse=True)
            figures = []
            for node in candidates:
                m = node.metadata
                if m["image_path"] not in [f["path"] for f in figures]:
                    figures.append({"path": m["image_path"],
                                    "caption": f"{m.get('caption') or 'Figure'} "
                                               f"({m['file_name']}, page {m['page_label']})"})
            figures = figures[:3]
            for fig in figures:
                st.image(fig["path"], caption=fig["caption"])

            with st.expander("Sources"):
                st.markdown("\n".join(sources))

        st.session_state.messages.append({"role": "assistant", "content": answer, "figures": figures})
        log_question(question, answer, sources)
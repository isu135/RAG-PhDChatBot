import os
from dotenv import load_dotenv
from llama_index.core import (VectorStoreIndex, SimpleDirectoryReader,
                              StorageContext, load_index_from_storage, Settings)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv(override=True)

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

# Smaller, overlapping chunks = more precise search
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

STORAGE = "storage"

if os.path.exists(STORAGE):
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=STORAGE))
    print("Loaded saved index.")
else:
    docs = SimpleDirectoryReader("Data", file_extractor={".pdf": PDFReader()}).load_data()
    print(f"Read {len(docs)} pages from your papers.")
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=STORAGE)
    print("Index built and saved.")

SYSTEM_PROMPT = """You answer questions about a set of research papers.
Rules:
- Use ONLY the context excerpts provided. Do not use outside knowledge.
- If the answer is not clearly in the excerpts, say: "I couldn't find that in the papers."
- Be precise with numbers, dataset names and results, and say which paper they come from.
- Distinguish between data the study collected/used for its experiments and
  datasets only mentioned in passing (e.g. used to pre-train a model)."""

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",   # remembers the conversation
    similarity_top_k=8,                  # look at more chunks
    system_prompt=SYSTEM_PROMPT,
)

print("\nType your question. 'reset' clears the conversation, 'exit' quits.")
while True:
    q = input("\nYou: ").strip()
    if q.lower() in ("exit", "quit"):
        break
    if q.lower() == "reset":
        chat_engine.reset()
        print("(conversation cleared)")
        continue

    response = chat_engine.chat(q)
    print("\nBot:", response)
    print("\nSources:")
    for node in response.source_nodes:
        snippet = node.get_content()[:120].replace("\n", " ")
        print(f"  - {node.metadata.get('file_name')}, page {node.metadata.get('page_label')} "
              f"(match {node.score:.2f})\n      \"{snippet}...\"")
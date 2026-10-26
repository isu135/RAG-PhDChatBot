import os
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv()  # reads your .env file

llm = AzureOpenAI(
    engine=os.getenv("CHAT_DEPLOYMENT"),
    model=os.getenv("CHAT_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

embed = AzureOpenAIEmbedding(
    deployment_name=os.getenv("EMBED_DEPLOYMENT"),
    model=os.getenv("EMBED_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

print("Chat test:", llm.complete("Say hello in one short sentence."))
print("Embedding test: got", len(embed.get_text_embedding("hello")), "numbers")
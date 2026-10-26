# PhD Papers RAG Chatbot

Ask questions about a collection of research papers and get answers with
paper and page citations. Built with LlamaIndex and Azure OpenAI.

## Setup
1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your Azure details
4. Put your PDFs in a folder called `Data/`
5. `python test_connection.py` to check the connection
6. `python ask_papers.py` to start asking questions
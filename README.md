# know-me-better

Get to know me by asking a chatbot with access to my files.

## Architecture

### Components

- Streamlit for UI
- Supabase Postgres Vector Database for Backend DB
- DeepInfra/Together API for access to LLMs
- LangChain for Data Ingestion and RAG Chains


### Data Ingestion
- Read all repos under a GitHub owner
- Embedding models via DeepInfra
- Ingest into Supabase Postgres instance


## Directory structure
```
know_me_better
    |--- docs
    |--- ingestion
    |--- rag
    |--- streamlit

```

## Setup
- Using poetry for dependency management 
- Python version: 3.10.13
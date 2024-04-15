import os

from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from supabase.client import Client, create_client


def rag(query, vector_store):
    retriever = vector_store.as_retriever(search_type="mmr")
    
    matched_docs = vector_store.similarity_search_with_relevance_scores(query)
    # matched_docs = retriever.get_relevant_documents(query)

    for i, d in enumerate(matched_docs):
        print(f"\n## Document {i}\n")
        print(d.page_content)

def main(query):
    load_dotenv()
    embedding_model = "all-mpnet-base-v2"

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

    rag(query, vector_store)

if __name__ == "__main__":
    query = "What is the meaning of life?"
    main(query)
import os

from dotenv import load_dotenv

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import DeepInfra
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from supabase.client import Client, create_client


def initialize_RAG(embedding_model="all-mpnet-base-v2", 
                   model_name="databricks/dbrx-instruct"):

    # Prompt template
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

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
    llm = DeepInfra(model_id=model_name)
    llm.model_kwargs = {
        "temperature": 0.1,
        "repetition_penalty": 1.2,
        "max_new_tokens": 250,
        "top_p": 0.9,
    }
    
    return prompt, vector_store, llm


def rag(query, prompt, vector_store, llm, debug=False):
    retriever = vector_store.as_retriever()
    
    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # matched_docs = vector_store.similarity_search_with_relevance_scores(query)
    matched_docs = retriever.get_relevant_documents(query)

    if debug:
        for i, d in enumerate(matched_docs):
            print(f"\n## Document {i}\n")
            print(d.page_content)

    return chain.invoke(query)


def main(query):
    load_dotenv()
    
    prompt, vector_store, llm = initialize_RAG()
    rag(query, prompt=prompt, vector_store=vector_store, llm=llm)


if __name__ == "__main__":
    query = "What is bayesian optimization?"
    main(query)
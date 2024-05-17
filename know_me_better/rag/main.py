import os

from dotenv import load_dotenv

from langchain_community.vectorstores import SupabaseVectorStore
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.llms import DeepInfra
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

from supabase.client import Client, create_client


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_rag(embedding_model="BAAI/bge-base-en-v1.5"):

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    embeddings = DeepInfraEmbeddings(
        model_id=embedding_model,
        query_instruction="",
        embed_instruction="",
        )

    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

    return prompt, vector_store


def rag(query, prompt, vector_store, model, debug=False):
    llm = DeepInfra(model_id=model)
    llm.model_kwargs = {
        "temperature": 0.1,
        "repetition_penalty": 1.2,
        "max_new_tokens": 250,
        "top_p": 0.9,
    }
    retriever = vector_store.as_retriever()

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # matched_docs = vector_store.similarity_search_with_relevance_scores(query)

    if debug:
        matched_docs = retriever.get_relevant_documents(query)
        for i, d in enumerate(matched_docs):
            print(f"\n## Document {i}\n")
            print(d.page_content)

    return chain.invoke(query)


if __name__ == "__main__":
    load_dotenv()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "databricks/dbrx-instruct"

    queries = ["What is bayesian optimization?",
               "Where did Jagan do his masters?",
               "Where did he work previously?",
               "Where is he working currently",
               "How many languages does he know?"]

    prompt, vector_store = initialize_rag()

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vector_store.as_retriever(),
    #     return_source_documents=True,
    # )

    for query in queries:
        response = rag(query=query, prompt=prompt, vector_store=vector_store, model=model_name)
        print(response)

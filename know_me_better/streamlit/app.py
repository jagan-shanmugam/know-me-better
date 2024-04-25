from know_me_better.rag.main import initialize_rag, rag

from dotenv import load_dotenv

import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun


def main():
    choices = ["databricks/dbrx-instruct", ]
    with st.sidebar:
        "[Read my CV here](https://platform.openai.com/account/api-keys)"
        "Contact me: jaganshanmugam@outlook.com"

        model = st.selectbox("Model", choices)

    st.title("🔎 Get to know me better")

    prompt, vector_store = initialize_rag()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hey! I'm a chatbot who can help you know Jagan better. "
                                             "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input(placeholder="Where did Jagan study his Masters?"):

        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        search = DuckDuckGoSearchRun(name="Search")
        # search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #                                 handle_parsing_errors=True)

        with (st.chat_message("assistant")):
            # st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            with st.spinner(text="Processing.."):
                response = rag(query=query, prompt=prompt, vector_store=vector_store, model=model)
            # search_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)


if __name__ == "__main__":
    load_dotenv()
    main()

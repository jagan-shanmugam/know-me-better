import os

from dotenv import load_dotenv


def test_llm_openai(base_url="https://api.deepinfra.com/v1/openai", 
                    model_name="databricks/dbrx-instruct"):
    # Assume openai>=1.0.0
    from openai import OpenAI

    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=os.getenv("DEEPINFRA_API_TOKEN"),
        base_url=base_url,
    )

    chat_completion = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello"}],
    )

    print(chat_completion.choices[0].message.content)


def test_llm_langchain(model_name="databricks/dbrx-instruct"):
    from langchain_community.llms import DeepInfra

    llm = DeepInfra(model_id=model_name)
    llm.model_kwargs = {
        "temperature": 0.1,
        "repetition_penalty": 1.2,
        "max_new_tokens": 250,
        "top_p": 0.9,
    }
    # run streaming inference
    for chunk in llm.stream("Who let the dogs out?"):
        print(chunk)


if __name__ == "__main__":
    load_dotenv()
    test_llm_openai()
    test_llm_langchain()

import os
from langchain_openai import ChatOpenAI

LLM_MODEL_NAME = "gpt-5"


def get_llm_model_name() -> str:
    return LLM_MODEL_NAME

def get_llm():
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=os.environ.get("DocuSun_GITHUB_TOKEN"), 
        base_url="https://models.inference.ai.azure.com",
        temperature=0
    )
    return llm 
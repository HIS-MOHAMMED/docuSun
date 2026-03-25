import os
from langchain_openai import ChatOpenAI

def get_llm():
    llm = ChatOpenAI(
        model="gpt-4o", 
        api_key=os.environ.get("GITHUB_TOKEN"), 
        base_url="https://models.inference.ai.azure.com",
        temperature=0
    )
    return llm
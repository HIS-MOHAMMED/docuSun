
from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    prompt_template = ChatPromptTemplate.from_template(
        (
            "Please answer the following question based on the provided `context` that follows the question.\n"
            "If you do not know the answer then just say 'I do not know'\n"
            "question: {question}\n"
            "context: ```{context}```\n"
        )
    )
    return prompt_template
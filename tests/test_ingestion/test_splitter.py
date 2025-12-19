from unittest.mock import patch, MagicMock
from engine.ingestion.splitter import split_documents
from langchain.docstore.document import Document

EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
def test_split_documents():
    # Arrange: create fake document to be split by splitter
    fake_document = Document(page_content ="""

### Chapter 6

WHAT SORT OF DESPOTISM DEMOCRATIC NATIONS HAVE TO FEAR

I had remarked during my stay in the United States that a democratic state of society, similar to that of the Americans, might offer singular facilities for the establishment of despotism; and I perceived, upon my return to Europe, how much use had already been made, by most of our rulers, of the notions, the sentiments, and the wants created by this same social condition, for the purpose of extending the circle of their power. This led me to think that the nations of Christendom would perhaps eventually undergo some oppression like that which hung over several of the nations of the ancient world.
A more accurate examination of the subject, and five years of further meditation, have not diminished my fears, but have changed their object.
No sovereign ever lived in former ages so absolute or so powerful as to undertake to administer by his own agency, and without the assistance of intermediate powers, all the parts of a great empire; none ever attempted to subject all his subjects indiscriminately to strict uniformity of regulation and personally to tutor and direct every member of the community. The notion of such an undertaking never occurred to the human mind; and if any man had conceived it, the want of information, the imperfection of the administrative system, and, above all, the natural obstacles caused by the inequality of conditions would speedily have checked the execution of so vast a design.

---

### Challenges of agent systems

Generally, the difficult parts of running an agent system for the LLM engine are:

1. From supplied tools, choose the one that will help advance to a desired goal: e.g. when asked `"What is the smallest prime number greater than 30,000?"`, the agent could call the `Search` tool with `"What is he height of K2"` but it won't help.
2. Call tools with a rigorous argument formatting: for instance when trying to calculate the speed of a car that went 3 km in 10 minutes, you have to call tool `Calculator` to divide `distance` by `time` : even if your Calculator tool accepts calls in the JSON format: `{”tool”: “Calculator”, “args”: “3km/10min”}` , there are many pitfalls, for instance:
    - Misspelling the tool name: `“calculator”` or `“Compute”` wouldn’t work
    - Giving the name of the arguments instead of their values: `“args”: “distance/time”`
    - Non-standardized formatting: `“args": "3km in 10minutes”`
3. Efficiently ingesting and using the information gathered in the past observations, be it the initial context or the observations returned after using tool uses.


So, how would a complete Agent setup look like?

## Running agents with LangChain

We have just integrated a `ChatHuggingFace` wrapper that lets you create agents based on open-source models in [🦜🔗LangChain](https://www.langchain.com/).

The code to create the ChatModel and give it tools is really simple, you can check it all in the [Langchain doc](https://python.langchain.com/docs/integrations/chat/huggingface). 

```python
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

chat_model = ChatHuggingFace(llm=llm)
""", 
    metadata = {"source":"test.pdf", "page":1})

    # Act: pass the fake document to the split_documents function to split into list of documents(chunks)
    chunks = split_documents(200, [fake_document],EMBEDDING_MODEL_NAME)

    # Assert: check the length of returned list, its content and metadata of some of them
    assert len(chunks) == 8
    assert chunks[3].metadata['source'] == 'test.pdf'
    assert chunks[7].page_content == 'chat_model = ChatHuggingFace(llm=llm)'
    assert chunks[0].metadata['page'] == 1
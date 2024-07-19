from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader

from dotenv import load_dotenv

load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

docs = get_documents_from_web("https://www.spiceworks.com/tech/artificial-intelligence/news/openai-launches-cost-effective-gpt-4o-mini-and-enhanced-integrations-for-enterprise-customers/")


model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
Answer User Question :
Context : {context}
User : {input}                                          
""")

# chain = prompt | model
chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt
)

response = chain.invoke({
    "input":"How cheap is gpt-4o-mini compared to gpt-3.5-turbo",
    "context":docs
})


print(response)
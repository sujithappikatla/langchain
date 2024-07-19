from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv

load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20 
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorStore

def create_chain(vectorStore):
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

    retriver = vectorStore.as_retriever(search_kwargs={"k":1})

    retriver_chain = create_retrieval_chain(
        retriver,
        chain
    )

    return retriver_chain



docs = get_documents_from_web("https://www.spiceworks.com/tech/artificial-intelligence/news/openai-launches-cost-effective-gpt-4o-mini-and-enhanced-integrations-for-enterprise-customers/")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input":"How cheap is gpt-4o-mini compared to gpt-3.5-turbo"
})


print(len(response["context"]))
print(response["answer"])
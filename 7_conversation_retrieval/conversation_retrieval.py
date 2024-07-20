from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from dotenv import load_dotenv

load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200 
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user context based on context : {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriver = vectorStore.as_retriever(search_kwargs={"k":4})
    
    retriver_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given above conversation history, generate query for relevant info lookup ")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriver,
        prompt=retriver_prompt
    )

    retriver_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retriver_chain


def ask_model(chain, query, chat_history):
    response = chain.invoke({
        "input":query,
        "chat_history":chat_history
    })
    print(f"AI : {response['answer']}")
    return response["answer"]


if __name__ == '__main__':
    docs = get_documents_from_web("https://www.spiceworks.com/tech/artificial-intelligence/news/openai-launches-cost-effective-gpt-4o-mini-and-enhanced-integrations-for-enterprise-customers/")
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)
    
    chat_history = []
    
    user_input = input("You : ")
    while user_input!="bye":
        output = ask_model(chain,user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        user_input = input("You : ")

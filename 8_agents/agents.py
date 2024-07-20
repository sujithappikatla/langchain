from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool


load_dotenv()


loader = WebBaseLoader("https://www.spiceworks.com/tech/artificial-intelligence/news/openai-launches-cost-effective-gpt-4o-mini-and-enhanced-integrations-for-enterprise-customers/")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200 
)
splitDocs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(splitDocs, embedding=embeddings)

retriever = vectorStore.as_retriever(search_kwargs={"k":3})


model = ChatOpenAI(
    model="gpt-4o-mini"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are friendly assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search= TavilySearchResults()
retriever_tools = create_retriever_tool(
    retriever,
    "gpt-4o-mini",
    "Use this tool for any information regarding new Openai model gpt-40-mini"
)

tools = [search, retriever_tools]

agent = create_openai_functions_agent(
    llm = model, 
    tools = tools,
    prompt=prompt
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def ask_model(agentExecutor, query, chat_history):
    response = agentExecutor.invoke({
        "input":query,
        "chat_history":chat_history
    })

    return response["output"]


if __name__ == '__main__':
    chat_history = []
    
    user_input = input("You : ")
    while user_input!="bye":
        output = ask_model(agentExecutor, user_input, chat_history)
        print(f"AI : {output}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        user_input = input("You : ")

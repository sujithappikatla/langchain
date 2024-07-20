from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are friendly AI Assitant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# chain = prompt | model
chain = LLMChain(
    llm = model,
    prompt = prompt,
    memory = memory,
    verbose = True
)


msg = {
    "input":"my name is john"
}

response = chain.invoke(msg)
print(response)


msg2 = {
    "input":"hello"
}

response = chain.invoke(msg2)
print(response)

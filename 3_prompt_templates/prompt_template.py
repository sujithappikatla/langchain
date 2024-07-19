from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# Prompt Template
prompt = ChatPromptTemplate.from_template("Tell me joke about {subject}")

# Create chain
chain = prompt | model

response = chain.invoke({"subject":"elon musk"})
print(response)

# ---------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate 10 sentences using following word"),
        ("human" , "{word}")
    ]
)

chain = prompt | model
response = chain.invoke({"word":"zeal"})

print(f"\n\n {response}")
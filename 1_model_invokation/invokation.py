from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# loads OPENAI_API_KEY from .env file.
# If not use, set api_key of ChatOpenAI function
load_dotenv()


def run(model):
    prompt = "Hi, How are you?"
    response = model.invoke(prompt)
    print(f"Response from Model : \n{response}")
    print(f"Model Output : \n{response.content}")


# Creates Model
model1 = ChatOpenAI(
    temperature=0, # 0-> factual answers, 1-> creative
    model="gpt-4o-mini",
    verbose=True,
    max_tokens=100,
)

model2 = ChatGroq(
    temperature=1,
    model="llama3-8b-8192",
    verbose=True
)

print(f"-----  OPENAI   ----------- \n\n")
run(model1)


print(f"\n-----  GROQ   ----------- \n\n")
run(model2)


print(f"\n----- Batching  ------- \n\n")
response = model1.batch(["hi, how are you?", "what is 2+3?"])
print(f"Response of Batching : \n")
print(f"Response 1 : {response[0]} \n")
print(f"Response 2 : {response[1]}")
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini"
)

response = model.stream("Write 100 word essay on LLM")

for chunk in response:
    # end by default is newline and cause each token to print on new line
    # flush will print out to terminal as data comes in
    print(chunk.content, end="", flush=True)
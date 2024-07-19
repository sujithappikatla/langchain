from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")


def str_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate 10 sentences using following word"),
            ("human" , "{word}")
        ]
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    return chain.invoke({"word":"zeal"})

print(f"\n {str_output_parser()}")
print(f"\n {type(str_output_parser())}") # string


def list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give me 10 breeds of {animal}. Give me output comma seperated"),
            ("human" , "{animal}")
        ]
    )
    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser
    return chain.invoke({"animal":"dog"})


print(f"\n {list_output_parser()}")
print(f"\n {type(list_output_parser())}") # list


def json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give me information about following person, Formatted in json format : {input_format}"),
            ("human" , "{person_name}")
        ]
    )

    class Person(BaseModel):
        name: str = Field(description="Name of person")
        age: int = Field(description="Age of Person")
        occupation : str = Field(description="Occupation of person")
    
    parser = JsonOutputParser(pydantic_object=Person)
    chain = prompt | model | parser
    return chain.invoke({
        "person_name":"Narendra Modi",
        "input_format":parser.get_format_instructions()
    })

print(f"\n {json_output_parser()}")
print(f"\n {type(json_output_parser())}") # json
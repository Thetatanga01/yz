# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
#model = OllamaLLM(model="llama3.2:1b", base_url="localhost", port=11434)
mistral = OllamaLLM(model="mistral:7b")
openAI = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="""return the answer and Do not provide any explanations, calculations, or additional text."""),    
    HumanMessage(content="How old is planet earth?")
]

# Invoke the model with a message
result = mistral.invoke(messages)
print("Full result:")
print(result)
print("Content only:")
print(result)

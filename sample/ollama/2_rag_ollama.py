import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "pdf", "sozlesme.pdf")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
) 


# model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 4. Recursive Character-based Splitting
    # Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
    # Balances between maintaining coherence and adhering to character limits.
    print("\n--- Using Recursive Character-based Splitting ---")
    rec_char_splitter = RecursiveCharacterTextSplitter(
          chunk_size=100,
          chunk_overlap=20,
            separators=["\n\n", "\n", ".", " "] 
        )
    docs = rec_char_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")







# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the user's question
# query = "What is the title of the contract?"
#query = "Sözleşmenin konusu nedir?"
query = "What is the name of the company that signed the document with Vodafone?"
#query = "What are the mail adresses of the companies?"
#query = "What is the address of the company that signed the document with Vodafone?"
#query = "On which date will the contract come into effect?"
#query = "what date is the contract effective?"
#query = "Is there any passage about the effective date of the contract?"


# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")
# model = OllamaLLM(model="llama3.2:1b")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Cevap ---")
# print("Full result:")
# print(result)
#print("Content only:")
print(result.content)



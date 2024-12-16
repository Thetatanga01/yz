import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Çevresel değişkenleri yükle
load_dotenv()

# Pinecone başlat
pc = PineconeClient(
    api_key=os.getenv("PINECONE_API_KEY")
)

# Pinecone index kontrolü
index_name = "efdal"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# PDF dosyasını yükle
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "pdf", "sozlesme.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

# Text splitter
rec_char_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "]
)
docs = rec_char_splitter.split_documents(documents)

# Pinecone vector store oluştur
vector_store = Pinecone.from_documents(
    docs, embeddings, index_name=index_name
)

# Sorgu
# Define the user's question
# query = "What is the title of the contract?"
#query = "Sözleşmenin konusu nedir?"
#query = "what are the responsibilities of the company?"
#query = "What is the name of the company that signed the document with Vodafone?"
#query = "What are the mail adresses of the companies?"
#query = "What is the address of the company that signed the document with Vodafone?"
query = "On which date will the contract come into effect?"
#query = "what date is the contract effective?"
#query = "Is there any passage about the effective date of the contract?"
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)

# Sonuçları göster
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# OpenAI modeliyle cevap üret
combined_input = (
    "here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "Please provide an answer based only on the provided documents. "
      "Return only the answer."
    + "If the answer is not found in the documents, respond with 'bilmiyorum'"
)

model = ChatOpenAI(model_name="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant. You speak only turkish."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)
print("\n--- Cevap ---")
print(result.content)

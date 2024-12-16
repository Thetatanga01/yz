import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "pdf", "sozlesme.pdf")
persistent_directory = os.path.join(current_dir, "db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    rec_char_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "]
    )
    docs = rec_char_splitter.split_documents(documents)

    print(f"Number of document chunks: {len(docs)}")

    # Create the vector store and persist it automatically
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("Vector store created successfully.")
else:
    print("Vector store already exists.")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize chat model
model = ChatOpenAI(model_name="gpt-4o")

# Soru-Cevap Döngüsü
print("Chat başlatıldı. Çıkmak için 'çıkış' yazabilirsiniz.")
while True:
    user_query = input("Soru: ")
    if user_query.lower() in ["çıkış", "exit"]:
        print("Çıkılıyor...")
        break

    relevant_docs = retriever.invoke(user_query)

    # Belge içeriklerini birleştir
    combined_input = (
        "here are some documents that might help answer the question: "
        + user_query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "Please provide an answer based only on the provided documents. "
        + "Return only the answer."
        + "If the answer is not found in the documents, respond with 'bilmiyorum'."
    )

    # Mesajları oluştur
    messages = [
        SystemMessage(content="You are a helpful assistant. You speak only turkish."),
        HumanMessage(content=combined_input),
    ]

    # Yanıtı al ve göster
    result = model.invoke(messages)
    if result.content.strip() == "bilmiyorum":
        print("Cevap bulunamadı. Daha detaylı veya farklı bir soru sorabilirsiniz.")
    else:
        print("\nCevap: ", result.content.strip())

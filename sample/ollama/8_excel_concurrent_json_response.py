import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd  # Excel işlemleri için pandas eklendi
from langchain.schema import Document
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "excel", "magaza_bilgileri.xlsx")  # Excel dosya yolu
persistent_directory = os.path.join(current_dir, "db/excel")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the Excel content into a DataFrame
    df = pd.read_excel(file_path)

    # Combine all columns into a single text field for processing
    documents = []
    for _, row in df.iterrows():
        document_text = " ".join([str(value) for value in row.values if pd.notnull(value)])
        documents.append(Document(page_content=document_text, metadata={"source": "Excel"}))

    print(f"Loaded {len(documents)} rows from Excel.")

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

model = ChatOpenAI(model_name="gpt-4o")

questions_with_fields = [
    {"question": "Bayi adi ne?", "field": "storeName"},
    {"question": "Vdm adi ne?", "field": "vdmName"},
    {"question": "what is the store subscriber usage m2?", "field": "storeSubscriberUsage"},
    {"question": "Banka adi nedir?", "field": "bankName"},
    {"question": "Banka sube adi ne?", "field": "bankBranchName"},
    {"question": "what is bank branch no?", "field": "bankBranchNo"},
    {"question": "Banka hesap no ne?", "field": "bankAccountNo"},
    {"question": "what is IBAN no?", "field": "bankIbanNo"},
    {"question": "what is company owners name?", "field": "companyOwnersName"},
    {"question": "what is company owners gsm no?", "field": "companyOwnersGsmNo"},
    {"question": "what is company owners tc number?", "field": "companyOwnersTcNo"},
]

responses = {}

# Fonksiyon tanımı

def ask_question(question, field_name):
    retry_count = 0
    while retry_count < 3:
        relevant_docs = retriever.invoke(question)  # Dokümanları getir

        # Dokümanları birleştir ve modele gönder
        combined_input = (
            "here are some documents that might help answer the question: "
            + question
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])
            + "Please provide an answer based only on the provided documents. "
            + "Return only the answer with no additional information. Do not add any other thing to the answer."
            + "If the answer is not found in the documents, respond with 'bilmiyorum'."
        )

        messages = [
            SystemMessage(content="You are a helpful assistant. You speak only turkish. "
                                      "Return only the requested information without any additional text or formatting. Do not add explanations, introductions, or extra details. Respond in Turkish."),
            HumanMessage(content=combined_input),
        ]

        result = model.invoke(messages)
        if result.content.strip() != "bilmiyorum":
            return field_name, result.content.strip()
        retry_count += 1

    return field_name, None

# Concurrent işlem için ThreadPoolExecutor kullanımı
with ThreadPoolExecutor() as executor:
    future_to_field = {executor.submit(ask_question, item["question"], item["field"]): item["field"] for item in questions_with_fields}

    for future in as_completed(future_to_field):
        field_name, response = future.result()
        responses[field_name] = response

# JSON formatında çıktı
response_json = json.dumps(responses, ensure_ascii=False, indent=4)

# Yanıtları yazdır
print(response_json)

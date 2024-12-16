import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Constants
define_directories = {
    "file_path": "excel/magaza_bilgileri.xlsx",
    "persistent_directory": "db/excel",
    "metadataName": "yusuf"
}


# Function to initialize vector store
def initialize_vector_store(file_path, persistent_directory):
    print("Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

    df = pd.read_excel(file_path)

    documents = []
    for index, row in df.iterrows():
        document_text = " ".join([str(value) for value in row.values if pd.notnull(value)])
        documents.append(Document(page_content=document_text,
                                  metadata={"source": define_directories["metadataName"], "row_number": index + 1}))

    print(f"Loaded {len(documents)} rows from Excel.")

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "])
    docs = splitter.split_documents(documents)

    print(f"Number of document chunks: {len(docs)}")

    db = Chroma.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-large"),
                               persist_directory=persistent_directory)
    print("Vector store created successfully.")


# Function to retrieve documents
def get_relevant_docs(retriever, query):
    all_docs = retriever.get_relevant_documents(query)
    return [doc for doc in all_docs if doc.metadata.get("source") == define_directories["metadataName"]]


# Function to ask a question with retries
def ask_question(retriever, model, question, field_name, retries=3):
    for attempt in range(retries):
        relevant_docs = get_relevant_docs(retriever, question)

        if not relevant_docs:
            continue

        combined_input = (
                f"here are some documents that might help answer the question: {question}\n\nRelevant Documents:\n"
                + "\n\n".join([doc.page_content for doc in relevant_docs])
                + "\nPlease provide an answer based only on the provided documents. Return only the answer with no additional information. Do not add any other thing to the answer. If the answer is not found in the documents, respond with 'bilmiyorum'."
        )

        messages = [
            SystemMessage(
                content="You are a helpful assistant. You speak only turkish. Return only the requested information without any additional text or formatting. Do not add explanations, introductions, or extra details. Respond in Turkish."),
            HumanMessage(content=combined_input),
        ]

        result = model.invoke(messages)

        if result.content.strip() != "bilmiyorum":
            return field_name, result.content.strip()

    return field_name, None


# Main function
def main():
    file_path = define_directories["file_path"]
    persistent_directory = define_directories["persistent_directory"]


    #initialize_vector_store(file_path, persistent_directory)

    db = Chroma(persist_directory=persistent_directory,
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
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
        {"question": "who is company owners name?", "field": "companyOwnersName"},
        {"question": "what is company owners gsm no?", "field": "companyOwnersGsmNo"},
        {"question": "what is company owners tc number?", "field": "companyOwnersTcNo"},
    ]

    responses = {}

    with ThreadPoolExecutor() as executor:
        future_to_field = {
            executor.submit(ask_question, retriever, model, item["question"], item["field"]): item["field"]
            for item in questions_with_fields
        }

        for future in as_completed(future_to_field):
            field_name, response = future.result()
            responses[field_name] = response

    response_json = json.dumps(responses, ensure_ascii=False, indent=4)
    print(response_json)


# Entry point
if __name__ == "__main__":
    main()

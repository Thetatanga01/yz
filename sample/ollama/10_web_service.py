import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Constants
define_directories = {
    "file_path": "excel/magaza_bilgileri.xlsx",
    "persistent_directory": "db/excel",
    "metadataName": "yusuf"
}

# Initialize vector store
@app.on_event("startup")
def initialize_vector_store():
    file_path = define_directories["file_path"]
    persistent_directory = define_directories["persistent_directory"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

    df = pd.read_excel(file_path)
    documents = []
    for index, row in df.iterrows():
        document_text = " ".join([str(value) for value in row.values if pd.notnull(value)])
        documents.append(Document(page_content=document_text,
                                   metadata={"source": define_directories["metadataName"], "row_number": index + 1}))

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "])
    docs = splitter.split_documents(documents)

    global db
    db = Chroma.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-large"),
                                persist_directory=persistent_directory)

# Request model for asking questions
class QuestionRequest(BaseModel):
    question: str
    field_name: str

# Utility function to retrieve documents
def get_relevant_docs(retriever, query):
    all_docs = retriever.get_relevant_documents(query)
    return [doc for doc in all_docs if doc.metadata.get("source") == define_directories["metadataName"]]

# Utility function to ask a question with retries
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
            return {field_name: result.content.strip()}

    return {field_name: None}

# API endpoint to ask questions
@app.post("/ask")
async def ask_question_api(request: QuestionRequest):
    try:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        model = ChatOpenAI(model_name="gpt-4o")

        response = ask_question(retriever, model, request.question, request.field_name)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example batch endpoint to ask multiple questions
@app.post("/ask_batch")
async def ask_batch(questions: list[QuestionRequest]):
    try:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        model = ChatOpenAI(model_name="gpt-4o")

        responses = {}
        with ThreadPoolExecutor() as executor:
            future_to_field = {
                executor.submit(ask_question, retriever, model, item.question, item.field_name): item.field_name
                for item in questions
            }

            for future in as_completed(future_to_field):
                field_name, response = future.result().popitem()
                responses[field_name] = response

        return responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

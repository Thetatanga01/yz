import os
import uuid

import traceback
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from multipart import file_path
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import json

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Constants
define_directories = {
    "base_directory": "uploads",
    "persistent_directory": "db"
}

# Global vector store dictionary
vector_stores = {}


def process_excel(inputs: dict):
    file_path = inputs["file_path"]
    persistent_directory = inputs["persistent_directory"]
    metadata_name = inputs["metadata_name"]
    print("file_path ---> ", inputs["file_path"])

    df = pd.read_excel(file_path)
    documents = []
    for index, row in df.iterrows():
        document_text = " ".join([str(value) for value in row.values if pd.notnull(value)])
        documents.append(Document(page_content=document_text,
                                  metadata={"source": metadata_name, "row_number": index + 1}))

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "])
    docs = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-large"),
                                         persist_directory=persistent_directory)
    return vector_store


def process_pdf(inputs: dict):
    file_path = inputs["file_path"]
    persistent_directory = inputs["persistent_directory"]
    metadata_name = inputs["metadata_name"]
    print("file_path ---> ", inputs["file_path"])

    reader = PdfReader(file_path)
    documents = []
    for page_number, page in enumerate(reader.pages):
        document_text = page.extract_text()
        documents.append(Document(page_content=document_text,
                                  metadata={"source": metadata_name, "page_number": page_number + 1}))

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "])
    docs = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-large"),
                                         persist_directory=persistent_directory)
    return vector_store


# Tools for processing different file types
process_excel_tool = Tool(
    name="process_excel",
    func=lambda inputs: print(f"Received inputs: {inputs}") or process_excel(
        inputs=json.loads(inputs)
    ),
    description="Process Excel (.xlsx) files. Input keys: file_path, persistent_directory, metadata_name."
)

process_pdf_tool = Tool(
    name="process_pdf",
    func=lambda inputs: print(f"Received inputs: {inputs}") or process_pdf(
        inputs=json.loads(inputs)
    ),
    description="Process PDF (.pdf) files. Input keys: file_path, persistent_directory, metadata_name."
)

# Chat model and agent
chat_model = ChatOpenAI(model_name="gpt-4")
prompt_template = PromptTemplate(
    input_variables=["tools", "tool_names", "file_path", "persistent_directory", "metadata_name", "agent_scratchpad"],
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: you should look these arguments file_path, persistent_directory and metadata_name when answering the question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {file_path}, {persistent_directory} and {metadata_name} 
Thought:{agent_scratchpad}"""
)

react_agent = create_react_agent(
    llm=chat_model,
    tools=[process_excel_tool, process_pdf_tool],
    prompt=prompt_template
)
agent = AgentExecutor.from_agent_and_tools(agent=react_agent, tools=[process_excel_tool, process_pdf_tool],
                                           verbose=True)


@app.post("/upload")
async def upload_file(metadataName: str, file: UploadFile = File(...)):
    base_directory = define_directories["base_directory"]
    persistent_directory = os.path.join(define_directories["persistent_directory"], metadataName)

    # Generate a unique file name
    file_uuid = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1].lower()
    file_name = f"{file_uuid}{file_extension}"
    file_path = os.path.join(base_directory, file_name)

    # Save the uploaded file
    os.makedirs(base_directory, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Agent decides which tool to use based on file extension
    try:
        result = agent.invoke({
            "file_path": file_path,
            "persistent_directory": persistent_directory,
            "metadata_name": metadataName,
            "agent_scratchpad": ""
        })
        vector_stores[metadataName] = result
    except Exception as e:
        error_message = f"Error processing file: {str(e)}\nTraceback: {traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_message)

    return {"message": f"File uploaded and vector store for metadataName '{metadataName}' created successfully."}


class QuestionRequest(BaseModel):
    question: str
    field_name: str


# Utility function to retrieve documents
def get_relevant_docs(retriever, query):
    all_docs = retriever.get_relevant_documents(query)
    return all_docs


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
async def ask_question_api(metadataName: str = Query(...), request: QuestionRequest = None):
    try:
        # Vector store'u seç
        # if metadataName not in vector_stores:
        #     raise HTTPException(status_code=404, detail=f"'{metadataName}' metadatasina sahip vector store bulunamadi.")

        db = Chroma(persist_directory=os.path.join(define_directories["persistent_directory"], metadataName),
                    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
        # retriever = vector_stores[metadataName].as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        model = ChatOpenAI(model_name="gpt-4o")

        response = ask_question(retriever, model, request.question, request.field_name)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example batch endpoint to ask multiple questions
@app.post("/ask_batch")
async def ask_batch(metadataName: str = Query(...), questions: list[QuestionRequest] = None):
    try:
        if metadataName not in vector_stores:
            raise HTTPException(status_code=404, detail=f"'{metadataName}' metadatasina sahip vector store bulunamadi.")

        retriever = vector_stores[metadataName].as_retriever(search_type="similarity", search_kwargs={"k": 5})
        model = ChatOpenAI(model_name="gpt-4o")

        responses = {}
        for item in questions:
            response = ask_question(retriever, model, item.question, item.field_name)
            responses.update(response)

        return responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

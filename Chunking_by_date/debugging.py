from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time
import requests
import re

from langchain.retrievers import EnsembleRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import JinaEmbeddings
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# === Config ===
JINA_API_KEY = ""
QDRANT_URL = "http://192.168.1.13:6333/"
COLLECTION_NAME = "bank_statements_new"
K = 10
LLM_URL = "http://192.168.1.11:8078/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"
RERANKER_MODEL = "BAAI/bge-reranker-base"

app = FastAPI(title="RAG Retriever API")

print("Initializing retrievers, reranker, and LLM...")

# === Qdrant + Embeddings ===
client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY, model_name="jina-embeddings-v3")

dense_vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense",
    content_payload_key="text"
)
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": K})

sparse_model = FastEmbedSparse(model_name="qdrant/bm25")
sparse_vectorstore = QdrantVectorStore(
    client=client,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    sparse_embedding=sparse_model,
    retrieval_mode=RetrievalMode.SPARSE,
    sparse_vector_name="sparse",
    content_payload_key="text"
)
sparse_retriever = sparse_vectorstore.as_retriever(search_kwargs={"k": K})

ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
compressor = CrossEncoderReranker(model=reranker, top_n=K)

# === Prompt Template ===
template = """You are a financial assistant helping with forensic investigation of bank transactions.

You are provided data extracted from a bank statement. This data may contain:
- A metadata section (with details like account holder name, address, account status) — if present.

- Branch Name/ IFSC Code
- Transaction ID
- Transaction Date
- Transaction Type
- Instrument Number
- Narration
- Debit Amount
- Credit Amount
- Line Balance

Instructions:
- Carefully analyze the table, and if metadata is present, use it.
- If a Transaction Date filter is provided, only consider rows matching that date.
- Use only the provided data. Do not assume anything beyond it.
- If the answer is not found, state that clearly.
- If the answer is found, explain how you derived it and finish with:

Answer: <final answer>

---

{metadata}

---

{context}

---

Question: {question}

Now, think step by step:"""
prompt = ChatPromptTemplate.from_template(template)

# === Helpers ===
def format_docs(docs):
    headers = [
        "Branch Name/ IFSC Code", "Transaction ID", "Transaction Date",
        "Transaction Type", "Instrument Number", "Narration",
        "Debit Amount", "Credit Amount", "Line Balance"
    ]
    table_rows = []
    seen_headers = False

    for doc in docs:
        lines = doc.page_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "Branch Name" in line and "Transaction ID" in line:
                if seen_headers:
                    continue
                seen_headers = True
                continue
            columns = re.split(r'\s{2,}|\t+', line)
            if len(columns) == len(headers):
                table_rows.append(columns)

    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in table_rows:
        markdown += "| " + " | ".join(row) + " |\n"

    return markdown

def format_metadata(docs):
    for doc in docs:
        text = doc.page_content.strip()
        if "Account No" in text or "Nomination" in text or "IFSC" in text:
            return text
    return "Metadata not available in this chunk."

# === Schemas ===
class RAGGenerateRequest(BaseModel):
    query: str
    date: Optional[str] = None

class RAGGenerateResponse(BaseModel):
    answer: str
    context: List[str]
    retrieval_time: float
    formatting_time: float
    rerank_time: float
    generation_time: float
    total_time: float

# === Endpoint ===
@app.post("/rag/generate", response_model=RAGGenerateResponse)
async def generate_rag(request: RAGGenerateRequest):
    total_start = time.time()

    retrieval_start = time.time()
    docs = ensemble_retriever.invoke(request.query)
    retrieval_time = time.time() - retrieval_start

    if request.date:
        docs = [
            doc for doc in docs
            if re.search(rf'\b{re.escape(request.date)}\b', doc.page_content)
        ]

    rerank_start = time.time()
    reranked_docs = compressor.compress_documents(docs, query=request.query)
    rerank_time = time.time() - rerank_start

    formatting_start = time.time()
    context_str = format_docs(reranked_docs)
    metadata_str = format_metadata(docs)
    formatting_time = time.time() - formatting_start

    formatted_prompt = prompt.invoke({
        "metadata": metadata_str,
        "context": context_str,
        "question": request.query
    }).to_string()

    print("------ FORMATTED PROMPT SENT TO LLM ------")
    print(formatted_prompt)
    print("------------------------------------------")

    # === Call OpenAI-compatible LLM ===
    generation_start = time.time()
    llm_payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": formatted_prompt}
        ]
    }
    response = requests.post(
        LLM_URL,
        headers={"Content-Type": "application/json"},
        json=llm_payload
    )

    if response.status_code != 200:
        raise RuntimeError(f"LLM returned error: {response.status_code} - {response.text}")

    data = response.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    generation_time = time.time() - generation_start
    total_time = time.time() - total_start

    return RAGGenerateResponse(
        answer=answer,
        context=[doc.page_content for doc in reranked_docs],
        retrieval_time=retrieval_time,
        formatting_time=formatting_time,
        rerank_time=rerank_time,
        generation_time=generation_time,
        total_time=total_time
    )
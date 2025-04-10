from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

HF_API_KEY = "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"  
DB_FAISS_PATH = "vectorstore/db_faiss"

try:
    if not os.path.exists(DB_FAISS_PATH):
        logger.info("Vector store not found. Creating new FAISS index...")
        loader = CSVLoader(file_path="shl_product_catalog.csv")
        data = loader.load()

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(data, embedding_model)
        db.save_local(DB_FAISS_PATH)
        logger.info("Vector store created and saved.")
    else:
        logger.info("Loading existing FAISS vector store...")

    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
except Exception as e:
    logger.error(f"Vector store initialization failed: {e}")
    raise RuntimeError(f"Vector store error: {e}")

try:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.5,
        max_length=256,
        huggingfacehub_api_token=HF_API_KEY
    )
    logger.info("LLM successfully initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Hugging Face model: {e}")
    raise RuntimeError(f"LLM error: {e}")

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
You are an intelligent recommendation assistant that helps hiring managers choose the most relevant SHL individual assessments.

Use only the information provided in the context to recommend SHL assessments based on the user's input. 
Do NOT hallucinate or assume any information that is not in the context. 
Give main focus on job role, test type and duration.

Each recommendation must include:
- Assessment Name
- URL (must be from SHL product catalog)
- Remote Testing Support (Yes/No)
- Adaptive/IRT Support (Yes/No)
- Duration
- Test Type

Context: {context}
Question: {question}

Recommend only 10 listings.  
Start your answer directly. No introductions or small talk.
"""

prompt_template = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt_template}
)

class Query(BaseModel):
    query: str

# Endpoint for recommendations
@app.post("/recommend")
async def recommend(data: Query):
    try:
        logger.info(f"Received query: {data.query}")
        result = qa_chain.invoke({"query": data.query})
        return {"recommendation": result["result"]}
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )

@app.get("/")
def health_check():
    return {"status": "OK", "message": "SHL Assessment Recommender API is running"}

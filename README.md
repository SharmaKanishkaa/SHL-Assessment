# SHL Assessment Recommendation System

An intelligent, Retrieval-Augmented Generation (RAG) powered tool that recommends the top 1–10 most relevant SHL assessments based on job descriptions or natural language queries. Designed to assist hiring managers in navigating the extensive SHL product catalog with precision and ease.

---

## Live Demo & API

- **Web App:** [Streamlit Demo](https://shlrecommendersys.streamlit.app/)
- **API:** [Render Web Service](https://shl-recommender-byqr.onrender.com)
- **Approach:** [Doc](https://drive.google.com/drive/u/0/recent)

---

## Problem Statement

Hiring managers often struggle to match SHL assessments to job descriptions due to unstructured input formats. This system bridges the gap by mapping free-text queries to relevant assessments using semantic similarity and large language models.

---

## Tools & Technologies

| Category            | Tools / Libraries                                          |
|---------------------|------------------------------------------------------------|
| Scripting & Logic   | Python                                                     |
| Web Scraping        | Selenium                                                   |
| Data Handling       | Pandas, LangChain `CSVLoader`                              |
| Embedding Models    | `sentence-transformers/all-MiniLM-L6-v2`                   |
| Vector Store        | FAISS (Facebook AI Similarity Search)                      |
| LLM & Embeddings    | `mistralai/Mistral-7B-Instruct-v0.3` via Hugging Face API  |
| UI Development      | Streamlit                                                  |

---

## End-to-End Workflow

### Web Scraping & Dataset Generation
- Scraped SHL Product Catalog using Selenium.
- Extracted fields:
  - Product Name, URL
  - Remote Testing Availability
  - Adaptive/IRT Support
  - Duration, Test Type, Job Role
  - Description
- Output: `shl_product_catalog.csv`

---

### Embedding & Vector Store
- Loaded dataset using LangChain `CSVLoader`.
- Embedded entries using `all-MiniLM-L6-v2`.
- Indexed embeddings with FAISS and saved to `vectorstore/db_faiss`.

---

### RAG Pipeline
- **Retriever:** FAISS fetches top-3 semantically similar assessments.
- **LLM Generator:** 
  - `Mistral-7B-Instruct-v0.3` from Hugging Face
  - Custom prompt ensures structured, relevant output.
- **LangChain’s `RetrievalQA`:** Combines user query + retrieved context for coherent responses.

---

### Interactive Streamlit UI
- Users input job descriptions or free-text queries.
- Top 1–10 assessment recommendations returned in a table:
  - Assessment Name
  - URL
  - Duration
  - Remote/Adaptive Support
  - Description

---

## Evaluation

| Metric           | Description                                                  |
|------------------|--------------------------------------------------------------|
| **Recall@3**     | Measures if relevant assessments are in top-3 recommendations |
| **MAP@3**        | Measures ranking precision of recommended items              |

- RAG improves relevance by combining semantic search + LLM generation.
- Prompt tuning minimizes hallucinations and ensures structured output.



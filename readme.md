
# FinSight-RAG 🪙📊
> **Intelligent Financial Assistant powered by Retrieval-Augmented Generation (RAG)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Hackathon%20Ready-success.svg)]()

---

## 📌 Project Overview
**FinSight-RAG** is an intelligent finance chatbot designed to help users navigate complex financial documents, market transaction records, regulatory filings, and compliance reports.

Built during the **AI Odyssey Mini Hackathon (AAAI Student Chapter)**, this chatbot leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, grounded answers directly sourced from a 15GB+ domain-specific financial dataset. Instead of relying purely on an AI model's general memory (which can guess or produce false facts), our chatbot retrieves verified excerpts from trusted financial data before generating answers.

---

## ✨ Key Features
- **🔍 Grounded Financial Search**: Queries are cross-referenced with official financial filings, market transaction datasets, and regulatory compliance records.
- **🛡️ Hallucination-Resistant Responses**: Answers are tied strictly to retrieved context with transparent citations and source highlights.
- **⚡ Fast Vector Retrieval**: Uses semantic search to pinpoint relevant financial figures, terms, and clauses in milliseconds.
- **💬 Interactive Chat Interface**: A clean, responsive web interface built for natural conversational Q&A, follow-up inquiries, and prompt suggestions.
- **📊 Real-time Source Transparency**: Displays the exact chunks and confidence scores used to answer each query.

---

## 🏗️ How It Works (The RAG Pipeline in Plain English)
1. **Data Ingestion & Cleaning**: Ingests raw financial records formatted as **Parquet / CSV & PDF filings** (e.g., CSV, JSON, PDF, or Parquet), stripping noise and structuring key tables.
2. **Chunking**: Breaks long documents into bite-sized, readable paragraphs (chunks) that preserve contextual meaning.
3. **Embedding Generation**: Converts text chunks into mathematical vectors (numerical fingerprints) using a pretrained embedding model.
4. **Vector Database Search**: When a user asks a question, the system finds the closest matching vectors (most relevant documents).
5. **Grounded Answer Generation**: Feeds the question along with the retrieved source text to a Large Language Model (LLM), which drafts an accurate, concise reply.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10+** or **Node.js 18+** (depending on your chosen backend)
- **Git** installed on your system
- An API Key for your LLM provider (e.g., Google Gemini, OpenAI, or Hugging Face)

### 2. Installation
Clone the repository and install the required dependencies:

```bash
# 1. Clone this repository
git clone https://github.com/[YOUR_GITHUB_ORGANIZATION_OR_USERNAME]/FinSight-RAG.git
cd FinSight-RAG

# 2. Create and activate a virtual environment (Python)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your secret keys:

```env
# AI Model & API Configuration
LLM_API_KEY="your_api_key_here"
EMBEDDING_MODEL="text-embedding-004" # or all-MiniLM-L6-v2 / text-embedding-3-small
VECTOR_DB_PATH="./data/vector_index"
DATASET_PATH="./data/finance_dataset.Parquet / CSV & PDF filings"
PORT=3000
```

### 4. Ingest and Index the Dataset
Process the financial dataset and build the local vector database:

```bash
# Run the preprocessing and indexing script
python ingest.py --data-path ./data/finance_dataset.Parquet / CSV & PDF filings
```

### 5. Run the Application
Launch the local web development server:

```bash
# For Streamlit interface:
streamlit run app.py

# OR for FastAPI + Web UI:
uvicorn main:app --reload --port 8000
```

Open your browser and navigate to `http://localhost:8501` (or `http://localhost:8000`).

---

## 💡 Example Usage & Sample Queries

Try asking the chatbot domain-specific financial queries such as:
- *"What were the quarterly revenue growth trends across major sectors in Q3?"*
- *"Summarize the risk factors disclosed in the recent regulatory compliance filings."*
- *"Identify any recorded discrepancies in transaction records dated between 2024-Q1 and 2024-Q2."*
- *"Explain the reserve liquidity requirements according to the provided banking guidelines."*

---

## 📂 Project Structure
```text
FinSight-RAG/
├── data/                      # Dataset storage (raw & processed)
│   └── finance_dataset.Parquet / CSV & PDF filings
├── src/
│   ├── ingestion/             # Cleaning, preprocessing & chunking scripts
│   ├── vector_store/          # Vector index and similarity search modules
│   ├── llm/                   # Prompt templates & LLM generation logic
│   └── ui/                    # Frontend interface (Streamlit / React)
├── app.py                     # Main application entry point
├── ingest.py                  # CLI pipeline for data indexing
├── requirements.txt           # Python dependencies
├── IMPLEMENTATION_PLAN.md     # Team architecture, workflows & roadmap
└── README.md                  # Project documentation
```

---

## 👥 Team Credits
Proudly developed by our 3-person team for **AI Odyssey Mini Hackathon**:

| Role | Name | Primary Responsibilities | Contact / Profile |
| :--- | :--- | :--- | :--- |
| **Member 1 (Data & AI)** | **Alex Chen (AI & Data Lead)** | Data Ingestion, Chunking Strategy, Vector Indexing, Prompt Engineering | [@github_handle](https://github.com) |
| **Member 2 (Backend & Cloud)** | **Jordan Taylor (Backend & Cloud)** | Retrieval API, Vector Database, Pipeline Integration, Live Deployment | [@github_handle](https://github.com) |
| **Member 3 (Frontend & UX)** | **Samira Patel (Frontend & UX)** | Web Chat Interface, Citation Viewer, Latency Optimization, Demo Walkthrough | [@github_handle](https://github.com) |

---

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

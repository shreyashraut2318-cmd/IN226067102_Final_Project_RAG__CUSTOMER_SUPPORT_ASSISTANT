# RAG-Based Customer Support Assistant

## Project Overview
This project is part of an **Advanced Generative AI Internship** at **Innomatics Research Labs**. It features a production-grade **Retrieval-Augmented Generation (RAG)** system designed for customer support. The system processes PDF knowledge bases, utilizes a graph-based workflow for decision-making, and includes a **Human-in-the-Loop (HITL)** escalation module for complex queries[cite: 1, 2, 9].

## Key Features
* **Contextual Question Answering**: Retrieves specific information from PDF documents to answer user queries.
* **Graph-Based Workflow**: Orchestrated using **LangGraph** to manage state transitions and conditional logic.
* **Conditional Routing**: Automatically detects when information is missing and routes the query accordingly.
* **Human-in-the-Loop (HITL)**: Supports manual intervention for low-confidence responses or complex intents.
* **Persistent Storage**: Uses **ChromaDB** to store and retrieve document embeddings locally[cite: 13, 31].

## Tech Stack
* **Framework**: FastAPI 
* **Orchestration**: LangGra
* **Vector Database**: ChromaDB 
* **LLM**: Llama 3 via Groq 
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)

## Project Structure
rag-assistant/
├── data/               # PDF knowledge base
├── chromadb_storage/   # Persistent vector database
├── src/
│   ├── ingestion.py    # PDF processing and embedding logic
│   ├── graph.py        # LangGraph nodes and routing logic
│   └── main.py         # FastAPI entry point
├── .env                # API Keys (Excluded via .gitignore)
├── requirements.txt    # Project dependencies
└── README.md           # Documentation

## Setup Instructions
1. Installation:
Clone the repository and install the dependencies:  pip install -r requirements.txt
2. Configuration:
Create a .env file in the root directory and add: GROQ_API_KEY=your_api_key_here
3. Data Ingestion: 
Place your PDF knowledge base in the data/ folder and type your file name in run_ingestion.py file then process it: python run_ingestion.py
4. Running the Application: 
Start the FastAPI server: python -m src.main

# Access the interactive Swagger UI at http://127.0.0.1:8000/docs
The system follows a 2-node flow (Retrieve -> Process) with conditional routing.
Success Path: Query -> Retrieve -> Process -> Response.
Escalation Path: Query -> Retrieve -> Process (No Context) -> Human Agent.

Contributor: Shreyash Naresh Raut AI & DS Student | B.Tech student at LSPGCOER affilated with DBATU 
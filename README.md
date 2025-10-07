# Hotel_Taj_RAG_Based_Receptionist
# Hotel Taj AI Q&A Assistant 💬

An AI-powered Question & Answer assistant for **Hotel Taj** using **Retrieval-Augmented Generation (RAG)**.  
This project allows users to ask questions about the restaurant (menus, chefs, timings, etc.) based on its text data, and get answers with source citations.

---

## Features

- Load restaurant data from text files (`Hotel_Taj.txt`).
- Split large documents into smaller chunks for efficient processing.
- Use **HuggingFace embeddings** (`all-MiniLM-L6-v2`) and **ChromaDB** for vector storage.
- Perform similarity search to retrieve relevant document chunks.
- Integrate **HuggingFace LLM** (`Nous-Hermes-13b-instruct-v0.1`) for natural language answers.
- Provide **answer + source document** via a user-friendly **Gradio** interface.
- Gracefully handle empty queries and errors.

---

## Demo

- Ask questions like:
  - "What are the menu prices?"
  - "Who is the head chef?"
  - "Is it plant-based?"
  - "What are the opening hours on Saturday?"

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/hotel-taj-qa.git
cd hotel-taj-qa

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt

HUGGINGFACE_API_KEY=your_huggingface_api_key

Add your restaurant data:
Hotel_Taj.txt
Usage
python app.py

This will:

Load and split documents.

Create a ChromaDB vector store.

Set up the RAG chain.

Launch a Gradio web app for interactive Q&A.

Open the URL displayed in your terminal to interact with the assistant.

How It Works

Document Loading & Splitting
Large text files are split into smaller chunks using RecursiveCharacterTextSplitter.

Embeddings & Vector Store
Chunks are embedded with HuggingFace embeddings and stored in ChromaDB for similarity search.

Retrieval-Augmented Generation (RAG)

Retrieve relevant chunks using vector similarity.

Feed retrieved chunks to a HuggingFace LLM to generate answers.

Return answers with source document references.

User Interface

Gradio app allows users to input questions and view answers with sources.

Includes example questions for testing.

Install all dependecies with
pip install -r requirements.txt
Contact

Developed by Sanket 

# Hotel Taj AI Q&A Assistant 💬

An AI-powered Question & Answer assistant for **Hotel Taj** built using **Retrieval-Augmented Generation (RAG)**. This project allows users to ask questions about the restaurant (menus, chefs, timings, etc.) based on text data and get answers with source citations.

---

## Installation

Instructions on how to get a copy of the project and run it on your local machine.

### Prerequisites

_A guide on how to install the tools needed for running the project._

- Python 3.10 or above
- Git
- HuggingFace API Key

### Step-by-Step Setup

```bash
# Clone the repository
git clone https://github.com/your-username/hotel-taj-qa.git
cd hotel-taj-qa
```
# Create and activate a virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace API key
# Linux/macOS:
export HUGGINGFACE_API_KEY=your_huggingface_api_key
# Windows:
set HUGGINGFACE_API_KEY=your_huggingface_api_key

# Add restaurant data
# Place Hotel_Taj.txt in this project directory

# Usage

Explain how to test the project and give some examples.

```bash
# Run the application
python app.py
```

# Run the application
python app.py
Ask questions like:

"What are the menu prices?"

"Who is the head chef?"

"Is the menu plant-based?"

"What are the opening hours on Saturday?"

# Deploy
To deploy a new project or host this app:

A server or cloud instance with Python 3.10+ installed

Install all dependencies using:
```bash
pip install -r requirements.txt
```





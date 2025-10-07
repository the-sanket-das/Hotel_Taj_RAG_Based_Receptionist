import os
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import gradio as gr
from langchain.chains import RetrievalQAWithSourcesChain


load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

DATA_FILE_PATH = "Hotel_Taj.txt"
loader = TextLoader(DATA_FILE_PATH, encoding = "utf-8")

raw_documents = loader.load()
print(f"Successfully loaded {len(raw_documents)} document(s).")

print("\nSplitting the loaded document into smaller chunks...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                               chunk_overlap = 150,) 


documents = text_splitter.split_documents(raw_documents)

if not documents:
    raise ValueError("Error: Splitting resulted in zero documents. Check the input file and splitter settings.")
print(f"Document split into {len(documents)} chunks.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("\nCreating ChromaDB vector store and embedding documents...")

vector_store = Chroma.from_documents(documents = documents, embedding = embeddings)  

vector_count = vector_store._collection.count()
print(f"ChromaDB vector store created with {vector_count} items.")

if vector_count == 0:
    raise ValueError("Vector store creation resulted in 0 items. Check previous steps.")

stored_data = vector_store._collection.get(include=["embeddings", "documents"], limit = 1)  

print("\nEmbedding vector:\n", stored_data['embeddings'][0])
print(f"\nFull embedding has {len(stored_data['embeddings'][0])} dimensions.")
print("\n--- Testing Similarity Search in Vector Store ---")
test_query = "What different menus are offered?"
print(f"Searching for documents similar to: '{test_query}'")

try:
    similar_docs = vector_store.similarity_search(test_query, k = 2)
    print(f"\nFound {len(similar_docs)} similar documents:")

    
    for i, doc in enumerate(similar_docs):
        print(f"\n--- Document {i+1} ---")
        
        content_snippet = doc.page_content[:700].strip() + "..."
        source = doc.metadata.get("source", "Unknown Source") 
        print(f"Content Snippet: {content_snippet}")
        print(f"Source: {source}")

except Exception as e:
    print(f"An error occurred during similarity search: {e}")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever configured successfully from vector store.")

llm = HuggingFaceEndpoint(
    repo_id="Nous-Hermes-13b-instruct-v0.1",
    task="text-generation",  
    temperature=0.7,
    max_new_tokens=512,
    huggingfacehub_api_token=huggingface_api_key
)

print("HuggingFace LLM successfully initialized.")

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm = llm,
                                                       chain_type = "stuff",
                                                       retriever = retriever,
                                                       return_source_documents = True, 
                                                       verbose = True) 

print("RetrievalQAWithSourcesChain created")

print("\n--- Testing the Full RAG Chain ---")
chain_test_query = "What kind of food does Hotel Taj serve?"
print(f"Query: {chain_test_query}")

try:
    result = qa_chain.invoke({"question": chain_test_query})

    if "source_documents" in result:
        print("\n--- Source Document Snippets ---")
        for i, doc in enumerate(result["source_documents"]):
            content_snippet = doc.page_content[:250].strip()
            print(f"Doc {i+1}: {content_snippet}")

except Exception as e:
    print(f"\nAn error occurred while running the chain: {e}")

def ask_HotelTajmadison_assistant(user_query):
    """
    Processes the user query using the RAG chain and returns formatted results.
    """
    print(f"\nProcessing Gradio query: '{user_query}'")
    if not user_query or user_query.strip() == "":
        print("--> Empty query received.")
        return "Please enter a question.", ""  # Handle empty input gracefully

    try:
        # Run the query through our RAG chain
        result = qa_chain.invoke({"question": user_query})

        # Extract answer and sources
        answer = result.get("answer", "Sorry, I couldn't find an answer in the provided documents.")
        sources = result.get("sources", "No specific sources identified.")

        # Basic formatting for sources (especially if it just returns the filename)
        if sources == DATA_FILE_PATH:
            sources = f"Retrieved from: {DATA_FILE_PATH}"
        elif isinstance(sources, list):  # Handle potential list of sources
            sources = ", ".join(list(set(sources)))  # Unique, comma-separated

        print(f"--> Answer generated: {answer[:100].strip()}...")
        print(f"--> Sources identified: {sources}")

        # Return the answer and sources to be displayed in Gradio output components
        return answer.strip(), sources

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(f"--> Error during chain execution: {error_message}")
        # Return error message to the user interface
        return error_message, "Error occurred"

print("\nSetting up Gradio interface...")

with gr.Blocks(theme=gr.themes.Soft(), title="Hotel Taj Q&A Assistant") as demo:
    gr.Markdown(
        """
        # Hotel Taj - AI Q&A Assistant 💬
        Ask questions about the restaurant based on its website data.
        The AI provides answers and cites the source document.
        *(Examples: What are the menu prices? Who is the chef? Is it plant-based?)*
        """
    )

    question_input = gr.Textbox(
        label = "Your Question:",
        placeholder = "e.g., What are the opening hours on Saturday?",
        lines = 2, 
    )    
    with gr.Row():
        answer_output = gr.Textbox(label="Answer:", interactive=False, lines=6) 
        sources_output = gr.Textbox(label="Sources:", interactive=False, lines=2)

    with gr.Row():
        submit_button = gr.Button("Ask Question", variant="primary")
        clear_button = gr.ClearButton(components=[question_input, answer_output, sources_output], value="Clear All")

    gr.Examples(
        examples=[
            "What are the different menu options and prices?",
            "Who is the head chef?",
            "What is Magic Farms?"],
        inputs=question_input, 
        
        cache_examples=False,
    )
    submit_button.click(fn = ask_HotelTajmadison_assistant, inputs = question_input, outputs = [answer_output, sources_output])

print("Gradio interface defined.")
print("\nLaunching Gradio app... (Stop the kernel or press Ctrl+C in terminal to quit)")
demo.launch()  
import gradio as gr
import os
import base64
from byaldi import RAGMultiModalModel
from PIL import Image
import io
import tempfile
import uuid

# Initialize the RAG model
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=0)

# Create a temporary directory to store images
temp_dir = tempfile.mkdtemp()

def save_image_temp(image):
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(temp_dir, filename)
    image.save(filepath)
    return filepath

def ingest_pdf(file):
    if file is None:
        return "No file uploaded."
    
    # Index the PDF
    RAG.index(
        input_path=file.name,
        index_name="uploaded_pdf",
        store_collection_with_index=True,
        overwrite=True
    )
    
    return "PDF ingested successfully."

def search_pdf(query):
    if not query:
        return "Please enter a search query.", []
    
    results = RAG.search(query, k=5)  # Limit to top 5 results
    if not results:
        return "No results found.", []
    
    image_paths = []
    for result in results:
        image_base64 = result.base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_path = save_image_temp(image)
        image_paths.append(image_path)
    
    return f"Found {len(image_paths)} results.", image_paths

def query_and_retrieve(query):
    response, image_paths = search_pdf(query)
    return response, image_paths

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Simplified RAG Interface with PDF Support")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        ingest_button = gr.Button("Ingest PDF")
    
    ingest_output = gr.Textbox(label="Ingestion Status")
    
    query_input = gr.Textbox(label="Enter your query")
    query_button = gr.Button("Submit Query")
    
    response_output = gr.Textbox(label="Response")
    image_gallery = gr.Gallery(label="Retrieved Images", columns=5, height=300)
    
    ingest_button.click(ingest_pdf, inputs=[pdf_input], outputs=[ingest_output])
    query_button.click(query_and_retrieve, inputs=[query_input], outputs=[response_output, image_gallery])

demo.launch()
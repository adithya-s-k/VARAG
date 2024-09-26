import os
import io
import torch
import lancedb
import gradio as gr
from PIL import Image
from typing import List
from dotenv import load_dotenv
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import fitz  # PyMuPDF
from transformers import CLIPProcessor, CLIPModel
import pyarrow as pa
from multiprocessing import Pool

load_dotenv()

# Assuming you have a VLM class defined elsewhere
from varag.vlms import OpenAI  # Replace with your actual VLM import

# Initialize CLIP model and processor for embeddings
MODEL_ID = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(device)

# Initialize VLM
vlm = OpenAI()  # Replace with your actual VLM initialization

# Connect to LanceDB
db = lancedb.connect("~/lancedb")

# Define schema for the unified table
schema = pa.schema([
    pa.field("document_name", pa.string()),
    pa.field("page_number", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), 512)),
    pa.field("image", pa.binary()),
])

# Create or open the unified table
table = db.create_table("unified_documents", schema=schema, mode="overwrite")

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    try:
        return convert_from_path(pdf_path)
    except PDFInfoNotInstalledError:
        print("Poppler not installed. Falling back to PyMuPDF.")
        pdf_document = fitz.open(pdf_path)
        return [Image.frombytes("RGB", [int(page.rect.width), int(page.rect.height)], page.get_pixmap().samples)
                for page in pdf_document]

def pil_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def embed_images(images: List[Image.Image]):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy()

def ingest_pdf(pdf_file):
    try:
        pdf_path = pdf_file.name
        images = pdf_to_images(pdf_path)
        embeddings = embed_images(images)
        
        with Pool() as p:
            image_bytes = p.map(pil_to_bytes, images)
        
        data = [{"document_name": os.path.basename(pdf_path),
                 "page_number": i,
                 "vector": emb.tolist(),
                 "image": img_bytes} 
                for i, (emb, img_bytes) in enumerate(zip(embeddings, image_bytes))]
        
        table.add(data)
        
        return f"PDF ingested and stored in unified table. Total documents: {table}"
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}"

def embed_text(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.squeeze().cpu().numpy()

def search_images(query: str):
    query_embedding = embed_text(query)
    
    results = table.search(query_embedding.tolist()).limit(3).to_list()
    
    images = [Image.open(io.BytesIO(result['image'])) for result in results]
    return images

def generate_response(query: str):
    try:
        images = search_images(query)
        
        # Combine images into a single image
        combined_image = Image.new('RGB', (images[0].width * len(images), images[0].height))
        for i, img in enumerate(images):
            combined_image.paste(img, (i * img.width, 0))
        
        # Generate response using VLM
        response = vlm.response(query, combined_image)
        
        return response, combined_image
    except Exception as e:
        return f"Error generating response: {str(e)}", None

# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Unified PDF Image RAG System")
        
        with gr.Tab("Ingest and Query"):
            pdf_input = gr.File(label="Upload PDF")
            ingest_button = gr.Button("Ingest PDF")
            ingest_output = gr.Textbox(label="Ingestion Status")
            
            query_input = gr.Textbox(label="Enter your query")
            query_button = gr.Button("Generate Response")
            response_output = gr.Textbox(label="VLM Response")
            image_output = gr.Image(label="Retrieved Images")
        
        ingest_button.click(ingest_pdf, inputs=pdf_input, outputs=ingest_output)
        query_button.click(generate_response, inputs=query_input, outputs=[response_output, image_output])
    
    return demo

# Launch the app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch()
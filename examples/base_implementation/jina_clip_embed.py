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
import pyarrow as pa
from multiprocessing import Pool
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

load_dotenv()

# Assuming you have a VLM class defined elsewhere
from varag.vlms import OpenAI  # Replace with your actual VLM import

# Initialize Jina CLIP model
MODEL_ID = "jinaai/jina-clip-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
model = SentenceTransformer("jinaai/jina-clip-v1", trust_remote_code=True)
# Initialize VLM
vlm = OpenAI()  # Replace with your actual VLM initialization

# Connect to LanceDB
db = lancedb.connect("~/lancedb")

# Define schema for the unified table
schema = pa.schema(
    [
        pa.field("document_name", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 768)),  # Adjust dimension if needed
        pa.field("image", pa.binary()),
    ]
)

# Create or open the unified table
table = db.create_table("unified_documents", schema=schema, mode="overwrite")


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    try:
        return convert_from_path(pdf_path)
    except PDFInfoNotInstalledError:
        print("Poppler not installed. Falling back to PyMuPDF.")
        pdf_document = fitz.open(pdf_path)
        return [
            Image.frombytes(
                "RGB",
                [int(page.rect.width), int(page.rect.height)],
                page.get_pixmap().samples,
            )
            for page in pdf_document
        ]


def pil_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# def embed_images(images: List[Image.Image]):
#     with torch.no_grad():
#         image_features = model.encode_image(images)
#     return image_features.cpu().numpy()


def embed_images(images: List[Image.Image]):
    image_features = model.encode(images)
    return image_features  # This is already a numpy array


def ingest_pdf(pdf_file):
    try:
        pdf_path = pdf_file.name
        images = pdf_to_images(pdf_path)
        embeddings = embed_images(images)

        with Pool() as p:
            image_bytes = p.map(pil_to_bytes, images)

        data = [
            {
                "document_name": os.path.basename(pdf_path),
                "page_number": i,
                "vector": emb.tolist(),
                "image": img_bytes,
            }
            for i, (emb, img_bytes) in enumerate(zip(embeddings, image_bytes))
        ]

        table.add(data)

        return f"PDF ingested and stored in unified table. Total documents: {table}"
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}"


# def embed_text(text: str):
#     with torch.no_grad():
#         text_features = model.encode_text([text])
#     return text_features.squeeze().cpu().numpy()


def embed_text(text: str):
    text_features = model.encode([text])
    return text_features.squeeze()  # This is already a numpy array


def search_images(query: str):
    query_embedding = embed_text(query)

    results = (
        table.search(
            query_embedding.tolist(), query_type="vector", vector_column_name="vector"
        )
        .limit(3)
        .to_list()
    )

    images = [Image.open(io.BytesIO(result["image"])) for result in results]
    return images


def generate_response(query: str):
    try:
        images = search_images(query)

        # Generate response using VLM with multiple images
        response = vlm.query(query, images)

        return response, images
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
            image_output = gr.Gallery(label="Retrieved Images")

        ingest_button.click(ingest_pdf, inputs=pdf_input, outputs=ingest_output)
        query_button.click(
            generate_response,
            inputs=query_input,
            outputs=[response_output, image_output],
        )

    return demo


# Launch the app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch()

import gradio as gr
import os
import lancedb
from dotenv import load_dotenv
from typing import List
from PIL import Image
import base64
import io
import argparse

# Import the colpali class and OpenAI VLM
from varag.rag import ColpaliRAG
from varag.vlms import OpenAI
from varag.utils import get_model_colpali

load_dotenv()

# Initialize shared database
shared_db = lancedb.connect("~/shared_rag_db")
model, processor = get_model_colpali("vidore/colpali-v1.2")


colpali_rag = ColpaliRAG(
    db=shared_db,
    table_name="colpaliDemo",
)

# Initialize VLM
vlm = OpenAI()


def ingest_pdfs(pdf_files, table_name, recursive, verbose):
    try:
        if table_name:
            colpali_rag.change_table(table_name)

        file_paths = [pdf_file.name for pdf_file in pdf_files]
        colpali_rag.index(
            file_paths, overwrite=False, recursive=recursive, verbose=verbose
        )
        return f"PDFs ingested successfully into table '{colpali_rag.table_name}'."
    except Exception as e:
        return f"Error ingesting PDFs: {str(e)}"


def search_and_analyze(query, table_name, topk):
    try:
        if table_name:
            colpali_rag.change_table(table_name)

        results = colpali_rag.search(query, k=topk)

        pil_images = []
        for result in results:
            image_data = result["image"]
            if isinstance(image_data, Image.Image):
                pil_images.append(image_data)
            elif isinstance(image_data, str):
                # Assume it's base64 encoded
                pil_images.append(Image.open(io.BytesIO(base64.b64decode(image_data))))
            elif isinstance(image_data, bytes):
                pil_images.append(Image.open(io.BytesIO(image_data)))
            else:
                raise ValueError(f"Unexpected image type: {type(image_data)}")

        # Prepare context for VLM
        context = f"Query: {query}\n\nRelevant image information:\n"
        for i, result in enumerate(results, 1):
            context += f"Image {i}: From document '{result['name']}', page {result['page_number']}\n"
            if "metadata" in result:
                context += f"Metadata: {result['metadata']}\n"
            if "page_text" in result:
                context += f"Page text: {result['page_text'][:500]}...\n\n"

        # Generate response using VLM
        vlm_response = vlm.query(context, pil_images, max_tokens=500)

        return vlm_response, pil_images
    except Exception as e:
        return f"Error generating response: {str(e)}", []


def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# HybridColpaliRAG Image Search and Analysis with VLM")

        with gr.Tab("Ingest PDFs"):
            pdf_input = gr.File(
                label="Upload PDF(s)", file_count="multiple", file_types=["pdf"]
            )
            table_name_input = gr.Textbox(
                label="Table Name (optional)", placeholder="default_table"
            )
            recursive_checkbox = gr.Checkbox(label="Recursive Indexing", value=False)
            verbose_checkbox = gr.Checkbox(label="Verbose Output", value=True)
            ingest_button = gr.Button("Ingest PDFs")
            ingest_output = gr.Textbox(label="Ingestion Status")

        with gr.Tab("Search and Analyze"):
            query_input = gr.Textbox(label="Enter your query")
            search_table_name_input = gr.Textbox(
                label="Table Name (optional)", placeholder="default_table"
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=3, step=1, label="Top K Results"
            )
            search_button = gr.Button("Search and Analyze")
            response_output = gr.Textbox(label="VLM Response")
            image_output = gr.Gallery(label="Retrieved Images")

        ingest_button.click(
            ingest_pdfs,
            inputs=[
                pdf_input,
                table_name_input,
                recursive_checkbox,
                verbose_checkbox,
            ],
            outputs=ingest_output,
        )
        search_button.click(
            search_and_analyze,
            inputs=[query_input, search_table_name_input, top_k_slider],
            outputs=[response_output, image_output],
        )

    return demo


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="TextRAG Gradio App")
    parser.add_argument(
        "--share", action="store_true", help="Enable Gradio share feature"
    )
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    demo = create_gradio_interface()
    demo.launch(share=args.share)

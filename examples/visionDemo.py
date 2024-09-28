import gradio as gr
import os
import lancedb
from varag.rag import VisionRAG
from sentence_transformers import SentenceTransformer
from varag.vlms import OpenAI
from dotenv import load_dotenv
import argparse


load_dotenv()

# Initialize shared database
shared_db = lancedb.connect("~/shared_rag_db")

# Initialize VisionRAG and VLM
embedding_model = SentenceTransformer("jinaai/jina-clip-v1", trust_remote_code=True)

vision_rag = VisionRAG(
    image_embedding_model=embedding_model,
    db=shared_db,
    table_name="visionDemo",
)
vlm = OpenAI()


def ingest_pdf(pdf_files, table_name, recursive, verbose):
    try:
        if table_name:
            vision_rag.change_table(table_name)

        file_paths = [pdf_file.name for pdf_file in pdf_files]
        vision_rag.index(
            file_paths, overwrite=False, recursive=recursive, verbose=verbose
        )
        return f"PDFs ingested successfully into table '{vision_rag.table_name}'."
    except Exception as e:
        return f"Error ingesting PDFs: {str(e)}"


def search_and_generate_response(query, table_name):
    try:
        if table_name:
            vision_rag.change_table(table_name)

        results = vision_rag.search(query, k=3)

        images = [result["image"] for result in results]

        print(len(images))

        # Prepare context for VLM
        context = f"Query: {query}\n\nRelevant image information:\n"
        for i, result in enumerate(results, 1):
            context += f"Image {i}: From document '{result['document_name']}', page {result['page_number']}\n"
            if "metadata" in result:
                context += f"Metadata: {result['metadata']}\n"

        response = vlm.query(context, images, max_tokens=500)

        return response, images
    except Exception as e:
        return f"Error generating response: {str(e)}", []


# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# VisionRAG Image Search and Analysis")

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
            search_button = gr.Button("Search and Analyze")
            response_output = gr.Textbox(label="VLM Response")
            image_output = gr.Gallery(label="Retrieved Images")

        ingest_button.click(
            ingest_pdf,
            inputs=[
                pdf_input,
                table_name_input,
                recursive_checkbox,
                verbose_checkbox,
            ],
            outputs=ingest_output,
        )
        search_button.click(
            search_and_generate_response,
            inputs=[query_input, search_table_name_input],
            outputs=[response_output, image_output],
        )

    return demo


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="VisionRAG Gradio App")
    parser.add_argument(
        "--share", action="store_true", help="Enable Gradio share feature"
    )
    return parser.parse_args()


# Launch the app
if __name__ == "__main__":
    args = parse_args()
    app = gradio_interface()
    app.launch(share=args.share)

import gradio as gr
import os
import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List
from PIL import Image
import base64
import io
import time

from varag.rag import SimpleRAG, VisionRAG, ColpaliRAG, HybridColpaliRAG
from varag.vlms import OpenAI
from varag.llms import OpenAI as OpenAILLM
from varag.chunking import FixedTokenChunker

load_dotenv()

# Initialize shared database
shared_db = lancedb.connect("~/shared_rag_db")

# Initialize embedding models
text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)
image_embedding_model = SentenceTransformer(
    "jinaai/jina-clip-v1", trust_remote_code=True
)

# Initialize RAG instances
simple_rag = SimpleRAG(
    text_embedding_model=text_embedding_model, db=shared_db, table_name="simpleDemo"
)
vision_rag = VisionRAG(
    image_embedding_model=image_embedding_model, db=shared_db, table_name="visionDemo"
)
colpali_rag = ColpaliRAG(db=shared_db, table_name="colpaliDemo")
# hybrid_rag = HybridColpaliRAG(
#     image_embedding_model=image_embedding_model, db=shared_db, table_name="hybridDemo"
# )

# Initialize VLM
vlm = OpenAI()
llm = OpenAILLM()


def ingest_data(pdf_files, use_ocr, chunk_size):
    results = []
    file_paths = [pdf_file.name for pdf_file in pdf_files]

    # SimpleRAG
    start_time = time.time()
    simple_rag.index(
        file_paths,
        recursive=False,
        chunking_strategy=FixedTokenChunker(chunk_size=chunk_size),
        metadata={"source": "gradio_upload"},
        overwrite=True,
        verbose=True,
        ocr=use_ocr,
    )
    simple_time = time.time() - start_time
    results.append(
        f"SimpleRAG ingestion complete. Time taken: {simple_time:.2f} seconds"
    )

    # VisionRAG
    start_time = time.time()
    vision_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    vision_time = time.time() - start_time
    results.append(
        f"VisionRAG ingestion complete. Time taken: {vision_time:.2f} seconds"
    )

    # ColpaliRAG
    start_time = time.time()
    colpali_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    colpali_time = time.time() - start_time
    results.append(
        f"ColpaliRAG ingestion complete. Time taken: {colpali_time:.2f} seconds"
    )

    # # HybridColpaliRAG
    # start_time = time.time()
    # hybrid_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    # hybrid_time = time.time() - start_time
    # results.append(
    #     f"HybridColpaliRAG ingestion complete. Time taken: {hybrid_time:.2f} seconds"
    # )

    return "\n".join(results)


def retrieve_data(query, top_k):
    results = {}

    # SimpleRAG
    simple_results = simple_rag.search(query, k=top_k)
    simple_context = "\n".join([r["text"] for r in simple_results])
    simple_response = llm.query(
        context=simple_context,
        system_prompt="Given the below information answer the questions",
        query=query,
    )
    results["SimpleRAG"] = {"response": simple_response, "context": simple_context}

    # VisionRAG
    vision_results = vision_rag.search(query, k=top_k)
    vision_images = [r["image"] for r in vision_results]
    vision_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
        [
            f"Image {i+1}: From document '{r['document_name']}', page {r['page_number']}"
            for i, r in enumerate(vision_results)
        ]
    )
    vision_response = vlm.query(vision_context, vision_images, max_tokens=500)
    results["VisionRAG"] = {
        "response": vision_response,
        "context": vision_context,
        "images": vision_images,
    }

    # ColpaliRAG
    colpali_results = colpali_rag.search(query, k=top_k)
    colpali_images = [r["image"] for r in colpali_results]
    colpali_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
        [
            f"Image {i+1}: From document '{r['document_name']}', page {r['page_number']}"
            for i, r in enumerate(colpali_results)
        ]
    )
    colpali_response = vlm.query(colpali_context, colpali_images, max_tokens=500)
    results["ColpaliRAG"] = {
        "response": colpali_response,
        "context": colpali_context,
        "images": colpali_images,
    }

    # # HybridColpaliRAG
    # hybrid_results = hybrid_rag.search(query, k=top_k, use_image_search=True)
    # hybrid_images = [r["image"] for r in hybrid_results]
    # hybrid_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
    #     [
    #         f"Image {i+1}: From document '{r['name']}', page {r['page_number']}\nText: {r['page_text'][:500]}..."
    #         for i, r in enumerate(hybrid_results)
    #     ]
    # )
    # hybrid_response = vlm.query(hybrid_context, hybrid_images, max_tokens=500)
    # results["HybridColpaliRAG"] = {
    #     "response": hybrid_response,
    #     "context": hybrid_context,
    #     "images": hybrid_images,
    # }

    return results


def update_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    return "API key updated successfully."


def change_table(simple_table, vision_table, colpali_table, hybrid_table):
    simple_rag.change_table(simple_table)
    vision_rag.change_table(vision_table)
    colpali_rag.change_table(colpali_table)
    # hybrid_rag.change_table(hybrid_table)
    return "Table names updated successfully."


def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Combined RAG Approaches Demo")

        with gr.Tab("Ingest Data"):
            pdf_input = gr.File(
                label="Upload PDF(s)", file_count="multiple", file_types=["pdf"]
            )
            use_ocr = gr.Checkbox(label="Use OCR (for SimpleRAG)")
            chunk_size = gr.Slider(
                50, 5000, value=200, step=10, label="Chunk Size (for SimpleRAG)"
            )
            ingest_button = gr.Button("Ingest PDFs")
            ingest_output = gr.Textbox(label="Ingestion Status")

        with gr.Tab("Retrieve Data"):
            query_input = gr.Textbox(label="Enter your query")
            top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top K Results")
            search_button = gr.Button("Search and Analyze")

            with gr.Row():
                simple_response = gr.Markdown(label="SimpleRAG Response")
                vision_response = gr.Markdown(label="VisionRAG Response")
                colpali_response = gr.Markdown(label="ColpaliRAG Response")
                # hybrid_response = gr.Markdown(label="HybridColpaliRAG Response")

            with gr.Row():
                simple_context = gr.Accordion("SimpleRAG Context", open=False)
                with simple_context:
                    gr.Markdown(elem_id="simple_context")

                vision_context = gr.Accordion("VisionRAG Context", open=False)
                with vision_context:
                    gr.Markdown(elem_id="vision_context")
                    gr.Gallery(label="VisionRAG Images")

                colpali_context = gr.Accordion("ColpaliRAG Context", open=False)
                with colpali_context:
                    gr.Markdown(elem_id="colpali_context")
                    gr.Gallery(label="ColpaliRAG Images")

                # hybrid_context = gr.Accordion("HybridColpaliRAG Context", open=False)
                # with hybrid_context:
                #     gr.Markdown(elem_id="hybrid_context")
                #     gr.Gallery(label="HybridColpaliRAG Images")

        with gr.Tab("Settings"):
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            update_api_button = gr.Button("Update API Key")
            api_update_status = gr.Textbox(label="API Update Status")

            simple_table_input = gr.Textbox(
                label="SimpleRAG Table Name", value="simpleDemo"
            )
            vision_table_input = gr.Textbox(
                label="VisionRAG Table Name", value="visionDemo"
            )
            colpali_table_input = gr.Textbox(
                label="ColpaliRAG Table Name", value="colpaliDemo"
            )
            # hybrid_table_input = gr.Textbox(
            #     label="HybridColpaliRAG Table Name", value="hybridDemo"
            # )
            update_table_button = gr.Button("Update Table Names")
            table_update_status = gr.Textbox(label="Table Update Status")

        ingest_button.click(
            ingest_data,
            inputs=[pdf_input, use_ocr, chunk_size],
            outputs=ingest_output,
        )

        search_button.click(
            retrieve_data,
            inputs=[query_input, top_k_slider],
            outputs=[
                simple_response,
                vision_response,
                colpali_response,
                # hybrid_response,
                simple_context,
                vision_context,
                colpali_context,
                # hybrid_context,
            ],
        )

        update_api_button.click(
            update_api_key, inputs=[api_key_input], outputs=api_update_status
        )

        update_table_button.click(
            change_table,
            inputs=[
                simple_table_input,
                vision_table_input,
                colpali_table_input,
                # hybrid_table_input,
            ],
            outputs=table_update_status,
        )

    return demo


if __name__ == "__main__":
    app = gradio_interface()
    app.launch()

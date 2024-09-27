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
from collections import namedtuple
import pandas as pd
import concurrent.futures
from varag.rag import SimpleRAG, VisionRAG, ColpaliRAG, HybridColpaliRAG
from varag.vlms import OpenAI
from varag.llms import OpenAI as OpenAILLM
from varag.chunking import FixedTokenChunker
from varag.utils import get_model_colpali

load_dotenv()

# Initialize shared database
shared_db = lancedb.connect("~/rag_db")

# Initialize embedding models
text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)
image_embedding_model = SentenceTransformer(
    "jinaai/jina-clip-v1", trust_remote_code=True
)
colpali_model, colpali_processor = get_model_colpali("vidore/colpali-v1.2")

# Initialize RAG instances
simple_rag = SimpleRAG(
    text_embedding_model=text_embedding_model, db=shared_db, table_name="simpleDemo"
)
vision_rag = VisionRAG(
    image_embedding_model=image_embedding_model, db=shared_db, table_name="visionDemo"
)
colpali_rag = ColpaliRAG(
    colpali_model=colpali_model,
    colpali_processor=colpali_processor,
    db=shared_db,
    table_name="colpaliDemo",
)
hybrid_rag = HybridColpaliRAG(
    colpali_model=colpali_model,
    colpali_processor=colpali_processor,
    image_embedding_model=image_embedding_model,
    db=shared_db,
    table_name="hybridDemo",
)

# Initialize VLM
vlm = OpenAI()
llm = OpenAILLM()

IngestResult = namedtuple("IngestResult", ["status_text", "progress_table"])


def ingest_data(pdf_files, use_ocr, chunk_size, progress=gr.Progress()):
    file_paths = [pdf_file.name for pdf_file in pdf_files]
    total_start_time = time.time()
    progress_data = []

    # SimpleRAG
    yield IngestResult(
        status_text="Starting SimpleRAG ingestion...\n",
        progress_table=pd.DataFrame(progress_data),
    )
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
    progress_data.append(
        {"Technique": "SimpleRAG", "Time Taken (s)": f"{simple_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"SimpleRAG ingestion complete. Time taken: {simple_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(0.25, desc="SimpleRAG complete")

    # VisionRAG
    yield IngestResult(
        status_text="Starting VisionRAG ingestion...\n",
        progress_table=pd.DataFrame(progress_data),
    )
    start_time = time.time()
    vision_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    vision_time = time.time() - start_time
    progress_data.append(
        {"Technique": "VisionRAG", "Time Taken (s)": f"{vision_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"VisionRAG ingestion complete. Time taken: {vision_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(0.5, desc="VisionRAG complete")

    # ColpaliRAG
    yield IngestResult(
        status_text="Starting ColpaliRAG ingestion...\n",
        progress_table=pd.DataFrame(progress_data),
    )
    start_time = time.time()
    colpali_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    colpali_time = time.time() - start_time
    progress_data.append(
        {"Technique": "ColpaliRAG", "Time Taken (s)": f"{colpali_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"ColpaliRAG ingestion complete. Time taken: {colpali_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(0.75, desc="ColpaliRAG complete")

    # HybridColpaliRAG
    yield IngestResult(
        status_text="Starting HybridColpaliRAG ingestion...\n",
        progress_table=pd.DataFrame(progress_data),
    )
    start_time = time.time()
    hybrid_rag.index(file_paths, overwrite=False, recursive=False, verbose=True)
    hybrid_time = time.time() - start_time
    progress_data.append(
        {"Technique": "HybridColpaliRAG", "Time Taken (s)": f"{hybrid_time:.2f}"}
    )
    yield IngestResult(
        status_text=f"HybridColpaliRAG ingestion complete. Time taken: {hybrid_time:.2f} seconds\n\n",
        progress_table=pd.DataFrame(progress_data),
    )
    # progress(1.0, desc="HybridColpaliRAG complete")

    total_time = time.time() - total_start_time
    progress_data.append({"Technique": "Total", "Time Taken (s)": f"{total_time:.2f}"})
    yield IngestResult(
        status_text=f"Total ingestion time: {total_time:.2f} seconds",
        progress_table=pd.DataFrame(progress_data),
    )


def retrieve_data(query, top_k, sequential=False):
    results = {}
    timings = {}

    def retrieve_simple():
        start_time = time.time()
        simple_results = simple_rag.search(query, k=top_k)

        print(simple_results)

        simple_context = "\n".join([r["text"] for r in simple_results])
        end_time = time.time()
        return "SimpleRAG", simple_context, end_time - start_time

    def retrieve_vision():
        start_time = time.time()
        vision_results = vision_rag.search(query, k=top_k)
        vision_images = [r["image"] for r in vision_results]
        end_time = time.time()
        return "VisionRAG", vision_images, end_time - start_time

    def retrieve_colpali():
        start_time = time.time()
        colpali_results = colpali_rag.search(query, k=top_k)
        colpali_images = [r["image"] for r in colpali_results]
        end_time = time.time()
        return "ColpaliRAG", colpali_images, end_time - start_time

    def retrieve_hybrid():
        start_time = time.time()
        hybrid_results = hybrid_rag.search(query, k=top_k, use_image_search=True)
        hybrid_images = [r["image"] for r in hybrid_results]
        end_time = time.time()
        return "HybridColpaliRAG", hybrid_images, end_time - start_time

    retrieval_functions = [
        retrieve_simple,
        retrieve_vision,
        retrieve_colpali,
        retrieve_hybrid,
    ]

    if sequential:
        for func in retrieval_functions:
            rag_type, content, timing = func()
            results[rag_type] = content
            timings[rag_type] = timing
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(func) for func in retrieval_functions]
            for future in concurrent.futures.as_completed(future_results):
                rag_type, content, timing = future.result()
                results[rag_type] = content
                timings[rag_type] = timing

    return results, timings


def query_data(query, retrieved_results):
    results = {}

    # SimpleRAG
    simple_context = retrieved_results["SimpleRAG"]
    simple_response = llm.query(
        context=simple_context,
        system_prompt="Given the below information answer the questions",
        query=query,
    )
    results["SimpleRAG"] = {"response": simple_response, "context": simple_context}

    # VisionRAG
    vision_images = retrieved_results["VisionRAG"]
    vision_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
        [f"Image {i+1}" for i in range(len(vision_images))]
    )
    vision_response = vlm.query(vision_context, vision_images, max_tokens=500)
    results["VisionRAG"] = {
        "response": vision_response,
        "context": vision_context,
        "images": vision_images,
    }

    # ColpaliRAG
    colpali_images = retrieved_results["ColpaliRAG"]
    colpali_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
        [f"Image {i+1}" for i in range(len(colpali_images))]
    )
    colpali_response = vlm.query(colpali_context, colpali_images, max_tokens=500)
    results["ColpaliRAG"] = {
        "response": colpali_response,
        "context": colpali_context,
        "images": colpali_images,
    }

    # HybridColpaliRAG
    hybrid_images = retrieved_results["HybridColpaliRAG"]
    hybrid_context = f"Query: {query}\n\nRelevant image information:\n" + "\n".join(
        [f"Image {i+1}" for i in range(len(hybrid_images))]
    )
    hybrid_response = vlm.query(hybrid_context, hybrid_images, max_tokens=500)
    results["HybridColpaliRAG"] = {
        "response": hybrid_response,
        "context": hybrid_context,
        "images": hybrid_images,
    }

    return results


def update_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    return "API key updated successfully."


def change_table(simple_table, vision_table, colpali_table, hybrid_table):
    simple_rag.change_table(simple_table)
    vision_rag.change_table(vision_table)
    colpali_rag.change_table(colpali_table)
    hybrid_rag.change_table(hybrid_table)
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
            ingest_output = gr.Markdown(
                label="Ingestion Status :",
            )
            progress_table = gr.DataFrame(
                label="Ingestion Progress", headers=["Technique", "Time Taken (s)"]
            )

        with gr.Tab("Retrieve and Query Data"):
            query_input = gr.Textbox(label="Enter your query")
            top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top K Results")
            sequential_checkbox = gr.Checkbox(label="Sequential Retrieval", value=False)
            retrieve_button = gr.Button("Retrieve")
            query_button = gr.Button("Query")

            retrieval_timing = gr.DataFrame(
                label="Retrieval Timings", headers=["RAG Type", "Time (s)"]
            )

            with gr.Row():
                simple_content = gr.Markdown(label="SimpleRAG Content")
                vision_gallery = gr.Gallery(label="VisionRAG Images")
                colpali_gallery = gr.Gallery(label="ColpaliRAG Images")
                hybrid_gallery = gr.Gallery(label="HybridColpaliRAG Images")

            with gr.Row():
                simple_response = gr.Markdown(label="SimpleRAG Response")
                vision_response = gr.Markdown(label="VisionRAG Response")
                colpali_response = gr.Markdown(label="ColpaliRAG Response")
                hybrid_response = gr.Markdown(label="HybridColpaliRAG Response")

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
            hybrid_table_input = gr.Textbox(
                label="HybridColpaliRAG Table Name", value="hybridDemo"
            )
            update_table_button = gr.Button("Update Table Names")
            table_update_status = gr.Textbox(label="Table Update Status")

        retrieved_results = gr.State({})

        def update_retrieval_results(query, top_k, sequential):
            results, timings = retrieve_data(query, top_k, sequential)
            timing_df = pd.DataFrame(
                list(timings.items()), columns=["RAG Type", "Time (s)"]
            )
            return (
                results["SimpleRAG"],
                results["VisionRAG"],
                results["ColpaliRAG"],
                results["HybridColpaliRAG"],
                timing_df,
                results,
            )

        retrieve_button.click(
            update_retrieval_results,
            inputs=[query_input, top_k_slider, sequential_checkbox],
            outputs=[
                simple_content,
                vision_gallery,
                colpali_gallery,
                hybrid_gallery,
                retrieval_timing,
                retrieved_results,
            ],
        )

        def update_query_results(query, retrieved_results):
            results = query_data(query, retrieved_results)
            return (
                results["SimpleRAG"]["response"],
                results["VisionRAG"]["response"],
                results["ColpaliRAG"]["response"],
                results["HybridColpaliRAG"]["response"],
            )

        query_button.click(
            update_query_results,
            inputs=[query_input, retrieved_results],
            outputs=[
                simple_response,
                vision_response,
                colpali_response,
                hybrid_response,
            ],
        )

        ingest_button.click(
            ingest_data,
            inputs=[pdf_input, use_ocr, chunk_size],
            outputs=[ingest_output, progress_table],
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
                hybrid_table_input,
            ],
            outputs=table_update_status,
        )

    return demo


if __name__ == "__main__":
    app = gradio_interface()
    app.launch()

import gradio as gr
from sentence_transformers import SentenceTransformer
from varag.llms import OpenAI
from varag.rag import SimpleRAG
from varag.chunking import FixedTokenChunker
import lancedb
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)
embedding_model = SentenceTransformer("BAAI/bge-base-en", trust_remote_code=True)
# embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", trust_remote_code=True)
# embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", trust_remote_code=True)

# Initialize shared database
shared_db = lancedb.connect("~/shared_rag_db")

# Initialize TextRAG with shared database
text_rag = SimpleRAG(
    text_embedding_model=embedding_model,
    db=shared_db,
    table_name="default_table",
)

# Initialize OpenAI client
llm = OpenAI()


def ingest_documents(files, chunk_size, use_ocr):
    pdf_paths = [file.name for file in files]
    result = text_rag.index(
        pdf_paths,
        recursive=False,
        chunking_strategy=FixedTokenChunker(chunk_size=chunk_size),
        metadata={"source": "gradio_upload"},
        overwrite=True,
        verbose=True,
        ocr=use_ocr,
    )
    return f"Ingestion complete. {result}"


def query_and_answer(query, num_results):
    # Search for relevant chunks
    search_results = text_rag.search(query, k=num_results)

    # Generate response using OpenAI
    context = "\n".join([r["text"] for r in search_results])
    response = llm.query(
        context=context,
        system_prompt="Given the below information answer the questions",
        query=query,
    )

    # Format the results
    formatted_results = "\n\n".join(
        [
            f"{'==='*50}\n\n\nChunk {i+1}:\n{r['text']}{r['chunk_index']}{r['document_name']}\n\n\n{'==='*50}"
            for i, r in enumerate(search_results)
        ]
    )

    return formatted_results, response


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# TextRAG Document Ingestion and Query System")

    with gr.Tab("Ingest Documents"):
        file_input = gr.File(
            file_count="multiple", label="Upload PDF Documents", file_types=["pdf"]
        )
        chunk_size = gr.Slider(50, 5000, value=200, step=10, label="Chunk Size")
        use_ocr = gr.Checkbox(label="Use OCR")
        ingest_button = gr.Button("Ingest Documents")
        ingest_output = gr.Textbox(label="Ingestion Result")

        ingest_button.click(
            ingest_documents,
            inputs=[file_input, chunk_size, use_ocr],
            outputs=ingest_output,
        )

    with gr.Tab("Query Documents"):
        query_input = gr.Textbox(label="Enter your query")
        num_results = gr.Slider(
            1, 10, value=5, step=1, label="Number of results to retrieve"
        )
        query_button = gr.Button("Search and Answer")
        retrieved_chunks = gr.Textbox(label="Retrieved Chunks")
        answer_output = gr.Textbox(label="Generated Answer")

        query_button.click(
            query_and_answer,
            inputs=[query_input, num_results],
            outputs=[retrieved_chunks, answer_output],
        )

demo.launch()

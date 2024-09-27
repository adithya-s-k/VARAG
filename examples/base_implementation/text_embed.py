import gradio as gr
import json
import logging
from pathlib import Path
from typing import List, Callable, Optional
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModel
import lancedb
import openai
from abc import ABC, abstractmethod
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_with_pymupdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def ingest_pdfs(pdf_paths: List[str], use_docling: bool = False) -> List[str]:
    if use_docling:
        from docling.datamodel.base_models import ConversionStatus
        from docling.datamodel.document import ConversionResult, DocumentConversionInput
        from docling.document_converter import DocumentConverter

        doc_converter = DocumentConverter()
        input_doc_paths = [Path(path) for path in pdf_paths]
        input_data = DocumentConversionInput.from_paths(input_doc_paths)

        conv_results = doc_converter.convert(input_data)

        markdown_list = []
        success_count = 0
        partial_success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                markdown_content = conv_res.render_as_markdown()
                markdown_list.append(markdown_content)
                success_count += 1
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                logger.info(
                    f"Document {conv_res.input.file} was partially converted with the following errors:"
                )
                for item in conv_res.errors:
                    logger.info(f"\t{item.error_message}")
                markdown_content = conv_res.render_as_markdown()
                markdown_list.append(markdown_content)
                partial_success_count += 1
            else:
                logger.info(f"Document {conv_res.input.file} failed to convert.")
                failure_count += 1

        logger.info(
            f"Processed {success_count + partial_success_count + failure_count} docs, "
            f"of which {failure_count} failed "
            f"and {partial_success_count} were partially converted."
        )

        if failure_count > 0:
            logger.warning(
                f"Failed to convert {failure_count} out of {len(input_doc_paths)} documents."
            )

        return markdown_list
    else:
        return [extract_text_with_pymupdf(pdf_path) for pdf_path in pdf_paths]


class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass


class TextSplitter(BaseChunker, ABC):
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len > self._chunk_size:
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total}, which is longer than the specified {self._chunk_size}"
                    )
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (
                        total + _len > self._chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0])
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text


class FixedTokenChunker(TextSplitter):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> List[str]:
        tokens = self._tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self._chunk_size - self._chunk_overlap):
            chunk = self._tokenizer.decode(tokens[i : i + self._chunk_size])
            chunks.append(chunk)
        return chunks


def embedder(chunk):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**tokens)
    embeddings = model_output.last_hidden_state[:, 0, :]
    embed = embeddings[0].numpy()
    return embed


def prepare_data(chunks, embeddings):
    data = []
    for chunk, embed in zip(chunks, embeddings):
        temp = {"text": chunk, "vector": embed}
        data.append(temp)
    return data


def lanceDBConnection(chunks, embeddings):
    db = lancedb.connect("/tmp/lancedb")
    data = prepare_data(chunks, embeddings)
    table = db.create_table("scratch", data=data, exist_ok=True)
    return table


base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using the provided contexts. Every answer you generate should have citations in this pattern  "Answer [position].", for example: "Earth is round [1][2].," if it's relevant.

Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

User question: {}

Contexts:
{}
"""


def process_pdf(pdf_file, use_docling):
    pdf_path = pdf_file.name
    markdown_results = ingest_pdfs([pdf_path], use_docling)
    chunks = FixedTokenChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunks.split_text(markdown_results[0])

    embeds = [embedder(chunk) for chunk in chunks]

    global table
    table = lanceDBConnection(chunks, embeds)

    return "PDF processed and stored in LanceDB."


def answer_question(question):
    query_embedding = embedder(question)
    result = table.search(query_embedding).limit(5).to_list()
    context = [r["text"] for r in result]

    prompt = base_prompt.format(question, context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content

    return "\n\nRetrieved Context:\n" + "\n".join(context) + "\n\nAnswer:\n" + answer


# Gradio Interface
def gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("# PDF Question Answering System")

        with gr.Tab("Upload PDF"):
            pdf_input = gr.File(label="Upload PDF")
            use_docling = gr.Checkbox(
                label="Use docling for PDF processing", value=False
            )
            process_button = gr.Button("Process PDF")
            pdf_output = gr.Textbox(label="Processing Status")

            process_button.click(
                process_pdf, inputs=[pdf_input, use_docling], outputs=pdf_output
            )

        with gr.Tab("Ask Questions"):
            question_input = gr.Textbox(label="Enter your question")
            answer_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer")

            answer_button.click(
                answer_question, inputs=question_input, outputs=answer_output
            )

    return app


# Launch the app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch()

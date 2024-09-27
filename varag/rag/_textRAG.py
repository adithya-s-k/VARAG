import os
import json
import logging
from typing import List, Dict, Any, Union
import torch
import lancedb
import pyarrow as pa
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from varag.chunking import BaseChunker, FixedTokenChunker

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextRAG:
    def __init__(
        self,
        text_embedding_model: SentenceTransformer,
        db_path: str = "~/textrag_db",
        table_name: str = "text_rag_table",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.db_path = os.path.expanduser(db_path)
        self.table_name = table_name
        self.db = lancedb.connect(self.db_path)
        self.schema = pa.schema(
            [
                pa.field("document_name", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("text", pa.string()),
                pa.field(
                    "vector", pa.list_(pa.float32(), 384)
                ),  # Dimension for all-MiniLM-L6-v2
                pa.field("metadata", pa.string()),
            ]
        )
        self.table = self.db.create_table(
            self.table_name, schema=self.schema, mode="overwrite"
        )

        self.text_embedding_model = text_embedding_model.to(self.device)

        self.openai_client = OpenAI()

    def extract_text_with_pymupdf(self, pdf_path):
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def index(
        self,
        path: Union[str, List[str]],
        recursive: bool = False,
        chunking_strategy: "BaseChunker" = None,
        metadata: Dict[str, str] = None,
        overwrite: bool = False,
        verbose: bool = True,
        ocr: bool = False,
    ):
        if overwrite:
            self.table = self.db.create_table(
                self.table_name, schema=self.schema, mode="overwrite"
            )

        if isinstance(path, str):
            paths = [path]
        else:
            paths = path

        chunker = chunking_strategy if chunking_strategy else FixedTokenChunker()

        for path in paths:
            if os.path.isfile(path):
                self._process_file(path, chunker, metadata, verbose, ocr)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(".pdf"):
                            file_path = os.path.join(root, file)
                            self._process_file(
                                file_path, chunker, metadata, verbose, ocr
                            )
                    if not recursive:
                        break

        return f"Indexing complete. Total documents in {self.table_name}: {len(self.table)}"

    def _process_file(
        self,
        file_path: str,
        chunker: "BaseChunker",
        metadata: Dict[str, str],
        verbose: bool,
        ocr: bool,
    ):
        if ocr:
            try:
                from docling.datamodel.document import DocumentConversionInput
                from docling.document_converter import DocumentConverter

                doc_converter = DocumentConverter()
                input_data = DocumentConversionInput.from_paths([Path(file_path)])
                conv_results = doc_converter.convert(input_data)

                if (
                    conv_results[0].status == "SUCCESS"
                    or conv_results[0].status == "PARTIAL_SUCCESS"
                ):
                    text = conv_results[0].render_as_markdown()
                else:
                    logger.info(f"Document {file_path} failed to convert.")
                    return
            except ImportError:
                logger.error(
                    "OCR functionality requires additional dependencies. "
                    "Please install them with: pip install varag[ocr]"
                )
                raise ImportError(
                    "OCR dependencies not installed. " "Run: pip install varag[ocr]"
                )
        else:
            text = self.extract_text_with_pymupdf(file_path)

        chunks = chunker.split_text(text)
        embeddings = self.text_embedding_model.encode(chunks, show_progress_bar=verbose)

        data = [
            {
                "document_name": os.path.basename(file_path),
                "chunk_index": i,
                "text": chunk,
                "vector": embedding.tolist(),
                "metadata": json.dumps(metadata or {}),
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        self.table.add(data)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.text_embedding_model.encode([query])[0]
        results = (
            self.table.search(
                query_embedding.tolist(),
                query_type="vector",
                vector_column_name="vector",
            )
            .limit(k)
            .to_list()
        )
        for result in results:
            result["metadata"] = json.loads(result["metadata"])
        return results

    def add_to_index(
        self,
        path: Union[str, List[str]],
        recursive: bool = False,
        chunking_strategy: "BaseChunker" = None,
        metadata: Dict[str, str] = None,
        verbose: bool = True,
        ocr: bool = False,
    ):
        return self.index(
            path,
            recursive,
            chunking_strategy,
            metadata,
            overwrite=False,
            verbose=verbose,
            ocr=ocr,
        )

    def generate_response(self, query: str, k: int = 5) -> str:
        results = self.search(query, k)
        context = [r["text"] for r in results]

        base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using the provided contexts. Every answer you generate should have citations in this pattern  "Answer [position].", for example: "Earth is round [1][2].," if it's relevant.

Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

User question: {question}

Contexts:
{context}
"""

        prompt = base_prompt.format(question=query, context="\n".join(context))
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )

        answer = response.choices[0].message.content
        return (
            "\n\nRetrieved Context:\n" + "\n".join(context) + "\n\nAnswer:\n" + answer
        )

    def change_table(self, new_table_name: str):
        self.table_name = new_table_name
        self.table = self.db.create_table(
            self.table_name, schema=self.schema, mode="overwrite"
        )
        print(f"Switched to new table: {self.table_name}")


# Usage example:
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)

    # Initialize TextRAG
    text_rag = TextRAG(
        text_embedding_model=embedding_model,
        db_path="~/visionrag_db",
        table_name="default_table",
    )

    # Initialize OpenAI client
    llm = OpenAI()

    # Index PDFs
    pdf_paths = ["path/to/pdf1.pdf", "path/to/pdf2.pdf"]
    result = text_rag.index(
        pdf_paths,
        recursive=True,
        chunking_strategy=FixedTokenChunker(chunk_size=500),
        metadata={"source": "example_data"},
        overwrite=True,
        verbose=True,
        ocr=True,
    )
    print(result)

    # Search for relevant chunks
    query = "What is the main topic of the documents?"
    search_results = text_rag.search(query, k=5)
    print("Search results:", search_results)

    # Generate response using OpenAI
    context = "\n".join([r["text"] for r in search_results])
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
        ],
    )
    print("Generated response:", response.choices[0].message.content)

    # Add new documents to the index
    new_pdf_paths = ["path/to/new_pdf.pdf"]
    result = text_rag.add_to_index(
        new_pdf_paths,
        recursive=False,
        chunking_strategy=FixedTokenChunker(chunk_size=500),
        metadata={"project": "new_project"},
        verbose=True,
        ocr=False,
    )
    print(result)

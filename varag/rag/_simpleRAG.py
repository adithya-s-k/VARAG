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


class SimpleRAG:
    def __init__(
        self,
        text_embedding_model: SentenceTransformer,
        db: Union[lancedb.connect, None] = None,
        db_path: str = "~/textrag_db",
        table_name: str = "text_rag_table",
        overwrite: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if db is None:
            self.db_path = os.path.expanduser(db_path)
            self.db = lancedb.connect(self.db_path)
        else:
            self.db = db

        self.table_name = table_name

        # Get the output dimension of the embedding model
        self.vector_dim = text_embedding_model.get_sentence_embedding_dimension()

        self.schema = pa.schema(
            [
                pa.field("document_name", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
                pa.field("metadata", pa.string()),
            ]
        )
        if overwrite:
            self.table = self.db.create_table(
                self.table_name, schema=self.schema, mode="overwrite"
            )
        else:
            self.table = self.db.create_table(
                self.table_name, schema=self.schema, exist_ok=True
            )

        self.text_embedding_model = text_embedding_model.to(self.device)

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
        else:
            self.table = self.db.create_table(
                self.table_name, schema=self.schema, exist_ok=True
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

    # def _process_file(
    #     self,
    #     file_path: str,
    #     chunker: "BaseChunker",
    #     metadata: Dict[str, str],
    #     verbose: bool,
    #     ocr: bool,
    # ):
    #     if ocr:
    #         try:
    #             from docling.datamodel.document import DocumentConversionInput
    #             from docling.document_converter import DocumentConverter

    #             doc_converter = DocumentConverter()
    #             input_data = DocumentConversionInput.from_paths([Path(file_path)])
    #             conv_results = doc_converter.convert(input_data)

    #             if (
    #                 conv_results[0].status == "SUCCESS"
    #                 or conv_results[0].status == "PARTIAL_SUCCESS"
    #             ):
    #                 text = conv_results[0].render_as_markdown()
    #             else:
    #                 logger.info(f"Document {file_path} failed to convert.")
    #                 return
    #         except ImportError:
    #             logger.error(
    #                 "OCR functionality requires additional dependencies. "
    #                 "Please install them with: pip install varag[ocr]"
    #             )
    #             raise ImportError(
    #                 "OCR dependencies not installed. " "Run: pip install varag[ocr]"
    #             )
    #     else:
    #         text = self.extract_text_with_pymupdf(file_path)

    #     chunks = chunker.split_text(text)
    #     embeddings = self.text_embedding_model.encode(chunks, show_progress_bar=verbose)

    #     data = [
    #         {
    #             "document_name": os.path.basename(file_path),
    #             "chunk_index": i,
    #             "text": chunk,
    #             "vector": embedding.tolist(),
    #             "metadata": json.dumps(metadata or {}),
    #         }
    #         for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    #     ]

    #     self.table.add(data)
    def _process_file(
        self,
        file_path: str,
        chunker: "BaseChunker",
        metadata: Dict[str, str],
        verbose: bool,
        ocr: bool,
    ):
        if ocr:
            logger.info(f"Using OCR for file: {file_path}")
            try:
                from docling.datamodel.document import DocumentConversionInput
                from docling.document_converter import DocumentConverter

                doc_converter = DocumentConverter()
                input_data = DocumentConversionInput.from_paths([Path(file_path)])
                logger.info(f"Starting OCR conversion for file: {file_path}")
                conv_results = doc_converter.convert(input_data)

                # Convert generator to list and process the first item
                conv_result = next(conv_results, None)
                if conv_result is None:
                    raise ValueError("No conversion results")

                text = conv_result.render_as_markdown()
                logger.info(
                    f"OCR conversion completed for file: {file_path}. Status: {conv_result.status}"
                )

            except ImportError:
                logger.error(
                    "OCR functionality requires additional dependencies. "
                    "Please install them with: pip install varag[ocr]"
                )
                raise ImportError(
                    "OCR dependencies not installed. Run: pip install varag[ocr]"
                )
            except Exception as e:
                logger.error(
                    f"OCR conversion failed for file: {file_path}. Error: {str(e)}"
                )
                return
        else:
            logger.info(f"Extracting text without OCR for file: {file_path}")
            text = self.extract_text_with_pymupdf(file_path)

        logger.info(f"Chunking text for file: {file_path}")
        chunks = chunker.split_text(text)
        logger.info(f"Generated {len(chunks)} chunks for file: {file_path}")

        logger.info(f"Generating embeddings for file: {file_path}")
        embeddings = self.text_embedding_model.encode(chunks, show_progress_bar=verbose)
        logger.info(f"Generated {len(embeddings)} embeddings for file: {file_path}")

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

        logger.info(f"Adding {len(data)} entries to the database for file: {file_path}")
        self.table.add(data)
        logger.info(f"Successfully processed file: {file_path}")

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

    def change_table(self, new_table_name: str):
        self.table_name = new_table_name
        self.table = self.db.create_table(
            self.table_name, schema=self.schema, exist_ok=True
        )
        print(f"Switched to new table: {self.table_name}")


# Usage example:
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)

    # Initialize TextRAG
    text_rag = SimpleRAG(
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

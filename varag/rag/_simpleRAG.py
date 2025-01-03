import os
import json
import uuid
import logging
from typing import List, Dict, Any, Union, Tuple
import torch
import lancedb
import pyarrow as pa
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from varag.chunking import BaseChunker, FixedTokenChunker
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


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
                pa.field("chunk_id", pa.string()),  # Added chunk_id field
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
                "chunk_id": str(uuid.uuid4()),  # Generate unique ID for each chunk
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
    def generate_eval_dataset(
        self, 
        num_samples: int = 100, 
        openai_model: str = "gpt-4",
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate an evaluation dataset by sampling chunks and creating QA pairs.
        
        Args:
            num_samples: Number of samples to generate
            openai_model: OpenAI model to use for generation
            temperature: Temperature for generation
            
        Returns:
            List of dictionaries containing evaluation data
        """
        client = OpenAI()
        
        # Sample random chunks from the database
        total_chunks = len(self.table)
        if total_chunks < num_samples:
            logger.warning(f"Requested {num_samples} samples but only {total_chunks} chunks available")
            num_samples = total_chunks
            
        sampled_chunks = self.table.take(np.random.choice(total_chunks, num_samples, replace=False))
        
        eval_dataset = []
        
        for chunk in tqdm(sampled_chunks, desc="Generating evaluation samples"):
            # Create prompt for GPT to generate question and answer
            prompt = f"""Given the following text, generate a relevant question and its corresponding answer.
            The question should be specific enough that this text chunk contains the complete answer.
            
            Text: {chunk['text']}
            
            Output the response in the following JSON format:
            {{
                "question": "the generated question",
                "ground_truth": "the answer that can be found in the text"
            }}
            """
            
            try:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                
                qa_pair = json.loads(response.choices[0].message.content)
                
                eval_dataset.append({
                    "chunk_id": chunk["chunk_id"],
                    "chunk_content": chunk["text"],
                    "question": qa_pair["question"],
                    "ground_truth": qa_pair["ground_truth"]
                })
                
            except Exception as e:
                logger.error(f"Error generating QA pair: {str(e)}")
                continue
                
        return eval_dataset


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

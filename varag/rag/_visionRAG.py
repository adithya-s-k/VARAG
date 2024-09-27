import json
import os
import io
from typing import List, Dict, Any, Union
import torch
import lancedb
import pyarrow as pa
from PIL import Image
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import fitz  # PyMuPDF
from tqdm import tqdm


class VisionRAG:
    def __init__(
        self,
        image_embedding_model: SentenceTransformer,
        db: Union[lancedb.connect, None] = None,
        db_path: str = "~/lancedb",
        table_name: str = "vision_rag_table",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_embedding_model = image_embedding_model.to(self.device)
        print(f"Using device: {self.device}")

        if db is None:
            self.db_path = os.path.expanduser(db_path)
            self.db = lancedb.connect(self.db_path)
        else:
            self.db = db

        self.table_name = table_name

        # Get the output dimension of the embedding model
        self.vector_dim = self.image_embedding_model.get_sentence_embedding_dimension()
        print(self.vector_dim)

        self.schema = pa.schema(
            [
                pa.field("document_name", pa.string()),
                pa.field("page_number", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), 768)),
                pa.field("image", pa.binary()),
                pa.field("metadata", pa.string()),
            ]
        )
        self.table = self.db.create_table(
            self.table_name, schema=self.schema, mode="overwrite"
        )

    def change_table(self, new_table_name: str):
        self.table_name = new_table_name
        self.table = self.db.create_table(
            self.table_name, schema=self.schema, mode="overwrite"
        )
        print(f"Switched to new table: {self.table_name}")

    def pdf_to_images(self, pdf_path: str, verbose: bool = False) -> List[Image.Image]:
        try:
            if verbose:
                print(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(pdf_path)
            if verbose:
                print(f"Converted {len(images)} pages")
            return images
        except PDFInfoNotInstalledError:
            print("Poppler not installed. Falling back to PyMuPDF.")
            pdf_document = fitz.open(pdf_path)
            images = []
            for page in tqdm(
                pdf_document, desc="Converting PDF pages", disable=not verbose
            ):
                img = Image.frombytes(
                    "RGB",
                    [int(page.rect.width), int(page.rect.height)],
                    page.get_pixmap().samples,
                )
                images.append(img)
            return images

    def pil_to_bytes(self, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def embed_images(self, images: List[Image.Image], verbose: bool = False):
        if verbose:
            print(f"Embedding {len(images)} images")
        with torch.no_grad():
            embeddings = self.image_embedding_model.encode(
                images, device=self.device, show_progress_bar=verbose
            )
        return embeddings

    def process_file(
        self, file_path: str, metadata: Dict[str, str] = None, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        file_name = os.path.basename(file_path)
        if verbose:
            print(f"Processing file: {file_name}")

        if file_path.lower().endswith(".pdf"):
            images = self.pdf_to_images(file_path, verbose)
        else:
            images = [Image.open(file_path)]

        embeddings = self.embed_images(images, verbose)

        if verbose:
            print("Converting images to bytes")
        image_bytes = [
            self.pil_to_bytes(img) for img in tqdm(images, disable=not verbose)
        ]

        return [
            {
                "document_name": file_name,
                "page_number": i,
                "vector": emb.tolist(),
                "image": img_bytes,
                "metadata": json.dumps(metadata or {}),
            }
            for i, (emb, img_bytes) in enumerate(zip(embeddings, image_bytes))
        ]

    def index(
        self,
        data_path: Union[str, List[str]],
        overwrite: bool = False,
        recursive: bool = False,
        metadata: Dict[str, str] = None,
        verbose: bool = False,
    ):
        if overwrite:
            self.table = self.db.create_table(
                self.table_name, schema=self.schema, mode="overwrite"
            )

        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path

        total_files = 0
        for path in data_paths:
            if os.path.isfile(path):
                total_files += 1
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    total_files += len(
                        [
                            f
                            for f in files
                            if f.lower().endswith(
                                (".pdf", ".jpg", ".jpeg", ".png", ".gif")
                            )
                        ]
                    )
                    if not recursive:
                        break

        with tqdm(
            total=total_files, desc="Indexing files", disable=not verbose
        ) as pbar:
            for path in data_paths:
                if os.path.isfile(path):
                    data = self.process_file(path, metadata, verbose)
                    self.table.add(data)
                    pbar.update(1)
                elif os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for file in files:
                            if file.lower().endswith(
                                (".pdf", ".jpg", ".jpeg", ".png", ".gif")
                            ):
                                file_path = os.path.join(root, file)
                                data = self.process_file(file_path, metadata, verbose)
                                self.table.add(data)
                                pbar.update(1)
                        if not recursive:
                            break  # Stop after processing the top-level directory

        print(
            f"Indexing complete. Total documents in {self.table_name}: {len(self.table)}"
        )

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        with torch.no_grad():
            query_embedding = self.image_embedding_model.encode(
                [query], device=self.device
            ).squeeze()

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
            result["image"] = Image.open(io.BytesIO(result["image"]))
        return results

    def add_to_index(
        self,
        new_data_path: Union[str, List[str]],
        recursive: bool = False,
        metadata: Dict[str, str] = None,
        verbose: bool = False,
    ):
        self.index(
            new_data_path,
            overwrite=False,
            recursive=recursive,
            metadata=metadata,
            verbose=verbose,
        )

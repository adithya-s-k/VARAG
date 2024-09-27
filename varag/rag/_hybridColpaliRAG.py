import json
import os
import io
from typing import List, Dict, Any, Union, TypeVar, Optional, cast
import PIL
import torch
import lancedb
import pyarrow as pa
from PIL import Image
from transformers import AutoModel, AutoProcessor
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import fitz  # PyMuPDF
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import base64
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from varag.utils import get_model_colpali

T = TypeVar("T")


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


class HybridColpaliRAG:
    def __init__(
        self,
        image_embedding_model: SentenceTransformer,
        colpali_model: Optional[ColPali] = None,
        colpali_processor: Optional[ColPaliProcessor] = None,
        model_name: str = "vidore/colpali-v1.2",
        db: Union[lancedb.connect, None] = None,
        db_path: str = "~/lancedb",
        table_name: str = "hybrid_colpali_rag_table",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.image_embedding_model = image_embedding_model.to(self.device)

        if colpali_model is None or colpali_processor is None:
            self.colpali_model, self.colpali_processor = get_model_colpali(model_name)
        else:
            self.colpali_model = colpali_model
            self.colpali_processor = colpali_processor

        if db is None:
            self.db_path = os.path.expanduser(db_path)
            self.db = lancedb.connect(self.db_path)
        else:
            self.db = db

        self.table_name = table_name

        self.schema = pa.schema(
            [
                pa.field("document_name", pa.string()),
                pa.field("page_number", pa.int32()),
                pa.field("image_vector", pa.list_(pa.float32(), 768)),
                pa.field("image", pa.string()),
                pa.field("page_text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("page_embedding_shape", pa.list_(pa.int64())),
                pa.field(
                    "page_embedding_flatten",
                    pa.list_(pa.float32()),
                ),
            ]
        )
        self.table = self.db.create_table(
            self.table_name, schema=self.schema, exist_ok=True
        )

    def embed_images(self, images: List[Image.Image], verbose: bool = False):
        if verbose:
            print(f"Embedding {len(images)} images")
        with torch.no_grad():
            embeddings = self.image_embedding_model.encode(
                images, device=self.device, show_progress_bar=verbose
            )
        return embeddings

    def change_table(self, new_table_name: str):
        self.table_name = new_table_name
        self.table = self.db.create_table(
            self.table_name, schema=self.schema, exist_ok=True
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

    def embed_images_colpali(self, images: List[Image.Image], verbose: bool = False):
        if verbose:
            print(f"Embedding {len(images)} images with ColPali")
        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self.colpali_processor.process_images(x),
        )
        embeddings = []
        for batch in tqdm(dataloader, disable=not verbose):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                embedding = self.colpali_model(**batch)
            embeddings.extend(list(torch.unbind(embedding.cpu())))
        return embeddings

    def pil_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def base64_to_pil(self, base64_str: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(base64_str)))

    def process_file(
        self, file_path: str, metadata: Dict[str, str] = None, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        file_name = os.path.basename(file_path)
        if verbose:
            print(f"Processing file: {file_name}")

        if file_path.lower().endswith(".pdf"):
            images = self.pdf_to_images(file_path, verbose)
            texts = self.extract_text_from_pdf(file_path)
        else:
            images = [Image.open(file_path)]
            texts = [""]  # No text for single image files

        colpali_embeddings = self.embed_images_colpali(images, verbose)
        image_embeddings = self.embed_images(images, verbose)

        if verbose:
            print("Converting images to base64")
        image_base64 = [
            self.pil_to_base64(img) for img in tqdm(images, disable=not verbose)
        ]

        return [
            {
                "document_name": file_name,
                "page_number": i,
                "page_embedding_flatten": colpali_emb.float()
                .numpy()
                .flatten()
                .tolist(),
                "page_embedding_shape": colpali_emb.shape,
                "image_vector": image_emb.tolist(),
                "image": img_base64,
                "page_text": text,
                "metadata": json.dumps(metadata or {}),
            }
            for i, (colpali_emb, image_emb, img_base64, text) in enumerate(
                zip(colpali_embeddings, image_embeddings, image_base64, texts)
            )
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        reader = fitz.open(pdf_path)
        texts = []
        for page in reader:
            texts.append(page.get_text())
        return texts

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
        else:
            self.table = self.db.create_table(
                self.table_name, schema=self.schema, exist_ok=True
            )

        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path

        total_files = sum(
            1
            for path in data_paths
            for root, _, files in os.walk(path)
            for file in files
            if file.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif"))
        )

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
                            break

        print(
            f"Indexing complete. Total documents in {self.table_name}: {len(self.table)}"
        )

    def flatten_and_zero_pad(tensor, desired_length):
        """Flattens a PyTorch tensor and zero-pads it to a desired length.

        Args:
            tensor: The input PyTorch tensor.
            desired_length: The desired length of the flattened tensor.

        Returns:
            The flattened and zero-padded tensor.
        """

        # Flatten the tensor
        flattened_tensor = tensor.view(-1)

        # Calculate the padding length
        padding_length = desired_length - flattened_tensor.size(0)

        # Check if padding is needed
        if padding_length > 0:
            # Zero-pad the tensor
            padded_tensor = torch.cat(
                [flattened_tensor, torch.zeros(padding_length, dtype=tensor.dtype)],
                dim=0,
            )
        else:
            # Truncate the tensor if it's already too long
            padded_tensor = flattened_tensor[:desired_length]

        return padded_tensor

    def get_query_embedding(self, query: str, model, processor):
        dataloader = DataLoader(
            dataset=ListDataset[str]([query]),
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: processor.process_queries(x),
        )

        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_query = model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        q = {"query": query, "embeddings": qs}
        return q

    def embed_text(self, text: str):
        text_features = self.image_embedding_model.encode([text], device=self.device)
        return text_features.squeeze()  # This is already a numpy array

    def process_patch_embeddings(self, x):
        patches = np.reshape(x["page_embedding_flatten"], x["page_embedding_shape"])
        unflattended_embeddinged = torch.from_numpy(patches).to(torch.bfloat16)

        return unflattended_embeddinged

    def search(
        self,
        query: str,
        k: int = 3,
        use_image_search: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with torch.no_grad():
            qs = self.get_query_embedding(
                query, model=self.colpali_model, processor=self.colpali_processor
            )
            query_embedding = self.embed_text(query)

            if use_image_search:
                limit = limit or 100

                print("using Image Seach for Initial Document Retrival")

                r = self.table.search(
                    query_embedding.tolist(),
                    query_type="vector",
                    vector_column_name="image_vector",
                )
            else:
                r = self.table.search().limit(limit)

        r = r.to_list()

        print("Total Document Image Retived Pre Colpali ReRanking:", len(r))

        all_pages_embeddings = [self.process_patch_embeddings(x) for x in r]

        scores = (
            self.colpali_processor.score(qs["embeddings"], all_pages_embeddings)
            .cpu()
            .numpy()
        )

        scores_tensor = torch.from_numpy(scores)

        # Use topk on the tensor
        top_k_indices = torch.topk(
            scores_tensor, k=min(k, scores_tensor.shape[1]), dim=1
        ).indices

        results = []
        for idx in top_k_indices[0].tolist():  # Convert indices to list
            try:
                page = r[idx]
                pil_image = self.base64_to_pil(page["image"])
                result = {
                    "name": page["document_name"],
                    "page_number": page["page_number"],
                    "image": pil_image,
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")
                continue

        return results

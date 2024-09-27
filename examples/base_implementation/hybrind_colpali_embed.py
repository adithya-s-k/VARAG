import base64
import io

import shutil
import os
import pyarrow as pa
import pandas as pd
from typing import Optional, List, cast, Union
import tempfile
import lancedb
import numpy as np
import PIL
import PIL.Image
from PIL import Image
import requests
import torch
import gradio as gr
from typing import List, TypeVar

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

from tqdm import tqdm
from pdf2image import convert_from_path
from pypdf import PdfReader
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import AutoModel
from dotenv import load_dotenv
from varag.vlms import OpenAI
from torch.utils.data import Dataset

MODEL_ID = "jinaai/jina-clip-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
jina_clip_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)

T = TypeVar("T")


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


load_dotenv()

# Initialize the VLM
vlm = OpenAI()


def embed_images(images: List[Image.Image]):
    image_features = jina_clip_model.encode_image(images)
    return image_features  # This is already a numpy array


def embed_text(text: str):
    text_features = jina_clip_model.encode_text([text])
    return text_features.squeeze()  # This is already a numpy array


def get_model_colpali(base_model_id: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "vidore/colpali-v1.2"
    if base_model_id is None:
        base_model_id = "google/paligemma-3b-mix-448"
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))
    return model, processor


model, processor = get_model_colpali()


def base64_to_pil(base64_str: str) -> PIL.Image.Image:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = PIL.Image.open(io.BytesIO(image_data))
    return image


def get_base64_image(image: Union[str, Image.Image]):
    if isinstance(image, str):
        # If image is a file path, open it
        with Image.open(image) as img:
            img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
    elif isinstance(image, Image.Image):
        # If image is already a PIL Image object
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
    else:
        raise ValueError(
            "Input must be either a file path string or a PIL Image object"
        )

    # Get the byte data and encode it to base64
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode()

    return f"data:image/png;base64,{img_base64}"


def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)

    images = convert_from_path(pdf_path)
    assert len(images) == len(page_texts)
    return (images, page_texts)


def get_pdf_embedding(pdf_path: str, model, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    page_images, page_texts = get_pdf_images(pdf_path=pdf_path)
    page_embeddings = []
    dataloader = DataLoader(
        dataset=ListDataset[str](page_images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    i = 0
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        page_embeddings = list(torch.unbind(embeddings_doc.to("cpu")))
        for page_embedding in page_embeddings:
            document = {
                "name": pdf_path,
                "page_idx": i,
                "page_image": page_images[i],
                "page_text": page_texts[i],
                "page_embedding": page_embedding,
            }
            i += 1

            yield document


def get_query_embedding(query: str, model, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def embedd_docs(pdf_path, model, processor):
    """
    Returns embeddings for all pages in a single PDF document.

    Args:
        pdf_path (str): Path to the PDF document.
        model (ColPali): Model instance.
        processor (AutoProcessor): Processor instance.

    Yields:
        dict: A dictionary containing embedding and metadata for each page.
    """
    if not os.path.isfile(pdf_path):
        raise ValueError(f"The path '{pdf_path}' is not a valid file.")

    print(f"Processing PDF: {pdf_path}")
    pdf_doc = get_pdf_embedding(pdf_path=pdf_path, model=model, processor=processor)
    for batch in tqdm(pdf_doc, desc="Processing pages"):
        yield batch


def create_db(docs_storage, table_name: str = "demo", db_path: str = "lancedb"):
    def _gen():
        for x in docs_storage:
            image_embeddings = embed_images([x["page_image"]])
            page_embedding_flatten = (
                x["page_embedding"].float().numpy().flatten().tolist()
            )
            yield [
                {
                    "name": x["name"],
                    "page_text": x["page_text"],
                    "image": get_base64_image(x["page_image"]),
                    "image_vectors": image_embeddings[0],
                    "page_idx": x["page_idx"],
                    "page_embedding_flatten": page_embedding_flatten,
                    "page_embedding_shape": list(x["page_embedding"].shape),
                }
            ]

    db = lancedb.connect(db_path)
    data = next(_gen())[0]
    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("page_text", pa.string()),
            pa.field("image", pa.string()),
            pa.field("image_vectors", pa.list_(pa.float32(), 768)),
            pa.field("page_idx", pa.int64()),
            pa.field("page_embedding_shape", pa.list_(pa.int64())),
            pa.field(
                "page_embedding_flatten",
                pa.list_(pa.float32(), len(data["page_embedding_flatten"])),
            ),
        ]
    )
    data = _gen()
    table = db.create_table(table_name, schema=schema, data=_gen(), mode="overwrite")
    return table


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
            [flattened_tensor, torch.zeros(padding_length, dtype=tensor.dtype)], dim=0
        )
    else:
        # Truncate the tensor if it's already too long
        padded_tensor = flattened_tensor[:desired_length]

    return padded_tensor


def search(
    query: str,
    table_name: str,
    model,
    processor,
    db_path: str = "lancedb",
    top_k: int = 3,
    image_vector=True,
    fts=False,
    vector=False,
    limit=10,
    where=None,
):
    qs = get_query_embedding(query=query, model=model, processor=processor)
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    query_embedding = embed_text(query)

    try:
        table.create_index(vector_column_name="image_vectors")
    except Exception:
        pass
    # Search over all dataset
    if vector and fts:
        raise ValueError("can't filter using both fts and vector")

    if fts:
        limit = limit or 100
        r = table.search(query, query_type="fts").limit(limit)
    elif image_vector:
        limit = limit or 100

        print("using image vectors for hybrid search")

        r = table.search(
            query_embedding.tolist(),
            query_type="vector",
            vector_column_name="image_vectors",
        )
    elif vector:
        limit = limit or 100
        vec_q = flatten_and_zero_pad(
            qs["embeddings"], table.to_pandas()["page_embedding_flatten"][0].shape[0]
        )
        r = table.search(vec_q.float().numpy(), query_type="vector").limit(limit)
    else:
        r = table.search().limit(limit)
    if where:
        r = r.where(where)

    r = r.to_list()

    print("length of retrived data:", len(r))

    def process_patch_embeddings(x):
        patches = np.reshape(x["page_embedding_flatten"], x["page_embedding_shape"])
        unflattended_embeddinged = torch.from_numpy(patches).to(torch.bfloat16)

        print(unflattended_embeddinged.shape)

        return unflattended_embeddinged

    all_pages_embeddings = [process_patch_embeddings(x) for x in r]

    # retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    # scores = retriever_evaluator.evaluate_colbert(
    #     [qs["embeddings"]], all_pages_embeddings
    # )
    scores = processor.score(qs["embeddings"], all_pages_embeddings).cpu().numpy()

    scores_tensor = torch.from_numpy(scores)

    # Use topk on the tensor
    top_k_indices = torch.topk(
        scores_tensor, k=min(top_k, scores_tensor.shape[1]), dim=1
    ).indices

    results = []
    for idx in top_k_indices[0].tolist():  # Convert indices to list
        page = r[idx]
        pil_image = base64_to_pil(page["image"])
        result = {
            "name": page["name"],
            "page_idx": page["page_idx"],
            "pil_image": pil_image,
        }
        results.append(result)
    return results


def ingest_pdf(pdf_file):
    try:
        # Get the file path from the uploaded file
        pdf_path = pdf_file.name

        # Check if the file exists and is not a directory
        if not os.path.isfile(pdf_path):
            return f"Error: The path '{pdf_path}' is not a valid file."

        # Create embeddings and store in the database
        docs_storage = embedd_docs(pdf_path, model, processor)
        table = create_db(docs_storage, table_name="pdf_query", db_path="lancedb")

        return f"PDF ingested successfully"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def query_pdf(query):
    # Search for relevant images
    results = search(query, "pdf_query", model, processor, db_path="lancedb", top_k=3)

    # Extract images and prepare them for display
    images = [result["pil_image"] for result in results]

    # Run VLM inference on the retrieved images
    vlm_response = vlm.response(query=query, images=images)

    return images, vlm_response


with gr.Blocks() as app:
    gr.Markdown("# PDF Query and Visual Language Model App")

    with gr.Tab("Ingest PDF"):
        pdf_input = gr.File(label="Upload PDF", type="filepath")
        ingest_button = gr.Button("Ingest PDF")
        ingest_output = gr.Textbox(label="Ingestion Result")

        ingest_button.click(ingest_pdf, inputs=[pdf_input], outputs=[ingest_output])
    with gr.Tab("Query PDF"):
        query_input = gr.Textbox(label="Enter your query")
        query_button = gr.Button("Search")
        image_output = gr.Gallery(label="Retrieved Images")
        response_output = gr.Textbox(label="VLM Response")

        query_button.click(
            query_pdf, inputs=[query_input], outputs=[image_output, response_output]
        )

# Launch the app
app.launch()

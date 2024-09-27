import torch
from typing import List, Dict, Any, Union, TypeVar, Optional, cast
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor


def get_model_colpali(model_name: str = "vidore/colpali-v1.2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))
    return model, processor

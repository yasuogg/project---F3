"""Frozen encoders for the RL refiner state (SigLIP image + MiniLM text)."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from PIL import Image


class FrozenEncoders(nn.Module):
    """SigLIP image encoder + sentence-transformers MiniLM text encoder, frozen."""

    def __init__(
        self,
        image_model: str = "google/siglip-base-patch16-224",
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.img_processor = AutoProcessor.from_pretrained(image_model)
        self.img_model = AutoModel.from_pretrained(image_model).to(device).eval()
        self.text_model = SentenceTransformer(text_model, device=device).eval()

        # dims
        self.img_dim = self.img_model.config.vision_config.hidden_size  # 768 for base
        self.txt_dim = self.text_model.get_sentence_embedding_dimension()  # 384

        for p in self.img_model.parameters():
            p.requires_grad = False
        # SentenceTransformer params already frozen by .eval(); ensure:
        for p in self.text_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Return (img_dim,) pooled image embedding."""
        inputs = self.img_processor(images=image, return_tensors="pt").to(self.device)
        out = self.img_model.get_image_features(**inputs)
        return out.squeeze(0).float()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Return (N, txt_dim)."""
        if not texts:
            return torch.zeros(0, self.txt_dim, device=self.device)
        emb = self.text_model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False,
            device=self.device, normalize_embeddings=True,
        )
        return emb.float()

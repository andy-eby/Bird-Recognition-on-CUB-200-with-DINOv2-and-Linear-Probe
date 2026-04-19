"""DINOv2 feature extractor backbone."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


# Mapping of model variant to HuggingFace model ID
DINOV2_MODEL_REGISTRY = {
    'vits14': 'facebook/dinov2-small',
    'vitb14': 'facebook/dinov2-base',
    'vitl14': 'facebook/dinov2-large',
}

# Expected output dimensions for each variant
DINOV2_OUTPUT_DIM = {
    'vits14': 384,
    'vitb14': 768,
    'vitl14': 1024,
}


class DINOv2Extractor(nn.Module):
    """
    DINOv2 frozen feature extractor.
    
    Loads a pretrained DINOv2 ViT model from HuggingFace and freezes all parameters.
    Returns the [CLS] token (last_hidden_state[:, 0, :]) as the image feature.
    """

    def __init__(self, model_variant: str = 'vitb14'):
        """
        Initialize DINOv2 extractor.
        
        Args:
            model_variant (str): One of 'vits14', 'vitb14', 'vitl14'.
                                 Defaults to 'vitb14' for CUB-200.
        """
        super().__init__()

        if model_variant not in DINOV2_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_variant: {model_variant}. "
                f"Choose from {list(DINOV2_MODEL_REGISTRY.keys())}"
            )

        self.model_variant = model_variant
        self.output_dim = DINOV2_OUTPUT_DIM[model_variant]

        # Load pretrained model from HuggingFace
        model_id = DINOV2_MODEL_REGISTRY[model_variant]
        self.model = AutoModel.from_pretrained(model_id)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Count frozen parameters
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

        # Sanity check
        print(f"✓ DINOv2 Extractor initialized")
        print(f"  Model variant: {self.model_variant}")
        print(f"  Output dimension: {self.output_dim}")
        print(f"  Frozen parameters: {frozen_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using frozen DINOv2.
        
        Args:
            x (torch.Tensor): Input images, shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: [CLS] token features, shape (batch_size, output_dim)
        """
        with torch.no_grad():
            outputs = self.model(x)
            # Extract [CLS] token (first token in sequence dimension)
            cls_token = outputs.last_hidden_state[:, 0, :]

        return cls_token

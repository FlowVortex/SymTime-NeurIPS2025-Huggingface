from dataclasses import dataclass
from typing import List, Literal, Optional, Dict
from enum import Enum

from transformers.configuration_utils import PretrainedConfig


@dataclass
class SymTimeConfig(PretrainedConfig):
    """
    Time series encoder configuration for SymTime Model.

    Parameters
    -----------
    patch_size
        The size of the patch to be used for the input data.
    num_layers
        The number of layers to be used for the encoder.
    d_model
        The dimension of the model.
    d_ff
        The dimension of the feedforward network.
    num_heads
        The number of heads to be used for the attention mechanism.
    norm
        The normalization to be used for the encoder.
    attn_dropout
        The dropout rate to be used for the attention mechanism.
    dropout
        The dropout rate to be used for the encoder.
    act
        The activation function to be used for the encoder.
    pre_norm
        Whether to use pre-norm for the encoder.
    """

    model_type = "time_series_transformer"

    def __init__(
        self,
        patch_size: int = 16,
        num_layers: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        norm: str = "BatchNorm",
        dropout: float = 0.1,
        act: str = "gelu",
        pre_norm: bool = False,
        initializer_factor: float = 0.05,
        **kwargs,
    ) -> None:
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm = norm
        self.dropout = dropout
        self.act = act
        self.pre_norm = pre_norm
        self.initializer_factor = initializer_factor

        super().__init__(**kwargs)

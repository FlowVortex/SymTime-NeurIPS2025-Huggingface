from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel

from configuration_symtime import SymTimeConfig
from layers import MultiHeadAttention, TSTEncoder, TSTEncoderLayer


class SymTimeModel(PreTrainedModel):
    """
    SymTime Model for Huggingface.

    Parameters
    ----------
    config: SymTimeConfig
        The configuration of the SymTime model.

    Attributes
    ----------
    config: SymTimeConfig
        The configuration of the SymTime model.
    encoder: TSTEncoder
        The encoder of the SymTime model.

    Methods
    -------
    forward(x: Tensor) -> Tuple[Tensor, Tensor]:
        Forward pass of the SymTime model.

    _init_weights(module: nn.Module) -> None:
        Initialize weights for the SymTime encoder stack.
    """

    config_class = SymTimeConfig

    def __init__(self, config: SymTimeConfig):
        super().__init__(config)
        self.config = config

        self.patch_size = config.patch_size
        self.stride = config.stride

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.encoder = TSTEncoder(
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            hidden_size=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            norm=config.norm,
            attn_dropout=config.dropout,
            dropout=config.dropout,
            act=config.act,
            pre_norm=config.pre_norm,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module) -> None:
        """Initialize weights for the SymTime encoder stack.

        The model is built on top of Hugging Face `PreTrainedModel`, so this method
        is called recursively via `post_init()`. We keep the initialization aligned
        with the current backbone structure in `layers.py`:

        - `TSTEncoder.W_P`: patch projection linear layer
        - `TSTEncoder.cls_token`: learnable CLS token
        - `TSTEncoderLayer.self_attn`: Q/K/V and output projections
        - `TSTEncoderLayer.ff`: feed-forward linear layers
        - `LayerNorm` / `BatchNorm1d`: normalization layers
        """
        super()._init_weights(module)

        factor = self.config.initializer_factor
        d_model = self.config.d_model
        num_heads = self.config.num_heads
        d_k = d_model // num_heads
        d_v = d_k

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=factor * (module.in_features**-0.5)
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm1d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, TSTEncoder):
            if hasattr(module, "cls_token") and module.cls_token is not None:
                nn.init.normal_(module.cls_token, mean=0.0, std=factor)
            if hasattr(module, "W_P") and isinstance(module.W_P, nn.Linear):
                nn.init.normal_(
                    module.W_P.weight,
                    mean=0.0,
                    std=factor * (module.W_P.in_features**-0.5),
                )
                if module.W_P.bias is not None:
                    nn.init.zeros_(module.W_P.bias)

        elif isinstance(module, MultiHeadAttention):
            nn.init.normal_(module.W_Q.weight, mean=0.0, std=factor * (d_model**-0.5))
            nn.init.normal_(module.W_K.weight, mean=0.0, std=factor * (d_model**-0.5))
            nn.init.normal_(module.W_V.weight, mean=0.0, std=factor * (d_model**-0.5))
            if module.W_Q.bias is not None:
                nn.init.zeros_(module.W_Q.bias)
            if module.W_K.bias is not None:
                nn.init.zeros_(module.W_K.bias)
            if module.W_V.bias is not None:
                nn.init.zeros_(module.W_V.bias)

            out_proj = module.to_out[0]
            nn.init.normal_(
                out_proj.weight, mean=0.0, std=factor * ((num_heads * d_v) ** -0.5)
            )
            if out_proj.bias is not None:
                nn.init.zeros_(out_proj.bias)

        elif isinstance(module, TSTEncoderLayer):
            for submodule in module.ff:
                if isinstance(submodule, nn.Linear):
                    nn.init.normal_(
                        submodule.weight,
                        mean=0.0,
                        std=factor * (submodule.in_features**-0.5),
                    )
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def patching(self, time_series: torch.Tensor) -> torch.Tensor:
        """Split a raw 1D time series into overlapping or non-overlapping patches.

        The encoder does not operate directly on the full sequence. Instead, it
        first converts the input into a sequence of local windows, where each
        window has length ``self.patch_size`` and consecutive windows are shifted
        by ``self.stride``. This patch-based representation reduces the temporal
        resolution while preserving local patterns that are useful for attention
        layers.

        If the sequence length is not compatible with the patch size, we pad the
        sequence on the right using replication padding so that the final patch
        extraction remains well-defined.
        """

        # Unpack the input shape for clarity: each sample is a 1D signal.
        batch_size, seq_length = time_series.shape

        # When the sequence length cannot be evenly covered by the patch size,
        # extend the sequence with replicated boundary values. This avoids
        # discarding the tail of the signal and keeps the patching procedure
        # consistent for every batch element.
        if seq_length % self.patch_size != 0:
            time_series = self.padding_patch_layer(time_series)

        # Convert the padded sequence into a patch tensor using a sliding window.
        # The resulting tensor contains local segments sampled along the last
        # dimension, which will be consumed by the transformer encoder.
        time_series = time_series.unfold(
            dimension=-1, size=self.patch_size, step=self.stride
        )

        return time_series

    def forward(
        self, x: Tensor, return_cls_token: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Run the full SymTime inference pipeline.

        The forward pass expects a 2D tensor of shape ``[batch_size, seq_length]``
        containing a batch of univariate time series. The input is first converted
        into patch embeddings through :meth:`patching`, and the resulting patch
        sequence is then passed into the transformer encoder.

        Parameters
        ----------
        x : Tensor
            Batched input time series with shape ``[batch_size, seq_length]``.
        return_cls_token : bool, optional
            If ``True``, the encoder also returns the learned CLS token output
            alongside the patch-level representations. This is useful when the
            downstream task needs a global sequence summary.

        Returns
        -------
        Tuple[Tensor, Tensor]
            The encoded patch sequence and, optionally, the CLS token output.
        """

        # Validate that the input follows the expected batch-by-time layout.
        assert (
            x.dim() == 2
        ), "Input time series must be a 2D tensor with shape of [batch_size, seq_length]."

        # Convert the raw signal into a patch-based representation before encoding.
        time_series = self.patching(x)

        # Feed the patch sequence into the transformer encoder and return its output.
        return self.encoder(time_series, return_cls_token=return_cls_token)

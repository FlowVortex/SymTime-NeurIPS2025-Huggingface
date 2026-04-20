from typing import Optional, Union, Tuple, Callable
import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange


def get_activation_fn(activation: Union[str, Callable]) -> nn.Module:
    """Select the activation function to use."""
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


class Transpose(nn.Module):
    """Transpose the dimensions of the input tensor"""

    def __init__(self, *dims, contiguous=False) -> None:
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class PositionalEmbedding(nn.Module):
    """Adding the positional encoding to the input for Transformer"""

    def __init__(self, hidden_size: int, max_len: int = 5000) -> None:
        super(PositionalEmbedding, self).__init__()

        # Calculate the positional encoding once in the logarithmic space.
        pe = torch.zeros(
            max_len, hidden_size
        ).float()  # Initialize a tensor of zeros with shape (max_len, hidden_size) to store positional encodings
        pe.requires_grad = (
            False  # Positional encodings do not require gradients as they are fixed
        )

        position = (
            torch.arange(0, max_len).float().unsqueeze(1)
        )  # Generate a sequence from 0 to max_len-1 and add a dimension at the 1st axis
        div_term = (
            torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
        ).exp()  # Calculate the divisor term in the positional encoding formula

        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # Apply the sine function to the even columns of the positional encoding matrix
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # Apply the cosine function to the odd columns of the positional encoding matrix

        pe = pe.unsqueeze(
            0
        )  # Add a batch dimension, changing the shape to (1, max_len, hidden_size)
        self.register_buffer(
            "pe", pe
        )  # Register the positional encodings as a buffer, which will not be updated as model parameters

    def forward(self, x: Tensor) -> Tensor:
        # Return the first max_len positional encodings that match the length of input x
        return x + self.pe[:, : x.size(1)]


class TSTEncoder(nn.Module):
    """Time series encoder backbone of SymTime"""

    def __init__(
        self,
        patch_size: int = 16,
        num_layers: int = 3,
        hidden_size: int = 128,
        num_heads: int = 16,
        d_k: int = None,
        d_v: int = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        # The Linear layer to project the input patches to the model dimension
        self.W_P = nn.Linear(patch_size, hidden_size)

        # Positional encoding
        self.pe = PositionalEmbedding(hidden_size=hidden_size)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Create the [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cls_mask = nn.Parameter(torch.ones(1, 1).bool(), requires_grad=False)

        # Create the encoder layer of the model backbone
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=act,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for _ in range(num_layers)
            ]
        )

        # model params init
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """model params init through apply methods"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: Tensor,  # x: [batch_size, patch_num, patch_size]
        attn_mask: Optional[Tensor] = None,  # attn_mask: [batch, num_patch]
    ) -> Tensor:
        batch_size = x.size(0)

        # Input patching embedding
        x = self.W_P(x)  # x: [batch_size, patch_num, model_dim]

        # Add the [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        # adjust the attn mask
        if attn_mask is not None:
            attn_mask = torch.cat(
                [self.cls_mask.expand(batch_size, -1), attn_mask], dim=1
            )

        # Add the positional embedding
        x = self.pe(x)
        x = self.dropout(x)  # x: [batch_size, patch_num, hidden_size]

        for mod in self.layers:
            x = mod(x, attn_mask=attn_mask)

        return x


class TSTEncoderLayer(nn.Module):
    """Patch-based Transformer module sublayer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        d_k: int = None,
        d_v: int = None,
        d_ff: int = 256,
        store_attn: int = False,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        pre_norm: bool = False,
    ) -> None:
        super(TSTEncoderLayer, self).__init__()

        assert (
            not hidden_size % num_heads
        ), f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        # If not specified, the number of heads is divided
        d_k = hidden_size // num_heads if d_k is None else d_k
        d_v = hidden_size // num_heads if d_v is None else d_v

        # Create the multi-head attention
        self.self_attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(hidden_size)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, hidden_size, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(hidden_size)

        # use pre-norm or not
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.attn = None

    def forward(
        self, src: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Multi-Head attention sublayer"""

        # Whether to use pre-norm for attention layer
        if self.pre_norm:
            src = self.norm_attn(src)

        # Multi-Head attention
        src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn

        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Whether to use pre-norm for ffn layer
        if self.pre_norm:
            src = self.norm_ffn(src)

        # Position-wise Feed-Forward
        src2 = self.ff(src)

        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return src


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism layer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        d_k: int = None,
        d_v: int = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x hidden_size]
            K, V:    [batch_size (bs) x q_len x hidden_size]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = hidden_size // num_heads if d_k is None else d_k
        d_v = hidden_size // num_heads if d_v is None else d_v

        self.num_heads, self.d_k, self.d_v = num_heads, d_k, d_v

        self.W_Q = nn.Linear(hidden_size, d_k * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(hidden_size, d_k * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(hidden_size, d_v * num_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = _ScaledDotProductAttention(
            hidden_size, num_heads, attn_dropout=attn_dropout
        )

        # Project output
        self.to_out = nn.Sequential(
            nn.Linear(num_heads * d_v, hidden_size), nn.Dropout(proj_dropout)
        )

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        bs = q.size(0)
        if k is None:
            k = q
        if v is None:
            v = q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(k).view(bs, -1, self.num_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(v).view(bs, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_v)
        )
        output = self.to_out(output)

        return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        res_attention: bool = False,
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = hidden_size // num_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=False)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        :param q: [batch_size, num_heads, num_token, d_k]
        :param k: [batch_size, num_heads, d_k, num_token]
        :param v: [batch_size, num_heads, num_token, d_k]
        :param attn_mask: [batch_size, num_heads, num_token]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale

        # Attention mask (optional)
        if (
            attn_mask is not None
        ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            attn_mask = rearrange(attn_mask, "b i -> b 1 i 1") * rearrange(
                attn_mask, "b i -> b 1 1 i"
            )
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

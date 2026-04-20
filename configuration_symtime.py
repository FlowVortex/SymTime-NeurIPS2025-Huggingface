from dataclasses import dataclass
from typing import List, Literal, Optional, Dict
from enum import Enum

from transformers.configuration_utils import PretrainedConfig


@dataclass
claas SymTimeConfig(PretrainedConfig):
    context_window: 256
    patch_len: 16
    stride: 8
    time_layers: 6
    d_model: 512
    n_heads: 8
    d_ff: 2048
    norm: BatchNorm
    attn_dropout: 0.1
    dropout: 0.1
    act: gelu
    pre_norm: False
    hidden_size: 768
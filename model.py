import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel

from configuration_symtime import SymTimeConfig
from layers import TSTEncoder


class SymTimeModel(PreTrainedModel):
    def __init__(self, config: SymTimeConfig):
        super().__init__(config)
        self.config = config
        self.encoder = TSTEncoder(
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            norm=config.norm,
            attn_dropout=config.attn_dropout,
            dropout=config.dropout,
            act=config.act,
            pre_norm=config.pre_norm,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module) -> None:
        """
        Initialize the weights of the `LiteSpecFormer` model.
        Including the initialization of the weights of the LayerNorm layer in the Transformer backbone,
        the weights of the RMSNorm layer in the Feedforward Network,
        and the weights of the depthwise and pointwise CNN in the Feedforward Network.
        """
        super()._init_weights(module)

        # Upload the factor for model initialization
        factor = self.config.initializer_factor

        if isinstance(module, nn.BatchNorm):
            # Initialize the weights of the LayerNorm in the Transformer backbone
            module.weight.data.fill_(factor * 1.0)

        elif isinstance(module, nn.Conv1d):
            # Initialize the depthwise and pointwise CNN in the Feedforward Network

            # Initialize the weights of the depthwise convolution
            module.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            # Initialize the biases of the depthwise convolution
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

            # Initialize the weights of the pointwise convolution
            module.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            # Initialize the biases of the pointwise convolution
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, MultiHeadAttention):
            # Initialize the weights of the query, key, and value layers

            # Upload the factor for model initialization
            d_model = self.config.d_model
            kv_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads

            # Initialize the weights of the query, key, and value layers
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * kv_proj_dim) ** -0.5)
            )
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            # The finnal projection layer
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * kv_proj_dim) ** -0.5)
            )

        elif isinstance(module, LiteSpecFormerModel):
            # Initialize the weights of the learnable embedding layer
            if self.forecasting_config.use_reg_token:
                module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)

        elif isinstance(module, ResidualBlock):
            # Initialize the weights of the embedding and output layers
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.hidden_layer.weight.size(-1) ** -0.5),
            )
            if (
                hasattr(module.hidden_layer, "bias")
                and module.hidden_layer.bias is not None
            ):
                module.hidden_layer.bias.data.zero_()

            # The hidden residual layer
            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.residual_layer.weight.size(-1) ** -0.5),
            )
            if (
                hasattr(module.residual_layer, "bias")
                and module.residual_layer.bias is not None
            ):
                module.residual_layer.bias.data.zero_()

            # The final output layer for the time series forecasting
            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.output_layer.weight.size(-1) ** -0.5)
            )
            if (
                hasattr(module.output_layer, "bias")
                and module.output_layer.bias is not None
            ):
                module.output_layer.bias.data.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.encoder(x)

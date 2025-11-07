"""Transformer decoder-only architecture for code generation.

This module implements a GPT-style transformer decoder optimized for
code generation tasks with memory-efficient features for RTX 5060Ti.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the RecursiveCodeLLM model.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of hidden states.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Dimension of FFN intermediate layer.
        max_position_embeddings: Maximum sequence length.
        dropout: Dropout probability.
        attention_dropout: Attention dropout probability.
        layer_norm_eps: Layer normalization epsilon.
        initializer_range: Standard deviation for weight initialization.
        use_flash_attention: Whether to use flash attention.
        gradient_checkpointing: Whether to use gradient checkpointing.
    """

    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer.

    Implements efficient multi-head self-attention with optional
    flash attention for improved memory efficiency.
    """

    def __init__(self, config: ModelConfig):
        """Initialize multi-head attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # Query, key, value projections
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of multi-head attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Mask tensor of shape (batch_size, 1, seq_len, seq_len).
            use_cache: Whether to return key/value for caching.
            past_key_value: Cached key and value from previous step.

        Returns:
            Tuple of (output tensor, cached key/value if use_cache).
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_attention_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Handle cached key/value
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        past_key_value = (key, value) if use_cache else None

        # Scaled dot-product attention
        # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, kv_len)
        # -> (batch, heads, seq_len, kv_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of values
        # (batch, heads, seq_len, kv_len) @ (batch, heads, kv_len, head_dim)
        # -> (batch, heads, seq_len, head_dim)
        context = torch.matmul(attention_probs, value)

        # Reshape and project
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)

        return output, past_key_value


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: ModelConfig):
        """Initialize feed-forward layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network.

        Args:
            hidden_states: Input tensor.

        Returns:
            Output tensor.
        """
        hidden_states = self.dense_in(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, config: ModelConfig):
        """Initialize transformer block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of transformer block.

        Args:
            hidden_states: Input tensor.
            attention_mask: Attention mask.
            use_cache: Whether to use caching.
            past_key_value: Cached key/value.

        Returns:
            Tuple of (output tensor, cached key/value).
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attention_output, past_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        hidden_states = residual + self.dropout(attention_output)

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)

        return hidden_states, past_key_value


class RecursiveCodeLLM(nn.Module):
    """Transformer decoder-only model for code generation.

    This model is optimized for recursive self-improvement and
    efficient training on consumer GPUs.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between embedding and output
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing

        logger.info(
            f"Initialized RecursiveCodeLLM with {self.num_parameters():,} parameters"
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings."""
        return self.token_embedding

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """Set input embeddings."""
        self.token_embedding = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> dict:
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            labels: Labels for language modeling of shape (batch_size, seq_len).
            use_cache: Whether to use key/value caching.
            past_key_values: Cached key/values from previous forward pass.

        Returns:
            Dictionary containing loss, logits, and optionally past_key_values.
        """
        batch_size, seq_len = input_ids.size()

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position IDs
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
            position_ids = torch.arange(
                past_length,
                seq_len + past_length,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = torch.arange(
                seq_len,
                dtype=torch.long,
                device=input_ids.device,
            )

        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=input_ids.device,
            )

        # Expand attention mask to 4D
        # (batch_size, seq_len) -> (batch_size, 1, seq_len, seq_len)
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask * causal_mask
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Pass through transformer layers
        all_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, past_kv = self._gradient_checkpointing_forward(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_cache,
                    past_key_value,
                )
            else:
                hidden_states, past_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_value=past_key_value,
                )

            if use_cache:
                all_past_key_values.append(past_kv)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten and compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": all_past_key_values if use_cache else None,
        }

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length.
            device: Device to create mask on.

        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len).
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 1, 0)
        return mask.unsqueeze(0).unsqueeze(0)

    def _gradient_checkpointing_forward(
        self,
        layer: TransformerBlock,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with gradient checkpointing.

        Args:
            layer: Transformer block.
            hidden_states: Input hidden states.
            attention_mask: Attention mask.
            use_cache: Whether to use caching.
            past_key_value: Cached key/value.

        Returns:
            Tuple of (output hidden states, cached key/value).
        """
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(layer),
            hidden_states,
            attention_mask,
            use_cache,
            past_key_value,
        )

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count number of parameters.

        Args:
            only_trainable: Whether to count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text using the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            max_length: Maximum length to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to generate per input.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID.

        Returns:
            Generated token IDs of shape (batch_size * num_return_sequences, max_length).
        """
        self.eval()

        batch_size = input_ids.size(0)
        device = input_ids.device

        # Expand input for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)

        generated = input_ids
        past_key_values = None

        for _ in range(max_length - input_ids.size(1)):
            # Forward pass
            outputs = self.forward(
                generated if past_key_values is None else generated[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

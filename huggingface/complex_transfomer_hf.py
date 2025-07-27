"""
Complex Embeddings Integration for Hugging Face Transformers
Author: Research Implementation
Description: Integration of complex embeddings into standard HF models for benchmarking
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    BertModel, BertConfig, GPT2Model, GPT2Config,
    AutoTokenizer, AutoModel, PretrainedConfig, PreTrainedModel
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import Optional, Tuple, Union
import math


class ComplexEmbeddings(nn.Module):
    """Enhanced Complex Embeddings with configurable variants"""
    def __init__(self, vocab_size, d_model, max_position_embeddings, gamma: float = 1.0, 
                 complex_variant: str = "sinusoidal"):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.gamma = gamma
        self.complex_variant = complex_variant
        self.max_position_embeddings = max_position_embeddings

        if complex_variant == "sinusoidal":
            # Original sinusoidal approach
            position_encoding_dim = d_model
            freqs = 1.0 / (10000 ** (torch.arange(0, position_encoding_dim, 2).float() / position_encoding_dim))
            self.register_buffer('omega', freqs)
        
        elif complex_variant == "learned_phase":
            # Learnable phase embeddings
            self.phase_embed = nn.Embedding(max_position_embeddings, d_model)
            
        elif complex_variant == "hybrid":
            # Hybrid: sinusoidal + learned components
            position_encoding_dim = d_model // 2
            freqs = 1.0 / (10000 ** (torch.arange(0, position_encoding_dim, 2).float() / position_encoding_dim))
            self.register_buffer('omega', freqs[:d_model//4])
            self.phase_embed = nn.Embedding(max_position_embeddings, d_model//2)

    def forward(self, input_ids, position_ids=None):
        real = self.vocab_embed(input_ids)
        seq_len = real.shape[-2]
        batch_size = real.shape[0]
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=real.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if self.complex_variant == "sinusoidal":
            positions = position_ids.float().unsqueeze(-1)
            angles = self.omega.unsqueeze(0).unsqueeze(0) * positions
            imag = self.gamma * torch.sin(angles).repeat_interleave(2, dim=-1)
            
        elif self.complex_variant == "learned_phase":
            imag = self.gamma * torch.tanh(self.phase_embed(position_ids))
            
        elif self.complex_variant == "hybrid":
            # Half sinusoidal, half learned
            positions = position_ids.float().unsqueeze(-1)
            angles = self.omega.unsqueeze(0).unsqueeze(0) * positions
            sin_part = self.gamma * torch.sin(angles).repeat_interleave(2, dim=-1)
            learned_part = self.gamma * torch.tanh(self.phase_embed(position_ids))
            imag = torch.cat([sin_part, learned_part], dim=-1)
        
        return torch.complex(real, imag)


class ComplexMultiHeadAttention(nn.Module):
    """Complex-aware Multi-Head Attention"""
    def __init__(self, config, variant="magnitude"):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.variant = variant
        self.alpha = getattr(config, 'complex_alpha', 0.3)

        # Complex projections
        self.query_real = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.query_imag = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key_real = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key_imag = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # Extract real and imaginary parts
        if torch.is_complex(hidden_states):
            real_part = hidden_states.real
            imag_part = hidden_states.imag
        else:
            real_part = hidden_states
            imag_part = torch.zeros_like(hidden_states)

        # Complex projections: (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
        q_real = self.query_real(real_part) - self.query_imag(imag_part)
        q_imag = self.query_real(imag_part) + self.query_imag(real_part)
        k_real = self.key_real(real_part) - self.key_imag(imag_part)
        k_imag = self.key_real(imag_part) + self.key_imag(real_part)
        
        # Reshape for multi-head attention
        query_layer = torch.complex(
            self.transpose_for_scores(q_real),
            self.transpose_for_scores(q_imag)
        )
        key_layer = torch.complex(
            self.transpose_for_scores(k_real),
            self.transpose_for_scores(k_imag)
        )
        value_layer = self.transpose_for_scores(self.value(real_part))

        # Complex attention scores
        attention_scores = torch.matmul(query_layer, key_layer.conj().transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Different variants for handling complex scores
        if self.variant == "magnitude":
            scores = torch.abs(attention_scores)
        elif self.variant == "phase":
            scores = torch.cos(torch.angle(attention_scores))
        elif self.variant == "real":
            scores = attention_scores.real
        elif self.variant == "hybrid":
            mag = torch.abs(attention_scores)
            phase = torch.cos(torch.angle(attention_scores))
            scores = mag + self.alpha * phase
        else:
            scores = torch.abs(attention_scores)  # Default to magnitude

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        attention_probs = nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ComplexBertConfig(BertConfig):
    """Extended BERT config for complex embeddings"""
    def __init__(self, use_complex_embeddings=True, complex_gamma=1.0, 
                 complex_variant="sinusoidal", complex_attention_variant="magnitude",
                 complex_alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        self.use_complex_embeddings = use_complex_embeddings
        self.complex_gamma = complex_gamma
        self.complex_variant = complex_variant
        self.complex_attention_variant = complex_attention_variant
        self.complex_alpha = complex_alpha


class ComplexBertEmbeddings(nn.Module):
    """BERT embeddings with complex positional encoding"""
    def __init__(self, config):
        super().__init__()
        if config.use_complex_embeddings:
            self.complex_embeddings = ComplexEmbeddings(
                vocab_size=config.vocab_size,
                d_model=config.hidden_size,
                max_position_embeddings=config.max_position_embeddings,
                gamma=config.complex_gamma,
                complex_variant=config.complex_variant
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if self.config.use_complex_embeddings:
            # Use complex embeddings (combines word + position)
            embeddings = self.complex_embeddings(input_ids, position_ids)
            # Take real part and add token type embeddings
            embeddings = embeddings.real + self.token_type_embeddings(token_type_ids)
        else:
            # Standard BERT embeddings
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ComplexBertModel(BertModel):
    """BERT Model with Complex Embeddings"""
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ComplexBertEmbeddings(config)
        
        # Replace attention layers if using complex attention
        if hasattr(config, 'use_complex_attention') and config.use_complex_attention:
            for layer in self.encoder.layer:
                layer.attention.self = ComplexMultiHeadAttention(
                    config, 
                    variant=config.complex_attention_variant
                )


class ComplexGPT2Config(GPT2Config):
    """Extended GPT2 config for complex embeddings"""
    def __init__(self, use_complex_embeddings=True, complex_gamma=1.0, 
                 complex_variant="sinusoidal", complex_attention_variant="magnitude",
                 complex_alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        self.use_complex_embeddings = use_complex_embeddings
        self.complex_gamma = complex_gamma
        self.complex_variant = complex_variant
        self.complex_attention_variant = complex_attention_variant
        self.complex_alpha = complex_alpha


class ComplexGPT2Model(GPT2Model):
    """GPT2 Model with Complex Embeddings"""
    def __init__(self, config):
        super().__init__(config)
        if config.use_complex_embeddings:
            self.wte = ComplexEmbeddings(
                vocab_size=config.vocab_size,
                d_model=config.n_embd,
                max_position_embeddings=config.n_positions,
                gamma=config.complex_gamma,
                complex_variant=config.complex_variant
            )
            # Remove positional embeddings since complex embeddings handle position
            self.wpe = None
        
        # Initialize weights
        self.init_weights()

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, 
                token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, use_cache=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None):
        
        if self.config.use_complex_embeddings and inputs_embeds is None:
            # Use complex embeddings
            inputs_embeds = self.wte(input_ids, position_ids)
            # Take real part for processing
            inputs_embeds = inputs_embeds.real
            
        return super().forward(
            input_ids=None,  # Set to None since we're using inputs_embeds
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


# Registry for easy model creation
MODEL_REGISTRY = {
    'complex-bert': (ComplexBertModel, ComplexBertConfig),
    'complex-gpt2': (ComplexGPT2Model, ComplexGPT2Config),
}


def create_complex_model(model_type, **config_kwargs):
    """Factory function to create complex embedding models"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class, config_class = MODEL_REGISTRY[model_type]
    config = config_class(**config_kwargs)
    model = model_class(config)
    return model, config


# Example usage and testing
if __name__ == "__main__":
    # Test Complex BERT
    bert_config = ComplexBertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        use_complex_embeddings=True,
        complex_variant="sinusoidal",
        complex_gamma=1.0
    )
    
    complex_bert = ComplexBertModel(bert_config)
    print("Complex BERT created successfully!")
    
    # Test Complex GPT2
    gpt2_config = ComplexGPT2Config(
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        use_complex_embeddings=True,
        complex_variant="hybrid",
        complex_gamma=0.8
    )
    
    complex_gpt2 = ComplexGPT2Model(gpt2_config)
    print("Complex GPT2 created successfully!")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    
    with torch.no_grad():
        bert_outputs = complex_bert(input_ids)
        gpt2_outputs = complex_gpt2(input_ids)
        
    print(f"BERT output shape: {bert_outputs.last_hidden_state.shape}")
    print(f"GPT2 output shape: {gpt2_outputs.last_hidden_state.shape}")
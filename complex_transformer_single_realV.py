# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from thop import profile
from scipy.stats import spearmanr
import math

import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ComplexEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, num_positions, gamma: float = 1.0):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.gamma = gamma

        position_encoding_dim = d_model
        freqs = 1.0 / (10000 ** (torch.arange(0, position_encoding_dim, 2).float() / position_encoding_dim)).to(device) # shape ()
        
        # Fixed: Don't duplicate frequencies, just use them for pairs
        self.register_buffer('omega', freqs)

    def forward(self, x):
        real = self.vocab_embed(x).to(device)

        seq_len = real.shape[-2]
        
        positions = torch.arange(seq_len, dtype=torch.float, device=real.device).unsqueeze(-1)

        angles = self.omega.unsqueeze(0).to(real.device) * positions

        imag = self.gamma * torch.sin(angles).repeat_interleave(2, dim=-1)

        if real.dim() == 3:
            imag = imag.unsqueeze(0).expand(real.shape[0], -1, -1)
        
        return torch.complex(real, imag) # [seq_len, d_model]
    

class MultiHeadAttentionComplex(nn.Module):
    def __init__(self, variant, d_model, num_heads, num_positions, alpha = 0.3, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_positions = num_positions
        
        self.W_q_real = nn.Linear(d_model, d_model, bias=False)
        self.W_q_imag = nn.Linear(d_model, d_model, bias=False)
        self.W_k_real = nn.Linear(d_model, d_model, bias=False)
        self.W_k_imag = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.d_k)
        self.variant = variant 
        
    def forward(self, z,  mask=None):
        # batch_size, seq_len = query.size(0), query.size(1)

        z_real = z.real
        z_imag = z.imag

        batch_size, seq_len = z_real.size(0), z_real.size(1)
        
        
        # Linear projections
        Q_real = (self.W_q_real(z_real) - self.W_q_imag(z_imag)).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        Q_imag = (self.W_q_real(z_imag) + self.W_q_imag(z_real)).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        Q = torch.complex(Q_real, Q_imag)


        K_real = (self.W_k_real(z_real) - self.W_k_imag(z_imag)).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_imag = (self.W_k_real(z_imag) + self.W_k_imag(z_real)).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.complex(K_real, K_imag)

        V = self.W_v(z_real).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention'
        attention_scores = torch.matmul(Q, K.conj().transpose(-2, -1)) * self.scale


        if self.variant == "magnitude":
            attention_scores = torch.abs(attention_scores) / np.sqrt(self.d_k)
        elif self.variant == "phase":
            attention_scores = torch.cos(torch.angle(attention_scores))/ np.sqrt(self.d_k)
        elif self.variant == "real":
            attention_scores = attention_scores.real / np.sqrt(self.d_k)
        elif self.variant == "hybrid_norm":
            mag = torch.abs(attention_scores)
            phase = torch.cos(torch.angle(attention_scores))
            attention_scores = (mag / mag.max()) + self.alpha * phase
            attention_scores = attention_scores / np.sqrt(self.d_k)
        elif self.variant == "hybrid":
            mag = torch.abs(attention_scores)
            phase = torch.cos(torch.angle(attention_scores))
            attention_scores = (mag + self.alpha * phase) / np.sqrt(self.d_k)

        elif self.variant == "apatt":  #named from paper 
            magnitude = torch.abs(attention_scores)
            sign = attention_scores / (magnitude + 1e-8)  # Complex sign
            attention_scores = magnitude / np.sqrt(self.d_k)

            v_complex = torch.complex(v, torch.zeros_like(v))
            v = sign.unsqueeze(-2) * v_complex.unsqueeze(-3)
            #TODO cant plot complex scores 
            
        elif self.variant == "riatt":
            attn_real = torch.real(attention_scores) / np.sqrt(self.d_k)
            attn_imag = torch.imag(attention_scores) / np.sqrt(self.d_k)

            attn_probs_real = self.softmax(attn_real)
            attn_probs_imag = self.softmax(attn_imag)

            attention_scores = torch.complex(attn_probs_real, attn_probs_imag)

            #TODO cant plot complex scores 
        
        
        # Apply attention mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(context)
        
        return output, attention_probs.mean(dim=1)  # Average over heads for visualization





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_positions, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_positions = num_positions
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(context)
        
        return output, attention_probs.mean(dim=1)  # Average over heads for visualization




class ComplexTransformerLayer(nn.Module):
    def __init__(self,variant, d_model, num_heads, d_ff, num_positions, alpha, dropout=0.1):
        """
        :param d_model: The dimension of the inputs and outputs of the layer
        :param d_internal: The "internal" dimension used in the self-attention computation
        """
        super().__init__()
        
        
        self.d_model = d_model
        self.num_positions  = num_positions
        self.alpha = alpha 

        self.attention = MultiHeadAttentionComplex(variant, d_model, num_heads, num_positions, alpha, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_vecs,  attention_mask=None):
        """
        :param input_vecs: complex tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the processed features
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """


        z = input_vecs.to(device)
        
        H, attention_map = self.attention(
            z, mask=attention_mask
        )        
        
        # Add and Norm
        Z1 = self.norm1(z.real+self.dropout(H))
        
        # Linear layers
        Z2 = self.feed_forward(Z1)

        # Add norm
        Z2 = self.norm2(Z1+self.dropout(Z2))

        return Z2  , attention_map 



class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_positions, dropout=0.1):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # raise Exception("Implement me")
        
        self.d_model = d_model
        self.num_positions  = num_positions

        self.attention = MultiHeadAttention(d_model, num_heads, num_positions, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_vecs, attention_mask=None):
        """
        :param input_vecs: an input tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the log probabilities of each position in the input
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """
        # raise Exception("Implement me")
        
        x = input_vecs.to(device)
        
        H, attention_map = self.attention(
            x, x, x, mask=attention_mask
        )        
        
        # Add and Norm
        Z1 = self.norm1(x+self.dropout(H))
        
        # Linear layers
        Z2 = self.feed_forward(Z1)

        # Add norm
        Z2 = self.norm2(Z1+self.dropout(Z2))

        return Z2  , attention_map      
                

# class PhaseAwareAttention(nn.Module):
#     def __init__(self, variant, d_model, d_internal, alpha: float = 0.3):
#         super().__init__()
#         self.d_model = d_model
#         self.d_k = d_internal
#         self.d_v = d_internal
#         self.variant  = variant 
        
#         # Complex-linear projections - using real linear layers but applying to complex inputs
#         self.q_proj_real = nn.Linear(d_model, d_internal, bias=False)
#         self.q_proj_imag = nn.Linear(d_model, d_internal, bias=False)
#         self.k_proj_real = nn.Linear(d_model, d_internal, bias=False)
#         self.k_proj_imag = nn.Linear(d_model, d_internal, bias=False)

#         #TODO
#         self.v_proj = nn.Linear(d_model, d_internal, bias=False)
        
        
#         self.softmax = nn.Softmax(dim=-1)
#         self.alpha = alpha
        
#     def forward(self, z):  # z: complex [seq_len, d_model]
#         # Extract real and imaginary parts
#         z_real = z.real
#         z_imag = z.imag
        
#         # Complex queries and keys
#         q_real = self.q_proj_real(z_real) - self.q_proj_imag(z_imag)  #(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
#         q_imag = self.q_proj_real(z_imag) + self.q_proj_imag(z_real)
#         q = torch.complex(q_real, q_imag)
        
#         k_real = self.k_proj_real(z_real) - self.k_proj_imag(z_imag)
#         k_imag = self.k_proj_real(z_imag) + self.k_proj_imag(z_real)
#         k = torch.complex(k_real, k_imag)
        
#         #TODO
#         # Values are real (for simplicity in this implementation)
#         v = self.v_proj(z_real).to(device)

        
#         # Phase-aware attention scores (complex dot product)
#         attn_scores = torch.matmul(q, k.conj().transpose(-2, -1)).to(device)
#         # attn_magnitude = torch.abs(attn_scores)  # Magnitude
#         # attn_phase = torch.angle(attn_scores)    # Phase

#         # attn_scores_real = (attn_magnitude/torch.max(attn_magnitude) + self.alpha * torch.cos(attn_phase) ) / np.sqrt(self.d_k)  # Use magnitude only

#         if self.variant == "magnitude":
#             scores = torch.abs(attn_scores) / np.sqrt(self.d_k)
#         elif self.variant == "phase":
#             scores = torch.cos(torch.angle(attn_scores))/ np.sqrt(self.d_k)
#         elif self.variant == "real":
#             scores = attn_scores.real / np.sqrt(self.d_k)
#         elif self.variant == "hybrid_norm":
#             mag = torch.abs(attn_scores)
#             phase = torch.cos(torch.angle(attn_scores))
#             scores = (mag / mag.max()) + self.alpha * phase
#             scores = scores / np.sqrt(self.d_k)
#         elif self.variant == "hybrid":
#             mag = torch.abs(attn_scores)
#             phase = torch.cos(torch.angle(attn_scores))
#             scores = (mag + self.alpha * phase) / np.sqrt(self.d_k)

#         elif self.variant == "apatt":  #named from paper 
#             magnitude = torch.abs(attn_scores)
#             sign = attn_scores / (magnitude + 1e-8)  # Complex sign
#             scores = magnitude / np.sqrt(self.d_k)

#             v_complex = torch.complex(v, torch.zeros_like(v))
#             v = sign.unsqueeze(-2) * v_complex.unsqueeze(-3)
            
#         elif self.variant == "riatt":
#             attn_real = torch.real(attn_scores) / np.sqrt(self.d_k)
#             attn_imag = torch.imag(attn_scores) / np.sqrt(self.d_k)

#             attn_probs_real = self.softmax(attn_real)
#             attn_probs_imag = self.softmax(attn_imag)

#             scores = torch.complex(attn_probs_real, attn_probs_imag)

#             #TODO cant plot complex scores 
        

#         attn_weights = self.softmax(scores)
        
#         # Attend to real values
#         output = torch.matmul(attn_weights, v)
        
#         return output, attn_weights

        

# Should contain your overall Transformer implementation with complex embeddings
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model,  num_heads, d_ff, num_classes, num_layers, variant, gamma_pe: float = 1.0, alpha_attn: float = 0.2,  dropout=0.1, use_token_type_embeddings=True):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: dimension of the model
        :param d_internal: internal dimension for attention
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.variant = variant
        self.use_token_type_embeddings = use_token_type_embeddings

        # Complex embeddings instead of regular embeddings + positional encoding
        self.complex_embedding = ComplexEmbeddings(self.vocab_size, self.d_model, self.num_positions, gamma=gamma_pe)
        

        if use_token_type_embeddings:
            self.token_type_embedding = nn.Embedding(2, d_model)  # For sentence A/B


        #TODO
        # Create multiple transformer layers
        self.first_layer = ComplexTransformerLayer(self.variant, d_model, num_heads, d_ff, num_positions, alpha=alpha_attn,  dropout=dropout)
       
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, num_positions, dropout= dropout) 
            for _ in range(self.num_layers-1)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)


        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.Tanh(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model // 2, num_classes)
        # )

        
        # Initialize weights
        self.apply(self._init_weights)
        
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #     elif isinstance(module, nn.LayerNorm):
    #         nn.init.zeros_(module.bias)
    #         nn.init.ones_(module.weight)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    def forward(self, indices, attention_mask=None, token_type_ids=None):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Complex embeddings (encodes both content and positional information via phase)
        Z = self.complex_embedding(indices.to(device))
        

         # Add token type embeddings if provided
        if token_type_ids is not None and self.use_token_type_embeddings:
                token_type_embeddings = self.token_type_embedding(token_type_ids)
                Z.real+= token_type_embeddings

        Z.real = self.dropout(Z.real)
        Z.imag = self.dropout(Z.imag)

        attention_maps = []

        Z, A = self.first_layer(Z, attention_mask)
        attention_maps.append(A)

        for transformer_layer in self.transformer_layers:
            Z, A = transformer_layer(Z, attention_mask)
            attention_maps.append(A)
            
        # Final classification layers

        cls_output = Z[:, 0, :] 
            
        logits = self.classifier(cls_output)
        
        return logits, attention_maps


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



def train_complex_classifier_glue(train, val, num_epochs,variant, save_path, num_positions = 512, num_classes =2 , loss = "CE", use_token_type_embeddings =True):
    """
    Train the complex transformer classifier
    """
    # Initialize the complex transformer model

    model_config = {
        'vocab_size': tokenizer.vocab_size,  # BERT vocab size, adjust based on your tokenizer
        'num_positions': num_positions,
        'd_model': 256,
        'variant': variant,
        'num_heads': 8,
        'd_ff': 1024,
        'num_classes': num_classes,  # Adjust based on task (2 for binary classification)
        'num_layers': 6,
        'dropout': 0.2,
        'use_token_type_embeddings': use_token_type_embeddings
    }
    
    # Initialize model
    model = Transformer(**model_config).to(device)

    if loss=="CE":
        lr = 1e-4
    else:
        lr = 3e-5

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # warmup_ratio = 0.1


    if loss == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()


    total_steps = len(train) * num_epochs
    # warmup_steps = int(total_steps * warmup_ratio)

    # def lr_lambda(step):
    #         if step < warmup_steps:
    #             return step / warmup_steps
    #         else:
    #             progress = (step - warmup_steps) / (total_steps - warmup_steps)
    #             return 0.5 * (1 + math.cos(math.pi * progress))
        
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    train_losses = []
    val_losses = []

    min_val_loss = 100000
    
    for t in range(0, num_epochs):
        model.train()

        # print("epoch {}".format(t))
        train_loss_this_epoch = 0.0   
        val_loss_this_epoch = 0.0     
        
        for batch in train:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get token_type_ids if available
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits, attention_maps = model(
                indices=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Compute loss
            if loss == "MSE":
                logits = torch.sigmoid(logits.squeeze(-1)) * 5.0
                labels = labels.float()
                loss_batch = criterion(logits, labels)

            else:
               loss_batch = criterion(logits, labels)

            optimizer.zero_grad()
            loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss_this_epoch += loss_batch.item()

        model.eval()
        with torch.no_grad():
            for batch in val:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Get token_type_ids if available
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                logits, attention_maps = model(
                    indices=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                if loss == "MSE":
                    logits = torch.sigmoid(logits.squeeze(-1)) * 5.0
                    labels = labels.float()
                    loss_batch = criterion(logits, labels)
                else:
                    loss_batch = criterion(logits, labels)

                val_loss_this_epoch += loss_batch.item()

  
        # print("epoch train loss {}".format(loss_this_epoch))
        train_losses.append(train_loss_this_epoch/len(train))
        val_losses.append(val_loss_this_epoch/len(val))

        if val_loss_this_epoch/len(val) < min_val_loss:
            min_val_loss = val_loss_this_epoch/len(val)
            torch.save(model.state_dict(),save_path)

    return model, train_losses, val_losses



def decode_complex_glue(state_dict, dataloader, variant, num_positions = 512, num_classes =2 , loss = "CE", use_token_type_embeddings =True):

    model_config = {
        'vocab_size': tokenizer.vocab_size,  # BERT vocab size, adjust based on your tokenizer
        'num_positions': num_positions,
        'd_model': 256,
        'variant': variant,
        'num_heads': 8,
        'd_ff': 1024,
        'num_classes': num_classes,  # Adjust based on task (2 for binary classification)
        'num_layers': 6,
        'dropout': 0.1,
        'use_token_type_embeddings': use_token_type_embeddings
    }
    
    # Initialize model
    model = Transformer(**model_config).to(device)
    model.load_state_dict(state_dict)

    model.eval()
    preds, targets = [], []
    flops = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].unsqueeze(-1).to(device)
            attention_mask = batch["attention_mask"].to(device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits, attention_maps = model(
                indices=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # macs, params = profile(model, inputs=(input_ids,
            #     attention_mask,
            #     token_type_ids))

            # macs = macs/1e9
            # flops.append(macs)

            if loss=="CE":
                pred = torch.argmax(logits, dim=-1)
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
            else:
                pred = logits.squeeze(-1)*5.0
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.squeeze(-1).cpu().numpy())


    if loss =="CE":
        # print(preds)

        e = accuracy_score(targets, preds)
        g = f1_score(targets, preds)
        f = matthews_corrcoef(targets, preds)
        print(f"Accuracy:{np.round(e,5)}, F1 score {np.round(g,5)}, MCC {np.round(f,5)}")
        return e
    else:
        print(preds)

        spearman_corr, _ = spearmanr(preds, targets)
        print(f"Spearman correlation: {spearman_corr:.5f}")
        print(f"FLOPS mean: {np.mean(flops):.5f}, FLOPS var: {np.std(flops):.5f}")
        return spearman_corr


            

def attention_plots_complex(state_dict, valdata, variant,save_path, num_positions = 512, num_classes =2 , use_token_type_embeddings =True):
    
    model_config = {
        'vocab_size': tokenizer.vocab_size,  # BERT vocab size, adjust based on your tokenizer
        'num_positions': num_positions,
        'd_model': 256,
        'variant': variant,
        'num_heads': 8,
        'd_ff': 1024,
        'num_classes': num_classes,  # Adjust based on task (2 for binary classification)
        'num_layers': 6,
        'dropout': 0.1,
        'use_token_type_embeddings': use_token_type_embeddings
    }
    
    # Initialize model
    model = Transformer(**model_config).to(device)
    model.load_state_dict(state_dict)


    model.eval()
    i = 0
    with torch.no_grad():
            input_ids = valdata["input_ids"].to(device)
            labels = valdata["labels"].unsqueeze(-1).to(device)
            attention_mask = valdata["attention_mask"].to(device)

            token_type_ids = valdata.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits, attention_maps = model(
                indices=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # attention maps shape (nun_layers, num_samples, seq_len, seq_len)
            # Loop over attention maps (e.g., from different layers or heads)
            for j in range(len(attention_maps)):
                for i, attn_map in enumerate(attention_maps[j]):
                    # Ensure attn_map is a 2D tensor (e.g., [seq_len, seq_len])
                    # If it's 4D: [batch_size, num_heads, seq_len, seq_len], take the relevant slice
                    if attn_map.dim() == 4:
                        attn_map = attn_map[0, 0]  # first example, first head
                    elif attn_map.dim() == 3:
                        attn_map = attn_map[0]     # first example

                    fig, ax = plt.subplots(figsize=(8, 8))

                    # Set axis labels using input tokens
                    non_pad_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]

                    tokens = tokenizer.convert_ids_to_tokens(input_ids[i][non_pad_indices]) 
                    im = ax.imshow(attn_map[non_pad_indices][:, non_pad_indices].detach().cpu().numpy(), cmap='hot', interpolation='nearest')

                    ax.set_xticks(np.arange(len(tokens)))
                    ax.set_yticks(np.arange(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                    ax.set_yticklabels(tokens, fontsize=8)
                    ax.xaxis.tick_top()

                    plt.tight_layout()
                    plt.savefig(f"{save_path}/complex_{variant}_sample{i}_attn_{j}.png")
                    plt.close(fig)
                    
                


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

from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
import torch.nn.functional as F

        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Sinusoidal positional encoding implementation
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=True):
        """
        :param d_model: dimensionality of the embedding layer to your model
        :param num_positions: the maximum sequence length this module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        self.d_model = d_model
        self.batched = batched
        
        # Create the sinusoidal positional encoding matrix
        pe = torch.zeros(num_positions, d_model).to(device)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        
        # Create the div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (if d_model is odd, the last column will be sine)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Register as buffer so it moves with the model to GPU/CPU but isn't a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        seq_len = x.shape[-2]
        
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor
            return x + self.pe[:seq_len, :].unsqueeze(0)
        else:
            return x + self.pe[:seq_len, :]


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
                

# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, num_heads, d_ff, num_classes, num_layers, dropout=0.1, use_token_type_embeddings=True):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        # raise Exception("Implement me")
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_token_type_embeddings = use_token_type_embeddings

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        self.position_embedding = SinusoidalPositionalEncoding(self.d_model, self.num_positions)

        if use_token_type_embeddings:
            self.token_type_embedding = nn.Embedding(2, d_model)  # For sentence A/B


        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, num_positions,dropout = dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
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
        

    def forward(self, indices , attention_mask=None, token_type_ids=None):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # raise Exception("Implement me")
        
        x = self.embedding(indices.to(device))

        # Add token type embeddings if provided
        if token_type_ids is not None and self.use_token_type_embeddings:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            x+= token_type_embeddings


        x = self.position_embedding(x)
        
        x = self.dropout(x)
        
        attention_maps = []
        for transformer_layer in self.transformer_layers:
            x, A = transformer_layer(x, attention_mask)
            attention_maps.append(A)
            

        cls_output = x[:, 0, :] 
            
        logits = self.classifier(cls_output)
        
        return logits, attention_maps
        

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



def train_sine_classifier_glue(train, val, num_epochs, save_path, num_positions = 512, num_classes =2 ,  loss = "CE",use_token_type_embeddings =True):
    """
    Train the complex transformer classifier
    """
    # Initialize the complex transformer model

    model_config = {
        'vocab_size': tokenizer.vocab_size,  # BERT vocab size, adjust based on your tokenizer
        'num_positions': num_positions,
        'd_model': 256,
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
        lr = 2e-5

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if loss == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    total_steps = len(train) * num_epochs
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
            
            if loss == "MSE":
                logits = logits.squeeze(-1)
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
                    logits = logits.squeeze(-1)
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
   

def decode_sine_glue(state_dict, dataloader,num_positions = 512, num_classes =2 , loss = "CE",  use_token_type_embeddings =True):
    model_config = {
        'vocab_size': tokenizer.vocab_size,  # BERT vocab size, adjust based on your tokenizer
        'num_positions': num_positions,
        'd_model': 256,
        'num_heads': 8,
        'd_ff': 1024,
        'num_classes': num_classes,  # Adjust based on task (2 for binary classification)
        'num_layers': 6,
        'dropout': 0.2,
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
                pred = logits.squeeze(-1)            
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.squeeze(-1).cpu().numpy())


    if loss =="CE":
        # print(preds)
        e = accuracy_score(targets, preds)
        g = f1_score(targets, preds)
        print(f"Accuracy:{np.round(e,5)}, F1 score {np.round(g,5)}")
        return e
    else:
        print(preds)
        spearman_corr, _ = spearmanr(preds, targets)
        print(f"Spearman correlation: {spearman_corr:.5f}")
        print(f"FLOPS mean: {np.mean(flops):.5f}, FLOPS var: {np.std(flops):.5f}")
        return spearman_corr




def attention_plots_sine(state_dict, valdata,save_path, num_positions = 512, num_classes =2 , use_token_type_embeddings =True):
    
    model_config = {
        'vocab_size': tokenizer.vocab_size,  # BERT vocab size, adjust based on your tokenizer
        'num_positions': num_positions,
        'd_model': 256,
        'num_heads': 8,
        'd_ff': 1024,
        'num_classes': num_classes,  # Adjust based on task (2 for binary classification)
        'num_layers': 6,
        'dropout': 0.2,
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
                    plt.savefig(f"{save_path}/sine_sample{i}_attn_{j}.png")
                    plt.close(fig)
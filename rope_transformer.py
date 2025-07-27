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


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)
        
        

class RoPEEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=20):
        """
        Rotary Position Embedding (RoPE)
        :param d_model: dimension of the model
        :param max_seq_len: maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        
        # Create frequency matrix
        freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))
        self.register_buffer("freqs", freqs)
        
        # Precompute position encodings
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)
        
        # Create cos and sin matrices
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k):
        """
        Apply rotary position embedding to query and key tensors
        :param q: query tensor [seq_len, d_model]
        :param k: key tensor [seq_len, d_model]
        :return: rotated q and k tensors
        """
        seq_len = q.shape[0]
        
        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len]  # [seq_len, d_model//2]
        sin = self.sin_cached[:seq_len]  # [seq_len, d_model//2]
        
        # Repeat cos and sin to match d_model dimensions
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [seq_len, d_model]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [seq_len, d_model]
        
        # Apply rotation
        q_rotated = q * cos + self.rotate_half(q) * sin
        k_rotated = k * cos + self.rotate_half(k) * sin
        
        return q_rotated, k_rotated


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # raise Exception("Implement me")
        
        self.d_model = d_model
        self.d_k = d_internal
        self.d_v = d_internal
        
        self.W_q =  nn.Linear(self.d_model ,self.d_k, bias = False)  #torch.rand(self.d_model ,self.d_k, requires_grad=True)
        self.W_k =  nn.Linear(self.d_model ,self.d_k, bias = False)  #torch.rand(self.d_model ,self.d_k, requires_grad=True)
        self.W_v =  nn.Linear(self.d_model ,self.d_v, bias = False)  # torch.rand(self.d_model ,self.d_v, requires_grad=True)
        
        # RoPE embedding
        self.rope = RoPEEmbedding(self.d_k)
        
        self.linear1 = nn.Linear(self.d_v, d_model)
        self.linear2 = nn.Linear(self.d_model, 32)
        self.linear3 = nn.Linear(32, self.d_model)
        
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        
        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, input_vecs):
        """
        :param input_vecs: an input tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the log probabilities of each position in the input
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """
        # raise Exception("Implement me")
        
        x = input_vecs
        

        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Apply RoPE to queries and keys
        Q, K = self.rope.apply_rotary_pos_emb(Q, K)
        
        attention_map = torch.matmul(Q, K.t())
        H = torch.matmul(self.softmax(attention_map/np.sqrt(self.d_k)), V)
        H = self.linear1(H)
        
        
        # Add and Norm
        Z1 = self.layernorm(x+H)
        
        # Linear layers
        Z2 = self.relu(self.linear2(Z1))

        Z2 = self.linear3(Z2)
        
        # Add norm
        Z2 = self.layernorm(Z2+Z1)

        return Z2  , attention_map      
        
        
        
        
        

# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
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
        self.d_k = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.transformerlayer = TransformerLayer(self.d_model, self.d_k)
        
        self.linear1 = nn.Linear(self.d_model, 32)
        self.linear2 = nn.Linear(32, self.num_classes)
        
        self.softmax = nn.Softmax()
        
        

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # raise Exception("Implement me")
        
        x = self.embedding(indices)
        
        attention_maps = []
        for i in range(self.num_layers):
            x, A = self.transformerlayer(x)
            attention_maps.append(A)
            
            
        Z = self.linear1(x)
        Z = torch.log(self.softmax(self.linear2(Z)))
        
        return Z, attention_maps
        
                  
        
# This is a skeleton for train_classifier: you can implement this however you want
def train_rope_classifier(args, train, dev, num_epochs):
    # raise Exception("Not fully implemented yet")

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(vocab_size = 27 , num_positions = 20, d_model = 64, d_internal = 64, num_classes = 3, num_layers = 4)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    loss_epochs = []
    for t in range(0, num_epochs):
        print("epoch {}".format(t))
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            x, y = train[ex_idx].input_tensor, train[ex_idx].output_tensor
            y_pred, mask = model(x)
            loss = loss_fcn(y_pred,y) # TODO: Run forward and compute loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print("epoch train loss {}".format(loss_this_epoch))
        loss_epochs.append(loss_this_epoch)


    model.eval()
    return model, loss_epochs


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode_rope(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("rope_plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))

    return float(num_correct) / num_total

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


# class ComplexEmbeddings(nn.Module):
#     def __init__(self, vocab_size, d_model, num_positions):
#         super().__init__()
#         self.vocab_embed = nn.Embedding(vocab_size, d_model)  # Content (real part)
#         self.pos_embed = nn.Embedding(num_positions, d_model)  # Position (imaginary/phase part)
#         self.d_model = d_model

#     def forward(self, x):  
#         # Content embeddings (real part)
#         real = self.vocab_embed(x)

#         seq_len = real.shape[-2]
        
#         # Positional embeddings (imaginary part for phase encoding)
#         positions = torch.arange(seq_len, dtype=torch.long)
#         imag = self.pos_embed(positions)

#         # param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        
#         return torch.complex(real, imag)  # [seq_len, d_model]

class ComplexEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, num_positions, gamma: float = 1.0):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.gamma = gamma

        position_encoding_dim = d_model
        freqs = 1.0 / (10000 ** (torch.arange(0, position_encoding_dim, 2).float() / position_encoding_dim)) # shape ()
        
        # Fixed: Don't duplicate frequencies, just use them for pairs
        self.register_buffer('omega', freqs)

    def forward(self, x):
        real = self.vocab_embed(x)

        seq_len = real.shape[-2]
        
        positions = torch.arange(seq_len, dtype=torch.float, device=real.device).unsqueeze(-1)

        angles = self.omega.unsqueeze(0).to(real.device) * positions

        imag = self.gamma * torch.sin(angles).repeat_interleave(2, dim=-1)

        if real.dim() == 3:
            imag = imag.unsqueeze(0).expand(real.shape[0], -1, -1)
        
        return torch.complex(real, imag) # [seq_len, d_model]
    

class PhaseAwareAttention(nn.Module):
    def __init__(self, variant, d_model, d_internal, alpha: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_internal
        self.d_v = d_internal
        self.variant  = variant 
        
        # Complex-linear projections - using real linear layers but applying to complex inputs
        self.q_proj_real = nn.Linear(d_model, d_internal, bias=False)
        self.q_proj_imag = nn.Linear(d_model, d_internal, bias=False)
        self.k_proj_real = nn.Linear(d_model, d_internal, bias=False)
        self.k_proj_imag = nn.Linear(d_model, d_internal, bias=False)

        #TODO
        self.v_proj_real = nn.Linear(d_model, d_internal, bias=False)
        self.v_proj_imag = nn.Linear(d_model, d_internal, bias=False)
        
        
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = alpha
        
    def forward(self, z):  # z: complex [seq_len, d_model]
        # Extract real and imaginary parts
        z_real = z.real
        z_imag = z.imag
        
        # Complex queries and keys
        q_real = self.q_proj_real(z_real) - self.q_proj_imag(z_imag)  #(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
        q_imag = self.q_proj_real(z_imag) + self.q_proj_imag(z_real)
        q = torch.complex(q_real, q_imag)
        
        k_real = self.k_proj_real(z_real) - self.k_proj_imag(z_imag)
        k_imag = self.k_proj_real(z_imag) + self.k_proj_imag(z_real)
        k = torch.complex(k_real, k_imag)
        
        #TODO
        v_real = self.v_proj_real(z_real) - self.v_proj_imag(z_imag)
        v_imag = self.v_proj_real(z_imag) + self.v_proj_imag(z_real)
        v = torch.complex(v_real, v_imag)

        
        # Phase-aware attention scores (complex dot product)
        attn_scores = torch.matmul(q, k.conj().transpose(-2, -1))
        # attn_magnitude = torch.abs(attn_scores)  # Magnitude
        # attn_phase = torch.angle(attn_scores)    # Phase

        # attn_scores_real = (attn_magnitude/torch.max(attn_magnitude) + self.alpha * torch.cos(attn_phase) ) / np.sqrt(self.d_k)  # Use magnitude only

        if self.variant == "magnitude":
            scores = torch.abs(attn_scores) / np.sqrt(self.d_k)
        elif self.variant == "phase":
            scores = torch.cos(torch.angle(attn_scores))/ np.sqrt(self.d_k)
        elif self.variant == "real":
            scores = attn_scores.real / np.sqrt(self.d_k)
        elif self.variant == "hybrid_norm":
            mag = torch.abs(attn_scores)
            phase = torch.cos(torch.angle(attn_scores))
            scores = (mag / mag.max()) + self.alpha * phase
            scores = scores / np.sqrt(self.d_k)
        elif self.variant == "hybrid":
            mag = torch.abs(attn_scores)
            phase = torch.cos(torch.angle(attn_scores))
            scores = (mag + self.alpha * phase) / np.sqrt(self.d_k)
        elif self.variant == "apatt":  #named from paper 
            magnitude = torch.abs(attn_scores)
            sign = attn_scores / (magnitude + 1e-8)  # Complex sign
            scores = magnitude / np.sqrt(self.d_k)

            v = sign.unsqueeze(-2) * v.unsqueeze(-3)
            
        elif self.variant == "riatt":
            attn_real = torch.real(attn_scores) / np.sqrt(self.d_k)
            attn_imag = torch.imag(attn_scores) / np.sqrt(self.d_k)

            attn_probs_real = self.softmax(attn_real)
            attn_probs_imag = self.softmax(attn_imag)

            scores = torch.complex(attn_probs_real, attn_probs_imag)

            #TODO cant plot complex scores 
        

        attn_weights = self.softmax(scores)
        
        # Attend to real values
        output = torch.matmul(attn_weights.to(torch.complex64) , v)
        
        return output, attn_weights

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
        
        
class ComplexTransformerLayer(nn.Module):
    def __init__(self,variant, d_model, d_internal,alpha):
        """
        :param d_model: The dimension of the inputs and outputs of the layer
        :param d_internal: The "internal" dimension used in the self-attention computation
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_internal
        
        self.attention = PhaseAwareAttention(variant, d_model, d_internal,alpha)
        
        self.linear1_real = nn.Linear(d_internal, d_model)
        self.linear1_imag = nn.Linear(d_internal, d_model)
        self.linear2 = nn.Linear(d_model, 32)
        self.linear3 = nn.Linear(32, d_model)
        
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, input_vecs):
        """
        :param input_vecs: complex tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the processed features
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """
        z = input_vecs
        
        # Phase-aware attention
        H, attention_map = self.attention(z)
        H_real = self.linear1_real(H.real) - self.linear1_imag(H.imag)
        H_imag = self.linear1_real(H.imag) + self.linear1_imag(H.real)
        
        # Add and Norm (using real part for residual connection)
        # use real part for layer norm but maintain complex structure
        Z1_real = self.layernorm1(z.real +  H.real)
        Z1 = torch.complex(Z1_real, z.imag + H.imag)

        
        # Feed-forward network - apply to real part only for simplicity
        Z2 = self.relu(self.linear2(Z1_real))
        Z2 = self.linear3(Z2)
        
        # Add and Norm
        Z2 = self.layernorm2(Z2 + Z1_real)


        # return Z2, attention_map
        #TODO phase information is maintained
        return torch.complex(Z2,Z1.imag) ,attention_map


# Should contain your overall Transformer implementation with complex embeddings
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers,gamma_pe: float = 1.0, alpha_attn: float = 0.2):
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
        self.d_k = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.variant = "hybrid" 
        
        # Complex embeddings instead of regular embeddings + positional encoding
        self.complex_embedding = ComplexEmbeddings(self.vocab_size, self.d_model, self.num_positions, gamma=gamma_pe)
        
        #TODO
        # Create multiple transformer layers
        self.transformer_layers = nn.ModuleList([
            ComplexTransformerLayer(self.variant,self.d_model, self.d_k, alpha=alpha_attn) 
            for _ in range(self.num_layers)
        ])

        

        self.linear1 = nn.Linear(2*self.d_model, 32)
        self.linear2 = nn.Linear(32, self.num_classes)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Complex embeddings (encodes both content and positional information via phase)
        Z = self.complex_embedding(indices)
        
        attention_maps = []

        Z, A = self.first_layer(Z)
        attention_maps.append(A)

        for transformer_layer in self.transformer_layers:
            Z, A = transformer_layer(Z)
            attention_maps.append(A)
            
        # Final classification layers
        Z = torch.cat([Z.real, Z.imag], dim=-1)  # Concatenate real and imaginary
        Z = self.linear1(Z)
        Z = torch.log(self.softmax(self.linear2(Z)))
        
        return Z, attention_maps



def train_complex_classifier(args, train, dev):
    """
    Train the complex transformer classifier
    """
    # Initialize the complex transformer model
    model = Transformer(vocab_size=27, num_positions=20, d_model=64, d_internal=64, num_classes=3, num_layers=4)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 20
    loss_epochs = []
    for t in range(0, num_epochs):
        print("epoch {}".format(t))
        loss_this_epoch = 0.0
        random.seed(t)
        
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        
        for ex_idx in ex_idxs:
            x, y = train[ex_idx].input_tensor, train[ex_idx].output_tensor
            y_pred, attention_maps = model(x)
            loss = loss_fcn(y_pred, y)
            
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
def decode_complex(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
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
                plt.savefig("complex_plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))

    return float(num_correct) / num_total
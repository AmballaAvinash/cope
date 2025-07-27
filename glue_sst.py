# letter_counting.py

import argparse
import json
import time
import numpy as np
from utils import *
from complex_transformer_single_realV import train_complex_classifier, decode_complex
from learned_transformer import train_learned_classifier, decode_learned, LetterCountingExample
from rope_transformer import train_rope_classifier, decode_rope
from sine_transformer import train_sine_classifier, decode_sin
import pandas as pd 

import matplotlib.pyplot as plt

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("glue", "sst2")
train_texts = dataset["train"]["sentence"]
train_labels = dataset["train"]["label"]
val_texts = dataset["validation"]["sentence"]
val_labels = dataset["validation"]["label"]


train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=64)

class SST2Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {
            "labels": torch.tensor(self.labels[idx])
        }
    def __len__(self):
        return len(self.labels)

train_dataset = SST2Dataset(train_encodings, train_labels)
val_dataset = SST2Dataset(val_encodings, val_labels)

# 4. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)



print(train_dataset[0])
print(val_dataset[0])
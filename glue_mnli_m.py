# letter_counting.py

import argparse
import json
import time
import numpy as np
from utils import *
from complex_transformer_single_realV import train_complex_classifier_glue, decode_complex_glue, attention_plots_complex
from learned_transformer import train_learned_classifier_glue, decode_learned_glue,  attention_plots_learned
from rope_transformer import train_rope_classifier_glue, decode_rope_glue,  attention_plots_rope
from sine_transformer import train_sine_classifier_glue, decode_sine_glue, attention_plots_sine
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

dataset = load_dataset("glue", "mnli_matched")


train_sentences1 = list(dataset["train"]["premise"])
train_sentences2 = list(dataset["train"]["hypothesis"])
train_labels = dataset["train"]["label"]

val_sentences1 = list(dataset["validation"]["premise"])
val_sentences2 = list(dataset["validation"]["hypothesis"])
val_labels = dataset["validation"]["label"]

# 2. Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. Tokenize paired inputs
train_encodings = tokenizer(train_sentences1, train_sentences2, truncation=True, padding="max_length", max_length=256)
val_encodings = tokenizer(val_sentences1, val_sentences2, truncation=True, padding="max_length", max_length=256)


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
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)


print(len(train_dataset))
print(len(val_dataset))


N_exp = 1
num_epochs = 30
mode = "train" # ["train", "test", "attention"]

# train
if mode == "train": 
    for variant in ["real", "magnitude", "phase", "hybrid", "hybrid_norm"]: #,
            train_loss_complex = []
            val_loss_complex = []
            # acc_complex = []

            print(f"format {variant}")

            for i in range(N_exp):
                print("Complex Experiment {}".format(i))
                model, train_loss, val_loss = train_complex_classifier_glue(train = train_loader, val = val_loader, num_classes = 3, num_epochs = num_epochs,variant = variant, save_path = "models_mnli_m/complex_{}_state_dict.pth".format(variant), use_token_type_embeddings =True)
                train_loss_complex.append(train_loss)
                val_loss_complex.append(val_loss)

                # test_acc = decode_complex_glue(model, val_loader)
                # acc_complex.append(test_acc)

            train_loss_complex = np.array(train_loss_complex)
            val_loss_complex = np.array(val_loss_complex)

            # acc_complex = np.array(acc_complex)
            df = pd.DataFrame(train_loss_complex, columns=[f"Epoch_{i+1}" for i in range(train_loss_complex.shape[1])])
            df.to_csv(f"loss_mnli_m/train_loss_complex_glue_{variant}.csv", index=False)

            df = pd.DataFrame(val_loss_complex, columns=[f"Epoch_{i+1}" for i in range(val_loss_complex.shape[1])])
            df.to_csv(f"loss_mnli_m/val_loss_complex_glue_{variant}.csv", index=False)
            # print("Complex valued accuracy , mean {} and std {}".format(np.mean(acc_complex), np.std(acc_complex)))
    



    # rope
    train_loss_rope = []
    val_loss_rope = []
    # acc_rope =[]
    for i in range(N_exp):
        print("ROPE Experiment {}".format(i))
        model, train_loss, val_loss = train_rope_classifier_glue(train = train_loader, val = val_loader,  num_classes = 3,  num_epochs = num_epochs, save_path = "models_mnli_m/rope_state_dict.pth",use_token_type_embeddings =True)
        train_loss_rope.append(train_loss)
        val_loss_rope.append(val_loss)
    
        # test_acc = decode_rope_glue(model, val_loader)
        # acc_rope.append(test_acc)

    train_loss_rope = np.array(train_loss_rope)
    val_loss_rope = np.array(val_loss_rope)

    # acc_rope = np.array(acc_rope)
    df = pd.DataFrame(train_loss_rope, columns=[f"Epoch_{i+1}" for i in range(train_loss_rope.shape[1])])
    df.to_csv("loss_mnli_m/train_loss_rope_glue.csv", index=False)

    df = pd.DataFrame(val_loss_rope, columns=[f"Epoch_{i+1}" for i in range(val_loss_rope.shape[1])])
    df.to_csv("loss_mnli_m/val_loss_rope_glue.csv", index=False)
    # print("ROPE valued accuracy , mean {} and std {}".format(np.mean(acc_rope), np.std(acc_rope)))




    # Learned 
    train_loss_learned = []
    val_loss_learned = []

    # acc_learned =[]
    for i in range(N_exp):
        print("Learned Experiment {}".format(i))
        model, train_loss, val_loss = train_learned_classifier_glue(train = train_loader, val = val_loader,  num_classes = 3, save_path="models_mnli_m/learned_state_dict.pth", num_epochs = num_epochs, use_token_type_embeddings =True)
        train_loss_learned.append(train_loss)
        val_loss_learned.append(val_loss)


        # test_acc = decode_learned_glue(model, val_loader)
        # acc_learned.append(test_acc)

    train_loss_learned = np.array(train_loss_learned)
    val_loss_learned = np.array(val_loss_learned)
    # acc_learned = np.array(acc_learned)
    df = pd.DataFrame(train_loss_learned, columns=[f"Epoch_{i+1}" for i in range(train_loss_learned.shape[1])])
    df.to_csv(f"loss_mnli_m/train_loss_learned_glue.csv", index=False)
    df = pd.DataFrame(val_loss_learned, columns=[f"Epoch_{i+1}" for i in range(val_loss_learned.shape[1])])
    df.to_csv(f"loss_mnli_m/val_loss_learned_glue.csv", index=False)
    # print("Learned valued accuracy , mean {} and std {}".format(np.mean(acc_learned), np.std(acc_learned)))



    # Sine 
    train_loss_sine = []
    val_loss_sine = []
    # acc_sine=[]
    for i in range(N_exp):
        print("SINE Experiment {}".format(i))
        model, train_loss, val_loss = train_sine_classifier_glue(train = train_loader, val = val_loader,  num_classes = 3, num_epochs = num_epochs, save_path = "models_mnli_m/sine_state_dict.pth", use_token_type_embeddings =True)
        train_loss_sine.append(train_loss)
        val_loss_sine.append(val_loss)
        
        # test_acc = decode_sine_glue(model, val_loader)
        # acc_sine.append(test_acc)

    train_loss_sine = np.array(train_loss_sine)
    val_loss_sine = np.array(val_loss_sine)
    # acc_sine = np.array(acc_sine)
    df = pd.DataFrame(train_loss_sine, columns=[f"Epoch_{i+1}" for i in range(train_loss_sine.shape[1])])
    df.to_csv(f"loss_mnli_m/train_loss_sine_glue.csv", index=False)

    df = pd.DataFrame(val_loss_sine, columns=[f"Epoch_{i+1}" for i in range(val_loss_sine.shape[1])])
    df.to_csv(f"loss_mnli_m/val_loss_sine_glue.csv", index=False)
    # print("Sine valued accuracy , mean {} and std {}".format(np.mean(acc_sine), np.std(acc_sine)))


elif mode == "test":

    # attention plots
    for variant in ["real", "magnitude", "phase", "hybrid", "hybrid_norm"]: #,
        print(f"CoPE {variant}")

        state_dict = torch.load("models_mnli_m/complex_{}_state_dict.pth".format(variant))
        _ = decode_complex_glue(state_dict, val_loader, variant, num_classes = 3 )

    print("RoPE")
    state_dict = torch.load("models_mnli_m/rope_state_dict.pth")
    _ = decode_rope_glue(state_dict, val_loader, num_classes = 3)

    print("learned")
    state_dict = torch.load("models_mnli_m/learned_state_dict.pth")
    _ = decode_learned_glue(state_dict, val_loader, num_classes = 3)

    print("sine")
    state_dict = torch.load("models_mnli_m/sine_state_dict.pth")
    _ = decode_sine_glue(state_dict, val_loader, num_classes = 3)



elif mode == "attention":
    # attention plots
    for variant in ["real", "magnitude", "phase", "hybrid", "hybrid_norm"]: #,
        state_dict = torch.load("models_mnli_m/complex_{}_state_dict.pth".format(variant))
        attention_plots_complex(state_dict, val_dataset[0:1], variant, save_path = "attention_plots_mnli_m")

    state_dict = torch.load("models_mnli_m/rope_state_dict.pth")
    attention_plots_rope(state_dict, val_dataset[0:1], save_path = "attention_plots_mnli_m")

    state_dict = torch.load("models_mnli_m/learned_state_dict.pth")
    attention_plots_learned(state_dict, val_dataset[0:1], save_path = "attention_plots_mnli_m")

    state_dict = torch.load("models_mnli_m/sine_state_dict.pth")
    attention_plots_sine(state_dict, val_dataset[0:1],  save_path = "attention_plots_mnli_m")

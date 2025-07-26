# letter_counting.py

import argparse
import json
import time
import numpy as np
from utils import *
from complex_transformer import train_complex_classifier, decode_complex
from learned_transformer import train_learned_classifier, decode_learned, LetterCountingExample
from rope_transformer import train_rope_classifier, decode_rope
from sine_transformer import train_sine_classifier, decode_sin

import matplotlib.pyplot as plt

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--task', type=str, default='BEFORE', help='task to run (BEFORE or BEFOREAFTER)')
    parser.add_argument('--train', type=str, default='data/lettercounting-train.txt', help='path to train examples')
    parser.add_argument('--dev', type=str, default='data/lettercounting-dev.txt', help='path to dev examples')
    parser.add_argument('--output_bundle_path', type=str, default='classifier-output.json', help='path to write the results json to (you should not need to modify)')
    args = parser.parse_args()
    return args


def read_examples(file):
    """
    :param file:
    :return: A list of the lines in the file, each exactly 20 characters long
    """
    all_lines = []
    for line in open(file):
        all_lines.append(line[:-1]) # eat the \n
    print("%i lines read in" % len(all_lines))
    return all_lines


def get_letter_count_output(input: str, count_only_previous: bool=True) -> np.array:
    """
    :param input: The string
    :param count_only_previous: True if we should only count previous occurrences, False for all occurrences
    :return: the output for the letter-counting task as a numpy array of 0s, 1s, and 2s
    """
    output = np.zeros(len(input))
    for i in range(0, len(input)):
        if count_only_previous:
            output[i] = min(2, len([c for c in input[0:i] if c == input[i]]))
        else:
            output[i] = min(2, len([c for c in input if c == input[i]]) - 1)  # count all *other* instances of input[i]
    return output


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    # Constructs the vocabulary: lowercase letters a to z and space
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    count_only_previous = True if args.task == "BEFORE" else False

    # Constructs and labels the data
    train_exs = read_examples(args.train)
    train_bundles = [LetterCountingExample(l, get_letter_count_output(l, count_only_previous), vocab_index) for l in train_exs]
    dev_exs = read_examples(args.dev)
    dev_bundles = [LetterCountingExample(l, get_letter_count_output(l, count_only_previous), vocab_index) for l in dev_exs]


    # trails

    N_exp = 1
    
    # loss_learned = []
    # acc_learned =[]
    # for i in range(N_exp):
    #     print("Learned Experiment {}".format(i))
    #     model, loss_epochs = train_learned_classifier(args, train_bundles, dev_bundles)
    #     loss_learned.append(loss_epochs)
    #     # Deco,des t he first 5 dev examples to display as output
    #     small_test_acc = decode_learned(model, dev_bundles[0:5], do_print=True, do_plot_attn=True)
    #     # Decodes 100 training examples and the entire dev set (1000 examples)
    #     print("Training accuracy (100 exs):")
    #     train_acc =  decode_learned(model, train_bundles[0:100])
    #     print("Dev accuracy (whole set):")
    #     test_acc = decode_learned(model, dev_bundles)
    #     acc_learned.append(test_acc)

    # loss_learned = np.array(loss_learned)
    # acc_learned = np.array(acc_learned)



    # loss_sine = []
    # acc_sine=[]
    # for i in range(N_exp):
    #     print("SINE Experiment {}".format(i))
    #     model, loss_epochs = train_sine_classifier(args, train_bundles, dev_bundles)
    #     loss_sine.append(loss_epochs)
    #     # Deco,des t he first 5 dev examples to display as output
    #     small_test_acc = decode_sin(model, dev_bundles[0:5], do_print=True, do_plot_attn=True)
    #     # Decodes 100 training examples and the entire dev set (1000 examples)
    #     print("Training accuracy (100 exs):")
    #     train_acc =  decode_sin(model, train_bundles[0:100])
    #     print("Dev accuracy (whole set):")
    #     test_acc = decode_sin(model, dev_bundles)
    #     acc_sine.append(test_acc)

    # loss_sine = np.array(loss_sine)
    # acc_sine = np.array(acc_sine)


    
    loss_complex = []
    acc_complex = []

    for i in range(N_exp):
        print("Complex Experiment {}".format(i))
        model, loss_epochs = train_complex_classifier(args, train_bundles, dev_bundles)
        loss_complex.append(loss_epochs)
        # Deco,des the first 5 dev examples to display as output
        small_test_acc = decode_complex(model, dev_bundles[0:5], do_print=True, do_plot_attn=True)
        # Decodes 100 training examples and the entire dev set (1000 examples)
        print("Training accuracy (100 exs):")
        train_acc =  decode_complex(model, train_bundles[0:100])
        print("Dev accuracy (whole set):")
        test_acc = decode_complex(model, dev_bundles)
        acc_complex.append(test_acc)

    loss_complex = np.array(loss_complex)
    acc_complex = np.array(acc_complex)


    loss_rope = []
    acc_rope =[]
    for i in range(N_exp):
        print("ROPE Experiment {}".format(i))
        model, loss_epochs = train_rope_classifier(args, train_bundles, dev_bundles)
        loss_rope.append(loss_epochs)
        # Deco,des t he first 5 dev examples to display as output
        small_test_acc = decode_rope(model, dev_bundles[0:5], do_print=True, do_plot_attn=True)
        # Decodes 100 training examples and the entire dev set (1000 examples)
        print("Training accuracy (100 exs):")
        train_acc =  decode_rope(model, train_bundles[0:100])
        print("Dev accuracy (whole set):")
        test_acc = decode_rope(model, dev_bundles)
        acc_rope.append(test_acc)

    loss_rope = np.array(loss_rope)
    acc_rope = np.array(acc_rope)
    

    # prints
    # print("Sine valued accuracy , mean {} and std {}".format(np.mean(acc_sine), np.std(acc_sine)))
    print("ROPE valued accuracy , mean {} and std {}".format(np.mean(acc_rope), np.std(acc_rope)))
    # print("Learned valued accuracy , mean {} and std {}".format(np.mean(acc_learned), np.std(acc_learned)))
    print("Complex valued accuracy , mean {} and std {}".format(np.mean(acc_complex), np.std(acc_complex)))

    
    # plots
# plt.errorbar(
#     range(1, len(loss_epochs) + 1),
#     np.mean(loss_learned, 0),
#     yerr=np.std(loss_learned, 0),
#     fmt='^',
#     label="Learned Valued",
#     color='purple',
#     linestyle="--"
# )

# ROPE
plt.errorbar(
    range(1, len(loss_epochs) + 1),
    np.mean(loss_rope, 0),
    yerr=np.std(loss_rope, 0),
    fmt='o',
    label="ROPE Valued",
    color='darkorange',
    linestyle="-"
)

# # Sine
# plt.errorbar(
#     range(1, len(loss_epochs) + 1),
#     np.mean(loss_sine, 0),
#     yerr=np.std(loss_sine, 0),
#     fmt='s',  # Square marker
#     label="Sine Valued",
#     color='teal',
#     linestyle=":"
# )

# Complex
plt.errorbar(
    range(1, len(loss_epochs) + 1),
    np.mean(loss_complex, 0),
    yerr=np.std(loss_complex, 0),
    fmt='d',  # Diamond marker
    label="Complex Valued",
    color='crimson',
    linestyle="-.")  # Dash-dot

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epochs")
plt.grid(True)
plt.savefig("train.png")



import numpy as np
from nltk.tokenize import word_tokenize
import re


def tokenize(text):
    text = re.sub(r'[,!?;-]', '.', text)
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() or token == '.']
    return tokens


def generate_vocab(tokens):
    vocab = dict()

    for i, token in enumerate(set(tokens)):
        vocab[token] = i

    return vocab


def one_hot_encode(token, vocab):
    vector = np.zeros(len(vocab))
    vector[vocab[token]] = 1
    return vector


def generate_train_data(tokens, vocab, context_size=2):
    V = len(vocab)
    X, Y = [], []

    for i, token in enumerate(tokens):
        if i % 1000 == 0:
            print(f'At token i={i}')
        context = tokens[i-context_size:i] + tokens[i+1:i+context_size+1]
        context_vector = np.zeros(V)

        for word in context:
            context_vector += np.array(one_hot_encode(word, vocab))
        context_vector = context_vector / len(context)
        center_vector = one_hot_encode(tokens[i], vocab)

        X.append(context_vector)
        Y.append(center_vector)

    return np.array(X), np.array(Y)


def generate_batch_data(tokens, vocab, context_size=2, batch_size=128):
    batches = 0

    while True:
        batch_tokens = tokens[batches*batch_size:(batches+1)*batch_size]
        if len(batch_tokens) < batch_size:
            break
        else:
            yield generate_train_data(batch_tokens, vocab, context_size)
            batches += 1

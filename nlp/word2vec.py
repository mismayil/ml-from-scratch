import numpy as np
from nltk.tokenize import word_tokenize
import re
from matplotlib import pyplot


def one_hot_encode(token, vocab):
    vector = np.zeros(len(vocab))
    vector[vocab[token]] = 1
    return vector.astype(int)


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


def generate_cbow_train_data(tokens, vocab, context_size=2):
    vocab_size = len(vocab)

    for i, token in enumerate(tokens):
        context = tokens[i-context_size:i] + tokens[i+1:i+context_size+1]
        context_vector = np.zeros(vocab_size)
        
        for word in context:
            context_vector += np.array(one_hot_encode(word, vocab))
        context_vector = context_vector / len(context)
        target_vector = one_hot_encode(tokens[i], vocab)

        yield context_vector, target_vector


def generate_cbow_batch_data(tokens, vocab, context_size=2, batch_size=128):
    def data_generator():
        batch_x = []
        batch_y = []

        for x, y in generate_cbow_train_data(tokens, vocab, context_size):
            if len(batch_x) < batch_size:
                batch_x.append(x)
                batch_y.append(y)
            else:
              yield np.array(batch_x), np.array(batch_y)
              batch_x = []
              batch_y = []

    return data_generator


def ohe_to_word(vector, vocab):
    word_idx = np.where(vector == 1)[0].item()
    for word, index in vocab.items():
        if index == word_idx:
            return word


def plot_embeddings(embeddings, words):
    result = compute_pca(embeddings, 2)
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
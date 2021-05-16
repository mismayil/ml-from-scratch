import os
import re
from collections import namedtuple

from nlp.pos import *
DATA_FOLDER = '/content/drive/My Drive/Colab Notebooks/data/wsj'
TaggedWord = namedtuple('TaggedWord', ['word', 'tag'])
START_TAG = '--s--'

def preprocess_wsj(text):
    corpus = []

    for paragraph_text in text.split('=+'):
        corpus.append(TaggedWord(word='', tag=START_TAG))
        for line in paragraph_text.split('\n'):
            parts = line.split(' ')
            for part in parts:
                part = part.strip().strip(']').strip('[')
                if len(part) > 0 and not part.startswith('='):
                    subparts = part.split('/')
                    if len(subparts) > 2:
                        word, tag = subparts[0].strip('\\') + '/' + subparts[1], subparts[2]
                    else:
                        word, tag = subparts
                    corpus.append(TaggedWord(word=word, tag=tag))
    return corpus

corpus = []
for file in os.listdir(DATA_FOLDER):
    if file.startswith('wsj'):
        with open(DATA_FOLDER+f'/{file}') as f:
            subcorpus = preprocess_wsj(f.read())
            corpus.extend(subcorpus)

split_num = int(len(corpus) * 0.8)
training_corpus = corpus[:split_num]
test_corpus = corpus[split_num:]
vocab = generate_vocab(corpus)
emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
transition_matrix = create_transition_matrix(0.001, tag_counts, transition_counts)
emission_matrix = create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
states = sorted(tag_counts.keys())
best_probs, best_paths = initialize(states, tag_counts, transition_matrix, emission_matrix, corpus, vocab)
viterbi_forward(transition_matrix, emission_matrix, test_corpus, best_probs, best_paths, vocab)
pred = viterbi_backward(best_probs, best_paths, test_corpus, states)
compute_accuracy(pred, test_corpus)
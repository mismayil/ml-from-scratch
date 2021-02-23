from collections import defaultdict

import nltk
nltk.download('reuters')
nltk.download('punkt')

from nltk.corpus import reuters
from nlp.n_grams import preprocess_sentence, preprocess_text

text = reuters.raw()
sentences = preprocess_text(text)

unigrams = build_n_grams(sentences, n=1)
bigrams = build_n_grams(sentences, n=2)

vocab = [key[0] for key in unigrams.keys()]

N_GRAMS_MAP = [unigrams, bigrams]

def suggest_word(text, vocab, context_size=1):
    sentence = preprocess_sentence(text)
    max_prob = None
    probable_word = None

    for word in vocab:
        context = sentence[-context_size:]
        n_gram = context + (context_size - len(context)) * [START_TOKEN] + [word]
        n_gram_prob = estimate_prob_n_gram(n_gram,
                                           n_grams=N_GRAMS_MAP[context_size],
                                           n_minus_1_grams=N_GRAMS_MAP[context_size-1],
                                           vocab_size=len(vocab))
        if max_prob is None or max_prob < n_gram_prob:
            max_prob = n_gram_prob
            probable_word = word
    
    return probable_word

print(suggest_word('move against the', vocab))


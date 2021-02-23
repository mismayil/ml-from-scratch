from collections import defaultdict

START_TOKEN = '<s>'
END_TOKEN = '<e>'

def preprocess_sentence(sentence):
    return [word.lower() for word in word_tokenize(sentence)]

def preprocess_text(text):
    return [preprocess_sentence(sentence) for sentence in sent_tokenize(text)]

def build_n_grams(sentences, n=2):
    n_grams = defaultdict(int)

    for sentence in sentences:
        sentence = [START_TOKEN] * (n-1) + sentence + [END_TOKEN]

        for i in range(len(sentence)-n+1):
            n_grams[tuple(sentence[i:i+n])] += 1
    
    return n_grams

def estimate_prob_n_gram(n_gram, n_grams, n_minus_1_grams, vocab_size, k=1):
    n_minus_1_gram = tuple(n_gram[:-1])
    return (n_grams.get(tuple(n_gram), 0) + k) / (n_minus_1_grams.get(n_minus_1_gram, 0) + k * vocab_size)

def calculate_perplexity(sentence, n_grams, n_minus_1_grams, vocab):
    n = len(list(n_grams.keys())[0])
    sentence = preprocess_sentence(sentence)
    sentence = [START_TOKEN] * (n-1) + sentence + [END_TOKEN]
    N = len(sentence)
    sentence_prob = 1

    for i in range(len(sentence)-n+1):
        prob = estimate_prob_n_gram(sentence[i:i+n], n_grams, n_minus_1_grams, len(vocab))
        sentence_prob *= prob
    
    return (1 / sentence_prob) ** (1 / N)
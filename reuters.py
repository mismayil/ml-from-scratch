import nltk
from nltk.corpus import reuters

from nn import Model, Linear, Activation, CrossEntropyLoss
from word2vec import tokenize, generate_vocab, generate_batch_data

nltk.download('reuters')
text = reuters.raw()
tokens = tokenize(text)
vocab = generate_vocab(tokens)

vocab_size = len(vocab)
embedding_size = 10
model = Model()
model.add(Linear(vocab_size, embedding_size))
model.add(Activation('relu'))
model.add(Linear(embedding_size, vocab_size))

model.set_loss(CrossEntropyLoss)

for X, y in generate_batch_data(tokens, vocab):
    model.train(X, y, epochs=1)

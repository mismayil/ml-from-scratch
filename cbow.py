import nltk
from nltk.corpus import reuters

from nn import Model, Linear, Activation, CrossEntropyLoss
from word2vec import tokenize, generate_vocab, generate_cbow_batch_data, plot_embeddings

with open(f'./data/shakespeare.txt') as f:
  text = f.read()

tokens = tokenize(text)
vocab = generate_vocab(tokens)

vocab_size = len(vocab)
embedding_size = 50
model = Model()
model.add(Linear(vocab_size, embedding_size))
model.add(Activation('relu'))
model.add(Linear(embedding_size, vocab_size))

model.set_loss(CrossEntropyLoss)

data_generator = generate_cbow_batch_data(tokens, vocab, batch_size=128)
model.train(data_generator, epochs=5000, learning_rate=0.05)
model.save()

embeddings = 0.5 * (model.layers[0].weights + model.layers[2].weights.T)

words = ['man', 'woman', 'king', 'queen']
idx = [vocab[word] for word in words]
plot_embeddings(embeddings[idx, :], words)

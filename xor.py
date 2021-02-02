from nn import Model, Linear, Activation, MSELoss
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def data_generator():
    yield X, y

model = Model()
model.add(Linear(2, 3))
model.add(Activation('tanh'))
model.add(Linear(3, 1))
model.add(Activation('tanh'))

model.set_loss(MSELoss)
model.train(data_generator, epochs=1000, learning_rate=0.1)

predictions = model.predict(X)
print(predictions)

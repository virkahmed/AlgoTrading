import datetime
import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from astropy.logger import level
from tf_keras.models import Sequential
from tf_keras.layers import Flatten, Dense
import yfinance as yf
from datetime import datetime,timedelta
import pickle

if not os.path.exists("trainSoB.pkl"):
    aapl = yf.download(tickers='AAPL', interval="1m", period="1d", start="2025-06-02", end="2025-06-03",
                       auto_adjust=False)
    with open("trainSoB.pkl", "wb") as f:
        pickle.dump(aapl, f)
else:
    with open("trainSoB.pkl", "rb") as f:
        aapl = pickle.load(f)





df = aapl['Close'][::10].stack()
apl = aapl['Close'].stack()

x_train = []
y_train = []



for i in list(df.index)[1:]:
    apnd = []
    for j in range(1,10):
        try:
            apnd.append(apl[i[0]- timedelta(minutes=j)][i[1]])
        except KeyError:
            print("nah")
    x_train.append(apnd)
    if apl[i]-apl[i[0]- timedelta(minutes=1)][i[1]] < 0:
        y_train.append(0)
    else:
        y_train.append(1)


x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')

if not os.path.exists("testSoB.pkl"):
    aapl = yf.download(tickers='AAPL', interval="1m", period="1d", start="2025-06-03", end="2025-06-04",
                       auto_adjust=False)
    with open("testSoB.pkl", "wb") as f:
        pickle.dump(aapl, f)
else:
    with open("testSoB.pkl", "rb") as f:
        aapl = pickle.load(f)

df = aapl['Close'][::10].stack()
apl = aapl['Close'].stack()


x_test = []
y_test = []

for i in list(df.index)[1:]:
    apnd = []
    for j in range(1,10):
        try:
            apnd.append(apl[i[0]- timedelta(minutes=j)][i[1]])
        except KeyError:
            print("nah")
    x_test.append(apnd)
    if apl[i]-apl[i[0]- timedelta(minutes=1)][i[1]] < 0:
        y_test.append(0)
    else:
        y_test.append(1)

x_test = np.array(x_test).astype('float32')
y_test = np.array(y_test).astype('float32')


def min_max_normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)
x_train = min_max_normalize(x_train)
y_train = min_max_normalize(y_train)

x_test = min_max_normalize(x_test)
y_test = min_max_normalize(y_test)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

model = Sequential([
    Flatten(input_shape = (9,1)),
    Dense(64, activation = 'sigmoid'),
    Dense(32, activation = 'sigmoid'),
    Dense(1, activation = 'softmax')
])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
mod = model.fit(x_train, y_train, epochs = 10,
                batch_size = 19,
                validation_split = .5)
print(mod)

results = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss, Test accuracy:', results)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mod.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(mod.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(mod.history['loss'], label='Training Loss', color='blue')
plt.plot(mod.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)

plt.suptitle("Model Training Performance", fontsize=16)
plt.tight_layout()
plt.show()
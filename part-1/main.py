import pandas as pd
import csv
import pickle as pkl
import numpy as np
from fasttext import FastText
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime

model = FastText.load_model('cc.en.300.bin')


class NetworkModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(300, 150)
        self.linear_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = self.input(x)
        x = self.linear_stack(x)
        x = self.output(x)
        return x



class Embeddings:
    def __init__(self, model) -> None:
        self.model = model
    def get_embeddings(self, words):
        return np.array([self.model.get_word_vector(i) for i in words])

embedding_class = Embeddings(model)


anew_dataset = pd.read_csv('./anew.csv')

anew_dict= dict()
words = anew_dataset['term']
valence_ratings = anew_dataset['pleasure']
arousal_ratings = anew_dataset['arousal']
word_embeddings = embedding_class.get_embeddings(words)
word_embedding_dict = dict(zip(words, word_embeddings))
for i in range(len(words)):
    anew_dict[words[i]] = [valence_ratings[i], arousal_ratings[i]]


X = word_embeddings
Y = np.array([anew_dict[i] for i in words])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

test_dataset = TensorDataset(X_test, y_test)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model_train = NetworkModel().to(device)
temp = input("Enter number of epochs: ")
if(temp.isdecimal()):
    epochs = int(temp)
else:
    epochs = 10000
learning_rate = 0.0003
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)
test_loss = []
train_loss = []
for i in range(epochs):
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        Y_pred = model_train(X.float())
        loss = loss_fn(Y_pred, Y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {i}, Training Loss: {loss.item()}')
    train_loss.append(loss.item())
    # Test Loss
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            Y_pred = model_train(X.float())
            test_loss_fn = loss_fn(Y_pred, Y.float())
        print(f'Epoch {i}, Test Loss: {test_loss_fn.item()}')
        test_loss.append(test_loss_fn.item())

plt.plot(train_loss[1000:], label='Training Loss')
plt.plot(test_loss[1000:], label='Test Loss')
plt.legend()
datestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
plt.title(datestring)
plt.savefig(f'./train_logs/{datestring}.png')
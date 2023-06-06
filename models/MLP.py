from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout


class MLP(Module):
    def __init__(self, n_inputs, dropout_rate):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 64)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.dropout1 = Dropout(dropout_rate)

        # second hidden layer
        self.hidden2 = Linear(64, 32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.dropout2 = Dropout(dropout_rate)

        # third hidden layer
        self.hidden3 = Linear(32, 16)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.dropout3 = Dropout(dropout_rate)

        # fourth hidden layer and output
        self.hidden4 = Linear(16, 1)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.dropout1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.dropout2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.dropout3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        return X



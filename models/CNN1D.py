import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, n_features, hidden_size, dropout_rate):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(hidden_size * 4 * (n_features // 8), hidden_size * 2)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1, 495)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

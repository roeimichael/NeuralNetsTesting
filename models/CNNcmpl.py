import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class ComplexCNN(nn.Module):
    def __init__(self, n_features, dropout_rate):
        super(ComplexCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate the size of the input to the first fully connected layer
        fc_input_size = 128 * (n_features // (2 ** 3))  # 2^3 because of 3 maxpooling layers

        self.fc1 = nn.Linear(fc_input_size, 64)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1, 495)
        out = checkpoint(self.layer1, x)
        out = checkpoint(self.layer2, out)
        out = checkpoint(self.layer3, out)

        out = out.view(out.size(0), -1)  # Flattening for the fully connected layer

        out = self.fc1(out)
        out = self.relu4(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out

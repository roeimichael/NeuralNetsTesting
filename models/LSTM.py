import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.view(x.size(0), 1, 495)

        device = x.device  # Get the device of the input tensor
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

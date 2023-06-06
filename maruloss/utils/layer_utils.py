from torch import nn


print(f"loaded {__name__}")

class ViewLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class TemperatureLayer(nn.Module):
    def __init__(self, temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        return (x * 1-self.temperature) + (0.5 * self.temperature)



import torch
from torch import nn

# https://www.researchgate.net/publication/312170477
class Expert1(nn.Module):
    def __init__(self):
        super(Expert1, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# test model correctness
if __name__ == '__main__':
    model = Expert1()
    test_input = torch.ones((64, 3, 32, 32))
    test_output = model(test_input)
    print(test_output.shape)

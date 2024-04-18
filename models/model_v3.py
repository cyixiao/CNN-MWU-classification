import torch
from torch import nn

# https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79
class Expert3(nn.Module):
    def __init__(self):
        super(Expert3, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    model = Expert3()
    test_input = torch.ones((64, 3, 32, 32))
    test_output = model(test_input)
    print(test_output.shape)

import torch
import torch.nn as nn

# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
class Expert4(nn.Module):
    def __init__(self):
        super(Expert4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    model = Expert4()
    test_input = torch.ones((64, 3, 32, 32))
    test_output = model(test_input)
    print(test_output.shape)

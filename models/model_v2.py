import torch
from torch import nn


# https://appliedmachinelearning.wordpress.com/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
class Expert2(nn.Module):
    def __init__(self):
        super(Expert2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = Expert2()
    test_input = torch.ones((64, 3, 32, 32))
    test_output = model(test_input)
    print(test_output.shape)

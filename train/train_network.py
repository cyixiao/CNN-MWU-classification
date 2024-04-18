import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from models.model_v1 import *
from models.model_v2 import *
from models.model_v3 import *
from models.model_v4 import *
model_list = [Expert1(), Expert2(), Expert3(), Expert4()]

# prepare dataset
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get dataset length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# gather dataset by using dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

for model_index in range(4):
    print("=================training model{}=================".format(model_index + 1))
    # get network model
    model = model_list[model_index]
    model.to(device)

    # use cross entropy loss as our main loss function
    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)

    # Construct Optimizer, us SGD
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    writer_path = "../logs/model{}_train_log".format(model_index + 1)
    # track training process
    writer = SummaryWriter(writer_path)

    # track train steps
    total_train_step = 0
    # track test steps
    total_test_step = 0
    # train rounds
    epoch = 50

    # start training for `epoch` times
    for i in range(epoch):
        print("--------------round{}--------------".format(i + 1))
        # training part
        model.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_function(outputs, targets)

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 500 == 0:
                print("model{}: training step: {}, loss: {}".format(model_index + 1, total_train_step, loss.item()))
                scalar_name = "model{} train_loss".format(model_index + 1)
                writer.add_scalar(scalar_name, loss.item(), total_train_step)

        # test part
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_function(outputs, targets)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy

        test_loss_scalar = "model{} test loss".format(model_index + 1)
        test_accuracy_scalar = "model{} test accuracy".format(model_index + 1)
        writer.add_scalar(test_loss_scalar, total_test_loss, total_test_step)
        writer.add_scalar(test_accuracy_scalar, total_accuracy / test_data_size, total_test_step)
        total_test_step += 1

    torch.save(model, "../trained_models/saved_model{}.pth".format(model_index + 1))
    print("Model{} has been successfully saved to '../trained_models/'".format(model_index + 1))

    writer.close()

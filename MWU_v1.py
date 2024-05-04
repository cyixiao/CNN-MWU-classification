import torch
import torchvision
import numpy
from torch.utils.data import DataLoader, Subset
from models.model_v1 import *
from models.model_v2 import *
from models.model_v3 import *
from models.model_v4 import *
from tqdm import tqdm

# set training data for MWU

mwu_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
total_entries = len(mwu_data)
subset_indices = list(range(total_entries - 9000, total_entries))  # Last 9000 indices

data_subset = Subset(mwu_data, subset_indices)
mwu_dataloader = DataLoader(data_subset, batch_size=1, shuffle=False)
total_num = len(data_subset)
print(total_num)

# get saved weight
weights_path = "trained_models/final_mwu_weights.pth"
weights = torch.load(weights_path, map_location=torch.device('cpu'))
print(weights)

# weights = [0.5, 0.8, 0.7, 0.4]

# set expert
if torch.cuda.is_available():
    device = torch.device('cuda')
    weights = torch.tensor(weights, device=device)
else:
    device = torch.device('cpu')

expert1 = torch.load("trained_models/saved_model1.pth", map_location=device)
expert2 = torch.load("trained_models/saved_model2.pth", map_location=device)
expert3 = torch.load("trained_models/saved_model3.pth", map_location=device)
expert4 = torch.load("trained_models/saved_model4.pth", map_location=device)
experts = numpy.array([expert1, expert2, expert3, expert4])

for expert in experts:
    expert.eval()

mwu_predictions = 0
expert1_predictions = 0
expert2_predictions = 0
expert3_predictions = 0
expert4_predictions = 0

with torch.no_grad():
    for data in tqdm(mwu_dataloader, total=total_num, desc="Processing", unit="img"):
        img, target = data
        img = img.to(device)
        target = target.to(device)
        outputs = [expert(img) for expert in experts]
        p = numpy.zeros(10)
        for i in range(4):
            p[outputs[i].argmax(1)] += weights[i]

        # find the class with the highest cumulative weight
        final_prediction = numpy.argmax(p)

        # check if the prediction is correct
        if final_prediction == target:
            mwu_predictions += 1

        # compute accuracy for each expert
        prediction1 = expert1(img).argmax(1)
        if prediction1 == target:
            expert1_predictions += 1

        prediction2 = expert2(img).argmax(1)
        if prediction2 == target:
            expert2_predictions += 1

        prediction3 = expert3(img).argmax(1)
        if prediction3 == target:
            expert3_predictions += 1

        prediction4 = expert4(img).argmax(1)
        if prediction4 == target:
            expert4_predictions += 1

expert1_accuracy = expert1_predictions / total_num
expert2_accuracy = expert2_predictions / total_num
expert3_accuracy = expert3_predictions / total_num
expert4_accuracy = expert4_predictions / total_num
accuracy = mwu_predictions / total_num

print("Accuracy of Expert 1: {}".format(expert1_accuracy))
print("Accuracy of Expert 2: {}".format(expert2_accuracy))
print("Accuracy of Expert 3: {}".format(expert3_accuracy))
print("Accuracy of Expert 4: {}".format(expert4_accuracy))
print("Accuracy of MWU system: {}".format(accuracy))

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset
from models.model_v1 import *
from models.model_v2 import *
from models.model_v3 import *
from models.model_v4 import *
from tqdm import tqdm

# Each expert provides a set of probabilities for each category, then combine these probabilities multiplied by their
# weights into a single set of probabilities, which is used to make the final prediction.

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

expert1 = torch.load("trained_models/saved_model1.pth", map_location=device)
expert2 = torch.load("trained_models/saved_model2.pth", map_location=device)
expert3 = torch.load("trained_models/saved_model3.pth", map_location=device)
expert4 = torch.load("trained_models/saved_model4.pth", map_location=device)
experts = np.array([expert1, expert2, expert3, expert4])

for expert in experts:
    expert.eval()

mwu_predictions = 0
expert_predictions = [0] * len(experts)

with torch.no_grad():
    for data in tqdm(mwu_dataloader, total=total_num, desc="Processing", unit="img"):
        img, target = data
        img = img.to(device)
        target = target.to(device)
        outputs = torch.stack([expert(img) for expert in experts])

        # weighted experts output probabilities, not the prediction
        weighted_outputs = torch.einsum('i,ijk->jk', weights_tensor, torch.softmax(outputs, dim=2))

        # get the final prediction
        final_prediction = torch.argmax(weighted_outputs, dim=1)

        # check if the prediction is correct
        if final_prediction == target:
            mwu_predictions += 1

        # compute accuracy for each expert
        for i, expert in enumerate(experts):
            expert_prediction = torch.argmax(outputs[i], dim=1)
            if expert_prediction == target:
                expert_predictions[i] += 1

# Calculate and print accuracies
expert_accuracies = [_ / total_num for _ in expert_predictions]
for i, accuracy in enumerate(expert_accuracies, 1):
    print(f"Accuracy of Expert {i}: {accuracy}")
print(f"Accuracy of MWU system: {mwu_predictions / total_num}")

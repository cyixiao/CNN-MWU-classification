import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from models.model_v1 import *
from models.model_v2 import *
from models.model_v3 import *
from models.model_v4 import *

# set training data for MWU
mwu_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
mwu_dataloader = DataLoader(mwu_data, batch_size=1, shuffle=True)
T = len(mwu_data)

# set expert
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

expert1 = torch.load("../trained_models/saved_model1.pth", map_location=device)
expert2 = torch.load("../trained_models/saved_model2.pth", map_location=device)
expert3 = torch.load("../trained_models/saved_model3.pth", map_location=device)
expert4 = torch.load("../trained_models/saved_model4.pth", map_location=device)
experts = np.array([expert1, expert2, expert3, expert4])

for expert in experts:
    expert.eval()
# set parameters
m = 4
T = 1000
epsilon = 0.01
rho = 1
print(epsilon)

# dictionaries of weights, multipliers, and losses
w = [1 for _ in range(m)]
p = [0 for _ in range(m)]
loss = [0 for _ in range(m)]
count = 1
# MWU loop
with torch.no_grad():
    for data in tqdm(mwu_dataloader, total=T, desc="Processing", leave=True):
        img, target = data
        img = img.to(device)
        target = target.to(device)
        outputs = [expert(img) for expert in experts]
        # compute loss function
        for i in range(m):
            if outputs[i].argmax(1) == target:
                loss[i] = -0.1
            else:
                loss[i] = 1
            # weight update
            w[i] = w[i] * (1 - epsilon * (loss[i] / rho))

        # ============= cross entropy loss function =============
        # for i, output in enumerate(outputs):
        #     loss = F.cross_entropy(output, target)  # Remove unsqueeze if not needed
        #     w[i] = w[i] * torch.exp(-epsilon * loss)
        # =======================================================

        count += 1
        if count >= T:
            break
weights_path = "../trained_models/final_mwu_weights.pth"
torch.save(w, weights_path)
print("weights: {}".format(w))

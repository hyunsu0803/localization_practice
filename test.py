from dataset import TestDataset
from model import CNN
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os


model = torch.load('model_output.pt')
model.eval()
test_dataset = TestDataset('./phasemap_samples2')
test_dataloader = DataLoader(test_dataset, batch_size=16)

device = torch.device('cuda')
criterion = nn.BCELoss()
avg_cost = 0
total_batch_num = len(test_dataloader)

for b_x, b_y in test_dataloader:
    logits = model(b_x.to(device))
    loss = criterion(logits, b_y.to(device))

    # save images of logits
    logits = np.array(logits.clone().detach().cpu())    # (B, 37, 126)
    avg_logits = np.sum(logits, axis=0, keepdims=False) #/ batch_size    # (37, 126)
    target = np.array(b_y.clone().detach().cpu())          # (B, 37, 126)
    
    plt.subplot(2, 1, 1)
    plt.imshow(target[0, :, :], vmin=0.0, vmax=1.0,)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(avg_logits,)# vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.savefig("test.png")
    
    # get accuracy & MAE
    estimated_doa = np.argmax(logits, axis=1)               # (B, 37, 126) => (B, 126)
    true_doa = np.argmax(target, axis=1)                    # (B, 37, 126) => (B, 126)
    vad = np.max(target, axis=1)                            # (B, 37, 126) => (B, 126)
    
    difference = vad * np.abs(true_doa - estimated_doa)           # (B, 126)
    correctness = vad * (difference <= np.ones_like(difference))    # (B, 126)
    n_correct = np.count_nonzero(correctness)
    n_active_frame = np.count_nonzero(vad)
    mae = 5 * np.sum(difference) / n_active_frame     

    print('Accuracy : %d / %d' % (n_correct, n_active_frame))
    print('MAE :', mae)

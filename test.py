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
    cpu_logits = np.array(logits.clone().detach().cpu())    # (B, 37, 126)
    avg_logits = np.sum(cpu_logits, axis=0, keepdims=False) #/ batch_size    # (37, 126)
    cpu_b_y = np.array(b_y.clone().detach().cpu())          # (B, 37, 126)

    plt.subplot(2, 1, 1)
    plt.imshow(cpu_b_y[0, :, :], vmin=0.0, vmax=1.0,)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(avg_logits,)# vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.savefig("test.png")
    
    # get accuracy
    argmax_logits = np.argmax(cpu_logits, axis=1)       # (B, 37, 126) => (B, 126)
    argmax_b_y = np.argmax(cpu_b_y, axis=1)             # (B, 37, 126) => (B, 126)
    difference = np.abs(argmax_b_y - argmax_logits)     # (B, 126)
    a = difference <= np.ones_like(difference)          # (B, 126)
    if np.max(argmax_b_y) > np.min(argmax_b_y):     # if the target is not the DOA 0
        a = a * argmax_b_y                          # vad masking
    n_correct = np.count_nonzero(a)
    n_active_frame = np.count_nonzero(argmax_b_y)

    print('Accuracy : %d / %d' % (n_correct, n_active_frame))

from dataset import PhaseMapDataset, ValidationDataset
from model import CNN
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random

seed = 777
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
    
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

batch_size = 16
train_path = "./phasemap_samples"

train_dataset = PhaseMapDataset(train_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = ValidationDataset(train_path)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

model = CNN().to(device).train()
optimizer = optim.Adam(model.parameters(), lr=0.001,)# weight_decay=1.0e-6)
criterion = nn.BCELoss()

epochs = 40
val_interval = 1

min_val_cost = 99
min_val_cost_epoch = -1
model.train()
for epoch in range(epochs):
    it = 0
    model.train()
    avg_cost = 0
    total_batch_num = len(train_dataloader)
    for b_x, b_y in train_dataloader:
        it += 1
        logits = model(b_x.to(device))
        loss = criterion(logits, b_y.to(device))

        avg_cost += loss / total_batch_num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if it % 10 == 0:
            # save images of logits
            cpu_logits = np.array(logits.clone().detach().cpu())    # (B, 37, 126)
            avg_logits = np.sum(cpu_logits, axis=0, keepdims=False) / batch_size    # (37, 126)
            cpu_b_y = np.array(b_y.clone().detach().cpu())
            plt.subplot(2, 1, 1)
            plt.imshow(cpu_b_y[0, :, :], vmin=0.0, vmax=1.0,)
            plt.colorbar()
            plt.subplot(2, 1, 2)
            plt.imshow(avg_logits,)# vmin=0.0, vmax=1.0)
            plt.colorbar()
            
            plt.savefig("./target_n_logits3/%d_%d.png" % (epoch+1, it//10))
            
            # get accuracy
            max_logits = np.argmax(cpu_logits, axis=1)  # (B, 37, 126) => (B, 126)
            max_b_y = np.argmax(cpu_b_y, axis=1)        # (B, 37, 126) => (B, 126)
            difference = np.abs(max_b_y - max_logits)   # (B, 126)
            # print(difference.shape)
            a = difference <= np.ones_like(difference)
            n_correct = np.count_nonzero(a)
            total = difference.shape[0] * difference.shape[1]
            print('Accuracy : %d / %d' % (n_correct, total))
    
    print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))
    
    if epoch % val_interval == 0:
        model.eval()
        val_cost = 0
        total_valbatch_num = len(validation_dataloader)
        for b_x, b_y in validation_dataloader:
            logits = model(b_x.to(device))
            loss = criterion(logits, b_y.to(device))

            val_cost += loss / total_valbatch_num
        print('Validation cost : {}'.format(val_cost))
    
        if val_cost < min_val_cost:
            min_val_cost = val_cost
            torch.save(model, 'model_output.pt')
            min_val_cost_epoch = epoch+1

print("best score : ", min_val_cost_epoch)
        



from dataset import PhaseMapDataset
from model import CNN
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

batch_size = 128
train_path = "./phasemap_samples"

train_dataset = PhaseMapDataset(train_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = CNN().to(device).train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 30

model.train()
for epoch in range(epochs):
    model.train()
    avg_cost = 0
    total_batch_num = len(train_dataloader)

    for b_x, b_y in train_dataloader:
        logits = model(b_x.to(device))
        loss = criterion(logits, b_y.to(device))

        avg_cost += loss / total_batch_num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 1))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 1))
        self.bn3 = nn.BatchNorm2d(num_features=16)
        
        self.fc1 = nn.Linear(257*16, 64)
        # self.fc1 = nn.Linear(44800, 512)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 37)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        print(x.shape)
        exit(1)
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        # print("x3.shape", x3.shape)
        x3 = x3.reshape(x3.shape[0], -1)
        # print("x3 reshape", x3.shape)
       
        x4 = self.relu(self.fc1(x3))
        # x5 = self.relu(self.fc2(x4))
        x6 = self.sigmoid(self.out(x4))
        # print("after sigmoid", x6)
        return x6


def main():
    train_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
    test_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

    batch_size = 128
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
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
        

if __name__ == '__main__':
    main()
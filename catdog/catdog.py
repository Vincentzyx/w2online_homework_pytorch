import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from PIL import Image

class CatDogData(torch.utils.data.Dataset):
    def __init__(self, root, transform, datatype="train"):
        self.imgFiles = list(sorted(os.listdir(os.path.join(root, datatype))))
        self.datatype = datatype
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.datatype, self.imgFiles[idx])).convert("RGB")
        label = self.imgFiles[idx].split(".")[0]
        if self.transform:
            imgArr = self.transform(img)
        else:
            imgArr = np.array(img)
            imgArr = imgArr / 255.0
            imgArr = np.transpose(imgArr, (2, 0, 1))
        sample = {"img": imgArr, "label": label}
        return sample

    def __len__(self):
        return len(self.imgFiles)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*61*61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).cuda()
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))        
        x = x.view([x.size(0), 16*61*61])
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        # x = F.softmax(x)
        return x

transform = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 

UseGPU = True
device = torch.device('cuda:0')
classes = ("dog", "cat")
net = Net()
if UseGPU:
    net = net.to(device)
dataset = CatDogData(r"", transform)
test_data = CatDogData(r"", transform, "test")
train_loader = torch.utils.data.DataLoader(dataset,shuffle=True, batch_size=512)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True)
CrossEntropy = nn.CrossEntropyLoss()

if UseGPU:
    CrossEntropy = CrossEntropy.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if os.path.exists("weights.pkl"):
    net.load_state_dict(torch.load('weights.pkl'))

for epoch in range(1000):
    stepCount = 0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        inputs = data["img"]
        labels = data["label"]
        labelsVec = []
        for l in labels:
            labelsVec.append(0 if l == "dog" else 1)
        labelsVec = np.array(labelsVec)
        labelsVec = torch.from_numpy(labelsVec).long()
        labelsVec = torch.tensor(labelsVec, dtype=torch.long)
        # print(labelsVec.shape)

        optimizer.zero_grad()
        if UseGPU:
            labelsVec = labelsVec.to(device)
            inputs = inputs.to(device)

        outputs = net(inputs)        
        loss = CrossEntropy(outputs, labelsVec)
        loss.backward()
        optimizer.step()

        stepCount+=1
        running_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / stepCount))
            running_loss = 0.0
            stepCount = 0
        if i % 50 == 0:
            torch.save(net.state_dict(), 'weights.pkl')
            correctCount = 0
            totalCount = 0
            for testitem in iter(test_loader):
                result = net(testitem["img"])
                result = torch.argmax(result)
                target = 0 if "dog" in testitem["label"] else 1
                totalCount+=1
                if target == result:
                    correctCount+=1
                if totalCount > 500:
                    break
            print("Acc:" + str(correctCount / totalCount))

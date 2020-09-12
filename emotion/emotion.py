import os
import numpy as np
from skimage.util import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from PIL import Image

class TextEmotionData(torch.utils.data.Dataset):
    def __init__(self, root, datatype="train"):
        rawText = ""
        self.data = []
        self.chars = []
        lCount = 0
        with open(os.path.join(root, "weibo_senti_100k.csv"), 'r', encoding="utf-8") as f:
            rawText = f.read()
            for line in rawText.split("\n"):
                lCount += 1
                self.data.append([int(line[:1]), line[2:]])
        if os.path.exists("chars.txt"):
            with open("chars.txt", 'r', encoding="utf-8") as f:
                self.chars = sorted(list(set(f.read())))
        else:
            self.chars = sorted(list(set(rawText)))
            with open("chars.txt", 'w', encoding="utf-8") as f:
                f.write("".join(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        if datatype == "train":
            self.data = self.data[2000:]
        else:
            self.data = self.data[:2000]
        print(lCount, " Lines")
        print(len(self.chars), " Chars")
        

    def __getitem__(self, idx):
        text = self.data[idx][1]
        if len(text) > 32:
            text = text[:32]
        textVec = np.zeros(32, dtype=np.int64)
        for i, c in enumerate(text):
            textVec[i] = self.char_indices[c]
        textVec = torch.from_numpy(textVec)
        sample = {"text": textVec, "label": self.data[idx][0]}
        return sample

    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.embedding = nn.Embedding(5931, 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.view(32, batch_size, -1)
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = x.view([batch_size, -1])
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

UseGPU = True
device = torch.device('cuda:0')
net = Net()
if UseGPU:
    net = net.to(device)
train_data = TextEmotionData("")
test_data = TextEmotionData("", "test")
train_loader = torch.utils.data.DataLoader(train_data,shuffle=True, batch_size=4096)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True)
CrossEntropy = nn.CrossEntropyLoss()

if UseGPU:
    CrossEntropy = CrossEntropy.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if os.path.exists("weights.pkl"):
    net.load_state_dict(torch.load('weights.pkl'))

for epoch in range(10000):
    stepCount = 0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        inputs = data["text"]
        labels = data["label"]
        labelsVec = []
        for l in labels:
            labelsVec.append(l)
        labelsVec = np.array(labelsVec, dtype=np.int64)
        labelsVec = torch.from_numpy(labelsVec)
        # labelsVec = torch.tensor(labelsVec, dtype=torch.long)

        optimizer.zero_grad()
        if UseGPU:
            labelsVec = labelsVec.to(device)
            inputs = inputs.to(device)
        outputs = net(inputs)
        # labelsVec = labelsVec.unsqueeze(1)
        loss = CrossEntropy(outputs, labelsVec)
        # print(outputs.shape, labelsVec.shape)
        loss.backward()
        optimizer.step()
        stepCount+=1
        running_loss += loss.item()
        if i % 20 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / stepCount))
            running_loss = 0.0
            stepCount = 0
        if i % 200 == 0:
            torch.save(net.state_dict(), 'weights.pkl')
            correctCount = 0
            totalCount = 0
            for testitem in iter(test_loader):
                text = testitem["text"].to(device)
                # print(text.shape)
                result = net(text).cpu()
                result = torch.argmax(result)
                target = testitem["label"]
                # print(result, "-", target)
                if target == result:
                    correctCount+=1
                totalCount+=1
                if totalCount > 500:
                    break
            print("Acc:" + str(correctCount / totalCount))

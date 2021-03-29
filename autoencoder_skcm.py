from __future__ import print_function
import time
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from sklearn import preprocessing

parList = sys.argv
indexSample = int(parList[1])
indexScore = int(parList[2])
indexGroup = int(parList[3])
epochs = int(parList[4])
learning_rate = float(parList[5])

if indexGroup == 1:
    num_hidden1 = 1000
    num_hidden2 = 100
    num_feature = 20
else:
    num_hidden1 = 400
    num_hidden2 = 100
    num_feature = 20

sampleType = ["metastatic_", "All_Symbol_"]
scoreType = ["lymphocyte_", "purity_"]
groupName = ["high", "middle", "low"]
combineName = sampleType[indexSample] + scoreType[indexScore] + groupName[indexGroup]
inputName = "SKCM_RNAseqv3_" + combineName + ".csv"
outputName = "autoencoder_" + combineName + "_" + str(num_feature) + "_" + str(learning_rate) + ".pth"

rna_df = pd.read_csv(inputName, sep="\t")
rna = np.array(rna_df)

num_input = len(rna)

if np.max(rna) > 10000:
    rna = np.log10(rna + 1)

rna = rna.T
rna = preprocessing.scale(rna)

print(sum(rna[:, 0]))
print(rna.shape)
print(len(rna))
print(combineName)
print(outputName)
print("hidden layer: " + str(num_hidden1) + ',' + str(num_hidden2))
print(learning_rate)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, num_hidden1),
            nn.ReLU(True),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(True),
            nn.Linear(num_hidden2, num_feature))

        self.decoder = nn.Sequential(
            nn.Linear(num_feature, num_hidden2),
            nn.ReLU(True),
            nn.Linear(num_hidden2, num_hidden1),
            nn.ReLU(True),
            nn.Linear(num_hidden1, num_input),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = AutoEncoder().cuda()
# model = autoencoder()
criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.KLDivLoss()
# criterion = nn.binary_cross_entropy()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

startTime = time.time()
print("start training!")
for epoch in range(epochs):
    for sample in rna:
        sample = Variable(torch.FloatTensor(sample)).cuda()
        # ===================forward=====================
        output = model(sample)
        loss = criterion(output, sample)
        # loss = nn.functional.binary_cross_entropy(output, sample)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.8f}'.format(epoch + 1, epochs, loss.data), time.time() - startTime)

torch.save(model.state_dict(), outputName)

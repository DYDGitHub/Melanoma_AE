from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import sys
from torch import nn
from torch.autograd import Variable
from sklearn import preprocessing


parList = sys.argv
indexSample = int(parList[1])
indexScore = int(parList[2])
indexGroup = int(parList[3])
# learning_rate = float(parList[4])

# indexSample = 0
# indexScore = 0
# indexGroup = 0
learning_rate = 0.00001

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
featureName = "AutoEncoder_features_" + combineName + "_" + str(num_feature) + "_" + str(learning_rate) + ".txt"
featureGeneFile = "AE_feature_gene_" + combineName + "_" + str(num_feature) + "_" + str(learning_rate) + ".txt"


rna_df = pd.read_csv(inputName, sep="\t")
rna = np.array(rna_df)

num_input = len(rna)

if np.max(rna) > 1000:
    rna = np.log10(rna + 1)

rna = rna.T
rna = preprocessing.scale(rna)

TSS = sum([np.linalg.norm(i, 2) for i in rna])
print(TSS)

print(np.linalg.norm(rna[0], 2))

print(sum(rna[:, 0]))
print(rna.shape)
print(len(rna))
print(combineName)
print("hidden layer: " + str(num_hidden1) + ',' + str(num_hidden2))
print(featureName)
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


model = AutoEncoder()
model.load_state_dict(torch.load(outputName, map_location='cpu'))


RSS = 0.0
for sample in rna:
    sample = Variable(torch.FloatTensor(sample))
    output = model.forward(sample)
    RSS += torch.norm(output - sample)

RSS = float(RSS.data)
print("R: ", 1 - (RSS / TSS))


for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

stateDict = model.state_dict()

print(list(stateDict.keys()))
matrix_1 = stateDict["encoder.0.weight"]
matrix_2 = stateDict["encoder.2.weight"]
matrix_3 = stateDict["encoder.4.weight"]


print(matrix_1.shape)
print(matrix_2.shape)
print(matrix_3.shape)

# feature_gene = np.zeros((num_feature, num_input))
# print(feature_gene.shape)

featureGeneMat = torch.mm(torch.mm(matrix_3, matrix_2), matrix_1)
featureGeneMat = np.array(featureGeneMat)
print(featureGeneMat.shape)


np.savetxt(featureGeneFile, featureGeneMat, delimiter='\t')

import torch.nn as nn
import torch


class AnomalyDetectorNet(nn.Module):
    def __init__(self,no_of_features):
        super(AnomalyDetectorNet, self).__init__()
        self.linear1 = nn.Linear(no_of_features,no_of_features*2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(no_of_features*2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, targets_train ):
        out = self.linear1(targets_train)
        out  = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        
        return out


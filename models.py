## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn

# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
            

        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv1_bn = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        # dim: conv4 -> (26-3)/1+1 = 24 -> maxpool -> 12
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv4_bn = nn.BatchNorm2d(256)
        
     
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(12*12*256, 4000)
        self.fc1_bn = nn.BatchNorm1d(4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000,136)
        self.dropout = nn.Dropout(0.2)
    
        
    def forward(self, x):
       
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        
        x = x.reshape(x.size(0),-1)
        
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)  
        
        return x

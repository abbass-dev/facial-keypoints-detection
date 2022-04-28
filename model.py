import torch
import torch.nn.functional as F
import torch.nn as nn
class FaceNet(nn.Module):
    def __init__(self,number_points=16):
        super(FaceNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,3))
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3))
        self.conv4 = nn.Conv2d(64,128,kernel_size=(3,3),padding=1)
        self.conv5 = nn.Conv2d(128,256,kernel_size=(3,3),padding=1)

        self.dense1 = nn.Linear(256*22*22,500)
        self.dense2 = nn.Linear(500,500)
        self.dense3 = nn.Linear(500,number_points*2)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.dropout(x,p=0.2)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.dropout(x,p=0.2)
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

if __name__=="__main__":
    f = FaceNet()
    face = FaceNet()
    print(face)
    image = torch.ones((4,1,96,96))
    x = face(image)
    print(x.shape)

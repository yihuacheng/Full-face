import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
#from Torch.ntools import VGG16

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        
        alexnet = torchvision.models.alexnet(pretrained=True)

        self.convNet = alexnet.features

        self.weightStream = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        self.FC = nn.Sequential(
            nn.Linear(256*13*13, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

        #self._init_weights(self.weightStream)

    def forward(self, x_in):
        faceFeature = self.convNet(x_in['face'])
        weight = self.weightStream(faceFeature)
        
        faceFeature = weight * faceFeature

        faceFeature = torch.flatten(faceFeature, start_dim=1)
        gaze = self.FC(faceFeature)

        return gaze

    def _init_weights(self, net):
        k = 0
        for m in net.modules():    
            if isinstance(m, nn.Conv2d):
                if k ==0 or k == 1:
                  nn.init.normal_(m.weight, mean=0, std=0.01)
                  nn.init.constant_(m.bias, 0.1)
                if k == 2:
                  nn.init.normal_(m.weight, mean=0, std=0.001)
                  nn.init.constant_(m.bias, 1)
                if k > 2:
                  print("ERROR IN WEIGHT INITIAL")
                  exit()
                k += 1

if __name__ == '__main__':
    m = model().cuda()
    '''feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),
                "left":torch.zeros(10,1, 36,60).cuda(),
                "right":torch.zeros(10,1, 36,60).cuda()
              }'''
    feature = {"head_pose": torch.zeros(10, 2).cuda(),
               "left": torch.zeros(10, 3, 36, 60).cuda(),
               "right": torch.zeros(10, 3, 36, 60).cuda(),
               "face": torch.zeros(10, 3, 448, 448).cuda()
               }
    a = m(feature)
    print(m)
    print(a)


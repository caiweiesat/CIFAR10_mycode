import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main=nn.Sequential(
            # nn.Conv2d(3,32,kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            #
            # nn.Conv2d(32,64,kernel_size=3,padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            #
            # nn.Conv2d(64,128,kernel_size=3,padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Dropout(0.35),
            # nn.MaxPool2d(kernel_size=2,stride=2),

            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # Block1
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Dropout(),
            nn.MaxPool2d(2, 2),

            # Block2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Dropout(),
            nn.MaxPool2d(2, 2),

            # Block3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            # nn.Dropout(),
            nn.MaxPool2d(2, 2),

        )
        self.fc=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            # nn.Linear(128*4*4,1024),
            # nn.ReLU(),
            # nn.Dropout(),

            nn.Linear(256,10),
            # nn.ReLU(),
            # nn.Dropout(),
            #
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(),
            #
            # nn.Linear(128,10),

        )
    def forward(self,x):
        x=self.main(x)
        x=self.fc(x)
        return x

class MyNetSingleColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.main=nn.Sequential(
            # nn.Conv2d(1,32,kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            #
            # nn.Conv2d(32,64,kernel_size=3,padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            #
            # nn.Conv2d(64,128,kernel_size=3,padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2,stride=2),

            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # Block1
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),

            # Block2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2, 2),

            # Block3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.MaxPool2d(2, 2),
        )
        self.fc=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            # nn.Linear(128*4*4,1024),
            # nn.ReLU(),
            # nn.Dropout(),

            nn.Linear(256,10),
            # nn.ReLU(),
            # nn.Dropout(),
            #
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(),
            #
            # nn.Linear(128,10),

        )
    def forward(self,x):
        x=self.main(x)
        x=self.fc(x)
        return x
import torch
import torch.nn as nn


class PlantPathologyClassifier(nn.Module):
    def __init__(self, ngpu):
        super(PlantPathologyClassifier, self).__init__()
        self.ngpu = ngpu

        # input shape [3 x 1365 x 2048] going into convolutional
        self.main_conv = nn.Sequential(

            nn.Conv2d(3, 16, 4, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, 4, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(16, 32, 4, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 4, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(64, 64, 4, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 4, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2, 1),


            nn.Conv2d(128, 256, 4, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 4, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2, 1),
            nn.AdaptiveAvgPool2d((7,7))
        )


        # out shape [ 256, 1, 1, 1 ] from convolutional going into fully conected classifier
        self.fc = nn.Sequential(

            nn.Linear(12544, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(100, 4)
        )

    def forward(self, image):

        conv_out = self.main_conv(image).view(-1, 12544)
        fc_input = conv_out

        return self.fc(fc_input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

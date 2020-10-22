from torch import nn
import torch.nn.functional as F

class myConvNet(nn.Module):
    def __init__(self, output_size=2):
        super(myConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1) # in-channel, out-channel, kernel size, stride, padding
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pool =  nn.AvgPool2d(4)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 64 -> 32
        x = F.relu(self.conv2(x)) # 32 -> 16
        x = F.relu(self.conv3(x)) # 16 -> 8
        x = self.pool(F.relu(self.conv4(x))) # 8 -> 4 -> 1
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class svhnCNN(nn.Module):

      def __init__(self, top_bn=True):

            super(svhnCNN, self).__init__()
            self.top_bn = top_bn
            self.main = nn.Sequential(
                  nn.Conv2d(3, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.MaxPool2d(2, 2, 1),
                  nn.Dropout2d(),

                  nn.Conv2d(128, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.MaxPool2d(2, 2, 1),
                  nn.Dropout2d(),

                  nn.Conv2d(256, 512, 3, 1, 0, bias=False),
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(512, 256, 1, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 128, 1, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.AdaptiveAvgPool2d((1, 1))
                  )

            self.linear = nn.Linear(128, 10)
            self.bn = nn.BatchNorm1d(10)

      def forward(self, input):
            output = self.main(input)
            output = self.linear(output.view(input.size()[0], -1))
            if self.top_bn:
                  output = self.bn(output)
            return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

def get_model(model_type, input_size, output_size):
    hidden_sizes = [1200, 600, 300, 150]

    if model_type == 'Linear':
        # Linear Model
        model = nn.Sequential(nn.Linear(input_size, output_size))
    elif model_type == '4-layer':
        # 3-layer model
        model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[2], output_size))
    elif model_type == 'myConvNet':
        # ConvNet with input size 64*64 for celebA
        model = myConvNet(output_size=output_size)
    else:
        model = svhnCNN()

    return model

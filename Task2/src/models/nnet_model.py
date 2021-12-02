import torch
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.nn import CrossEntropyLoss
import numpy as np
from .base_model import MyModel
from ..losses.MyLoss import MyLoss


class MyNNet(MyModel):
    def __init__(self, core_management):
        super(MyNNet, self).__init__(core_management)

        self.hideen_dim_0 = None
        self.hideen_dim_1 = 64
        self.hideen_dim_2 = 16

        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None

        self.regularization = 1e-4  # 这个值越大，最终得到得分布值越接近于恒等于mean

        self.initialized = False

    def initialization(self, input_dimension):
        self.input_dimension = input_dimension
        self.total_epoch = 50000
        self.hideen_dim_0 = self.input_dimension

        self.layer0 = torch.nn.Linear(self.input_dimension, self.hideen_dim_0, bias=True).to(self.device)
        kaiming_uniform_(self.layer0.weight, nonlinearity='leaky_relu')
        self.layer1 = torch.nn.Linear(self.hideen_dim_0, self.hideen_dim_1, bias=True).to(self.device)
        kaiming_uniform_(self.layer1.weight, nonlinearity='leaky_relu')
        self.layer2 = torch.nn.Linear(self.hideen_dim_1, self.hideen_dim_2, bias=True).to(self.device)
        kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        self.layer3 = torch.nn.Linear(self.hideen_dim_2, 4, bias=True).to(self.device)
        xavier_uniform_(self.layer3.weight)

        # self.loss = R2Score(self.core_management.train_Y, self.core_management.device, self.regularization)
        self.loss = F.cross_entropy
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)

        self.initialized = True

    def forward(self, input_):
        output0 = F.leaky_relu(self.layer0(input_))

        output1 = F.leaky_relu(self.layer1(output0))

        output2 = F.relu(self.layer2(output1))

        final_activate = torch.nn.Softmax(dim=1)
        predicted_y = final_activate(self.layer3(output2))
        predicted_y = final_activate(predicted_y)

        return predicted_y

    def compute_loss(self, predicted_y, train_Y):
        return self.loss(predicted_y, train_Y)


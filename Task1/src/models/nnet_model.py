import torch
import torch.nn.functional as F
import numpy as np
from .base_model import MyModel
from ..losses.MyLoss import MyLoss


class MyNNet(MyModel):
    def __init__(self, core_management):
        super(MyNNet, self).__init__(core_management)

        self.hideen_dim_0 = 5
        # self.hideen_dim_1 = 128
        # self.hideen_dim_2 = 256

        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None

        self.regularization = 1e-4  # 这个值越大，最终得到得分布值越接近于恒等于mean

        self.initialized = False

    def initialization(self):
        self.batch_size, self.input_dimension = self.core_management.full_X.shape
        self.input_dimension = self.input_dimension
        self.total_epoch = 5000

        self.layer0 = torch.nn.Linear(self.input_dimension, self.hideen_dim_0, bias=True).to(self.device)
        self.layer1 = torch.nn.Linear(self.hideen_dim_0, 1, bias=True).to(self.device)

        # self.loss = R2Score(self.core_management.train_Y, self.core_management.device, self.regularization)
        self.loss = MyLoss(regularization=self.regularization)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)

        self.initialized = True

    def forward(self, input):
        output0 = F.leaky_relu(self.layer0(input))
        # output0_halv = F.dropout(output0, p=0.2)

        predicted_y = (self.layer1(output0))

        return predicted_y

    def compute_loss(self, predicted_y, train_Y):
        return self.loss(predicted_y, train_Y)


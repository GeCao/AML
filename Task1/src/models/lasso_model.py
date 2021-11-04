import torch
import torch.nn.functional as F
from .base_model import MyModel
from ..losses.MyLoss import MyLoss


# TODO: This is not lasso, still waiting for implemented
class MyLasso(MyModel):
    def __init__(self, core_management):
        super(MyLasso, self).__init__(core_management)

        self.hideen_dim_0 = 256
        self.hideen_dim_1 = 128
        self.hideen_dim_2 = 64
        self.hideen_dim_3 = 64
        self.hideen_dim_4 = 128
        self.hideen_dim_5 = 256

        '''
        self.hideen_dim_0 = 512
        self.hideen_dim_1 = 512
        self.hideen_dim_2 = 256
        self.hideen_dim_3 = 128
        self.hideen_dim_4 = 64
        self.hideen_dim_5 = 32
        self.hideen_dim_6 = 16
        '''

        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None
        self.layer6 = None

        self.regularization = 1e-5

        self.initialized = False

    def initialization(self):
        self.batch_size, self.input_dimension = self.core_management.full_X.shape
        self.input_dimension = self.input_dimension
        self.total_epoch = 10000

        self.layer0 = torch.nn.Linear(self.input_dimension, self.hideen_dim_0).to(self.device)
        self.layer1 = torch.nn.Linear(self.hideen_dim_0, self.hideen_dim_1).to(self.device)
        self.layer2 = torch.nn.Linear(self.hideen_dim_1, self.hideen_dim_2).to(self.device)
        self.layer3 = torch.nn.Linear(self.hideen_dim_2, self.hideen_dim_3).to(self.device)
        self.layer4 = torch.nn.Linear(self.hideen_dim_3, self.hideen_dim_4).to(self.device)
        self.layer5 = torch.nn.Linear(self.hideen_dim_4, self.hideen_dim_5).to(self.device)
        self.layer6 = torch.nn.Linear(self.hideen_dim_5, 1).to(self.device)

        # self.loss = R2Score(self.core_management.train_Y, self.core_management.device, self.regularization)
        self.loss = MyLoss(regularization=self.regularization)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.1)

        self.initialized = True

    def outlier_handling(self):
        pass

    def forward(self, input):
        output0 = F.leaky_relu_(self.layer0(input))
        output0_halv = F.dropout(output0, p=0.5)

        output1 = F.leaky_relu_(self.layer1(output0_halv))

        output2 = F.leaky_relu_(self.layer2(output1))

        output3 = F.leaky_relu_(self.layer3(output2))

        output4 = F.leaky_relu_(self.layer4(output3))

        output5 = F.leaky_relu_(self.layer5(output4))

        predicted_y = F.leaky_relu_(self.layer6(output5))

        return predicted_y

    def compute_loss(self, predicted_y, train_Y):
        return self.loss(predicted_y, train_Y)


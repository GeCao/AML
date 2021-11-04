import torch
import torch.nn.functional as F
from .base_model import MyModel
from ..losses.MyLoss import MyLoss


# TODO: This is not lasso, still waiting for implemented
class MyLasso(MyModel):
    def __init__(self, core_management):
        super(MyLasso, self).__init__(core_management)

        self.hideen_dim_1 = 64
        self.hideen_dim_2 = 16
        self.hideen_dim_3 = 64
        self.hideen_dim_4 = 200

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None

        self.regularization = 0.000005

        self.initialized = False

    def initialization(self):
        self.batch_size, self.input_dimension = self.core_management.full_X.shape
        self.input_dimension = self.input_dimension
        self.total_epoch = 10000

        self.layer1 = torch.nn.Linear(self.input_dimension, self.hideen_dim_1).to(self.device)
        self.layer2 = torch.nn.Linear(self.hideen_dim_1, self.hideen_dim_2).to(self.device)
        self.layer3 = torch.nn.Linear(self.hideen_dim_2, self.hideen_dim_3).to(self.device)
        self.layer4 = torch.nn.Linear(self.hideen_dim_3, self.hideen_dim_4).to(self.device)
        self.layer5 = torch.nn.Linear(self.hideen_dim_4, 1).to(self.device)

        # self.loss = R2Score(self.core_management.train_Y, self.core_management.device, self.regularization)
        self.loss = MyLoss(regularization=self.regularization)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.1)

        self.initialized = True

    def outlier_handling(self):
        pass

    def forward(self, input):
        output1 = F.leaky_relu_(self.layer1(input))
        output1_halv = F.dropout(output1, p=0.5)

        output2 = F.leaky_relu_(self.layer2(output1_halv))
        output2_halv = output2  # F.dropout(output2, p=0.5)

        output3 = F.leaky_relu_(self.layer3(output2_halv))

        output4 = F.leaky_relu_(self.layer4(output3))

        predicted_y = F.leaky_relu_(self.layer5(output4))

        return predicted_y

    def compute_loss(self, predicted_y, train_Y):
        return self.loss(predicted_y, train_Y)


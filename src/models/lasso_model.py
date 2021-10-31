import torch
import torch.nn.functional as F
from .base_model import MyModel
from ..losses.r2_score import R2Score


# TODO: This is not lasso, still waiting for implemented
class MyLasso(MyModel):
    def __init__(self, core_management):
        super(MyLasso, self).__init__(core_management)

        self.hideen_dim_1 = 200
        self.hideen_dim_2 = 50

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None

        self.regularization = 1e-4

        self.initialized = False

    def initialization(self):
        self.batch_size, self.input_dimension = self.core_management.full_X.shape
        self.input_dimension = self.input_dimension
        self.total_epoch = 2000

        self.layer1 = torch.nn.Linear(self.input_dimension, self.hideen_dim_1)
        self.layer2 = torch.nn.Linear(self.hideen_dim_1, self.hideen_dim_2)
        self.layer3 = torch.nn.Linear(self.hideen_dim_2, 1)

        # self.loss = R2Score(self.core_management.train_Y, self.core_management.device, self.regularization)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.1)

        self.initialized = True

    def outlier_handling(self):
        pass

    def forward(self, input):
        output1 = F.leaky_relu_(self.layer1(input))
        output1_halv = F.dropout(output1, p=0.5)

        output2 = F.leaky_relu_(self.layer2(output1_halv))
        output2_halv = F.dropout(output2, p=0.5)

        predicted_y = F.leaky_relu_(self.layer3(output2_halv))

        return predicted_y

    def compute_loss(self, predicted_y, train_Y):
        return self.loss(predicted_y, train_Y)


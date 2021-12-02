import torch
import torch.nn.functional as F


class MyModel(torch.nn.Module):
    def __init__(self, core_management):
        super(MyModel, self).__init__()
        self.core_management = core_management
        self.device = core_management.device

        self.batch_size = None
        self.input_dimension = None
        self.total_epoch = 0

        self.loss = None
        self.optimizer = None

        self.initialized = False

    def initialization(self):
        pass

    def forward(self, input):
        pass

    def compute_loss(self, predicted_y):
        pass

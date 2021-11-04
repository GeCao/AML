import torch
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self, regularization):
        super(MyLoss, self).__init__()
        self.regularization = regularization

    def forward(self, predicted_y, train_Y):
        return F.mse_loss(predicted_y, train_Y) + self.regularization * torch.norm(predicted_y, p=1)


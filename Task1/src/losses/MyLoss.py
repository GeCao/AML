import torch
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self, regularization):
        super(MyLoss, self).__init__()
        self.regularization = regularization

    def forward(self, predicted_y, train_Y):
        # loss_nominator = torch.sum((self.train_Y - predicted_y) * (self.train_Y - predicted_y))
        # return 1.0 - loss_nominator / self.loss_denominator
        return F.mse_loss(predicted_y, train_Y) + self.regularization * torch.norm(predicted_y, p=1)


import torch
import torch.nn.functional as F


class R2Score(torch.nn.Module):
    def __init__(self, train_Y, device, regularization):
        super(R2Score, self).__init__()
        self.train_Y = train_Y
        self.device = device
        self.regularization = regularization

        mean_y = torch.mean(self.train_Y)
        self.loss_denominator = torch.sum((self.train_Y - mean_y) * (self.train_Y - mean_y))

    def forward(self, predicted_y, train_Y):
        # loss_nominator = torch.sum((self.train_Y - predicted_y) * (self.train_Y - predicted_y))
        # return 1.0 - loss_nominator / self.loss_denominator
        return F.mse_loss(predicted_y, train_Y) + self.regularization * torch.norm(predicted_y, p=1)


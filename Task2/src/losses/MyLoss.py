import torch
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self, regularization):
        super(MyLoss, self).__init__()
        self.regularization = regularization

    def forward(self, predicted_y, train_Y):
        new_predicted_y = torch.where(predicted_y < 4.0, predicted_y, torch.tensor(1000.0).to(torch.float32).to('cuda'))
        new_predicted_y = torch.where(new_predicted_y >= 0.0, new_predicted_y, torch.tensor(1000.0).to(torch.float32).to('cuda'))
        # new_predicted_y = torch.floor(new_predicted_y)
        # new_predicted_y = new_predicted_y.to(torch.int)
        return F.mse_loss(new_predicted_y, train_Y) + self.regularization * torch.norm(new_predicted_y, p=1)


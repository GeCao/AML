import torch


class UnitGaussianNormalizer(object):
    def __init__(self, core_management, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.initialized = False

    def initialization(self, x, eps=0.00001):
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = (x).mean(0)
        self.std = (x).std(0)
        self.eps = eps

        self.initialized = True

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
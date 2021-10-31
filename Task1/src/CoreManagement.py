import os
import numpy as np
import torch

from .data_factory import DataFactory
from .log_factory import LogFactory
from .models.lasso_model import MyLasso
from .utils import *


class CoreComponent:
    def __init__(self, model='lasso', device=None):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        print("The root path of our project: ", self.root_path)
        self.device = 'cuda' if device is None else device  # choose with your preference

        model_name = 'lasso' if model is None else model  # choose with your preference
        if model_name == 'lasso':
            self.train_model = MyLasso(self)
        else:
            self.train_model = None

        self.log_factory = LogFactory(self, log_to_disk=False)
        self.data_factory = DataFactory(self)

        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None

        self.initialized = False

    def initialization(self):
        self.log_factory.initialization()
        self.log_factory.Slog(MessageAttribute.EInfo, sentences="Log Factory fully created")

        self.data_factory.initialization()

        self.train_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_train.csv"))
        self.train_X = self.data_factory.process_dataset(self.train_X)
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from X_train.csv, with shape as {}".format(self.train_X.shape))
        self.train_X = torch.autograd.Variable(torch.from_numpy(np.array(self.train_X)).float()).to(self.device)

        self.train_Y = self.data_factory.read_dataset(os.path.join(self.data_path, "y_train.csv"))
        self.train_Y = self.data_factory.process_dataset(self.train_Y)
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from y_train.csv, with shape as {}".format(self.train_Y.shape))
        self.train_Y = torch.autograd.Variable(torch.from_numpy(np.array(self.train_Y)).float()).to(self.device)

        self.test_X = self.data_factory.read_dataset(os.path.join(self.data_path, "X_test.csv"))
        self.test_X = self.data_factory.process_dataset(self.test_X)
        self.log_factory.Slog(MessageAttribute.EInfo,
                              sentences="Read data completed from X_test.csv, with shape as {}".format(self.test_X.shape))
        self.test_X = torch.autograd.Variable(torch.from_numpy(np.array(self.test_X)).float()).to(self.device)

        # self.test_Y = read_dataset(os.path.join(self.data_path, "y_test.csv"))
        # self.log_factory.Slog(MessageAttribute.EInfo,
                              # sentences="Read data completed from y_test.csv, with shape as {}".format(self.test_Y.shape))

        self.train_model.initialization()

        self.initialized = True

    def run(self):
        computed_losses = []
        for epoch in range(self.train_model.total_epoch):
            predicted_y = self.train_model(self.train_X)
            loss = self.train_model.compute_loss(predicted_y, self.train_Y)
            if epoch % 200 == 0:
                self.log_factory.Slog(MessageAttribute.EInfo, sentences="Epoch={}, while loss={}".format(epoch, loss))
                computed_losses.append(loss.detach().clone())
            self.train_model.optimizer.zero_grad()
            loss.backward()
            self.train_model.optimizer.step()
        model_evaluation(computed_losses, epoch_step=200)

    def kill(self):
        self.log_factory.kill()